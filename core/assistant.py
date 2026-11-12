"""ЖИ-кеңесші (чат-бот) / ИИ-консультант.

Жұмыс принципі (RAG — retrieval-augmented):
1. Сұраудың тілін анықтау (kk / ru).
2. Ниетін (intent) анықтау: сәлемдесу, кітапхана ережелері (FAQ),
   кітаптың бар-жоғын тексеру немесе кітап іздеу/ұсыну.
3. Каталогтан немесе FAQ базасынан ең жақын жауапты табу.
4. Жауапты пайдаланушы тілінде құрастыру.

Ақылы API қажет емес. Қаласаңыз, .env файлында OPENAI_API_KEY көрсетіп,
табылған кітаптар негізінде LLM жауабын қосуға болады (міндетті емес).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .search import DATA_DIR, SearchEngine
from .text import detect_lang, normalize

GREETINGS = {"сәлем", "салем", "сәлеметсіз", "сәлеметсізбе", "амансыз", "қайырлы", "привет",
             "здравствуйте", "здравствуй", "добрый", "салам", "hello", "hi"}
AVAILABILITY_MARKERS = ["бар ма", "бар ма?", "қолжетімді", "есть ли", "есть в наличии", "в наличии",
                        "имеется", "можно взять", "алуға бола ма", "қалды ма", "осталось"]
SERVICE_MARKERS = ["қашан", "уақыт", "сағат", "брондау", "брондаймын", "ұзарт", "айыппұл", "кешіктір",
                   "қайда", "қай жерде", "мекенжай", "билет", "тіркел", "тәртіб", "сөре", "қалай аламын", "неше кітап",
                   "режим", "во сколько", "когда", "продл", "штраф", "опоздан", "просроч", "адрес", "где",
                   "забронир", "бронь", "билет", "записат", "полк", "как взять", "сколько книг",
                   "выходн", "демалыс"]


@dataclass
class Answer:
    text: str
    book_ids: list[int] = field(default_factory=list)
    intent: str = "search"
    lang: str = "kk"


class Assistant:
    def __init__(self, engine: SearchEngine, availability_fn=None):
        self.engine = engine
        # availability_fn(book) -> бос дана саны (дерекқордан). Берілмесе — copies.
        self.availability_fn = availability_fn or (lambda b: b["copies"])
        with open(DATA_DIR / "faq.json", encoding="utf-8") as f:
            self.faq = json.load(f)
        faq_docs = [normalize(item["q_kk"] + " " + item["q_ru"]) for item in self.faq]
        self.faq_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.faq_matrix = self.faq_vec.fit_transform(faq_docs)

    # ------------------------------------------------------------------ #
    def _faq_match(self, query: str) -> tuple[dict | None, float]:
        sims = cosine_similarity(self.faq_vec.transform([normalize(query)]), self.faq_matrix)[0]
        best = int(sims.argmax())
        return self.faq[best], float(sims[best])

    @staticmethod
    def _has_any(q: str, markers: list[str]) -> bool:
        return any(m in q for m in markers)

    def _book_line(self, book: dict, lang: str) -> str:
        title = book["title_kk"] if lang == "kk" else book["title_ru"]
        free = self.availability_fn(book)
        if lang == "kk":
            status = f"✅ {free} дана бар, сөре {book['shelf']}" if free else "⏳ қазір бос данасы жоқ"
        else:
            status = f"✅ в наличии {free} шт., полка {book['shelf']}" if free else "⏳ сейчас нет свободных экземпляров"
        return f"**{title}** — {book['author']} ({status})"

    # ------------------------------------------------------------------ #
    def answer(self, query: str, lang: str | None = None) -> Answer:
        lang = lang or detect_lang(query)
        q = normalize(query)
        if not q:
            return Answer(self._t(lang, "empty"), intent="empty", lang=lang)

        # 1) Сәлемдесу
        words = q.split()
        if words and words[0] in GREETINGS and len(words) <= 3:
            return Answer(self._t(lang, "greet"), intent="greet", lang=lang)

        # 2) Кітапхана ережелері (FAQ)
        faq, faq_score = self._faq_match(query)
        if faq_score >= 0.45 or (self._has_any(q, SERVICE_MARKERS) and faq_score >= 0.22):
            return Answer(faq["a_kk"] if lang == "kk" else faq["a_ru"], intent="faq", lang=lang)

        # 3) Кітаптың бар-жоғын тексеру
        results = self.engine.search(query, top_k=3)
        if self._has_any(q, AVAILABILITY_MARKERS) and results:
            top = results[0].book
            text = self._t(lang, "availability") + "\n\n" + self._book_line(top, lang)
            return Answer(text, [top["id"]], intent="availability", lang=lang)

        # 4) Кітап іздеу / ұсыну
        if not results or results[0].score < 0.08:
            return Answer(self._t(lang, "not_found"), intent="not_found", lang=lang)

        llm_text = self._llm_answer(query, [r.book for r in results], lang)
        if llm_text:
            return Answer(llm_text, [r.book["id"] for r in results], intent="search_llm", lang=lang)

        lines = [self._t(lang, "found")]
        for i, r in enumerate(results, 1):
            ann = r.book["annotation_kk"] if lang == "kk" else r.book["annotation_ru"]
            lines.append(f"{i}. {self._book_line(r.book, lang)}\n   _{ann}_")
        return Answer("\n\n".join(lines), [r.book["id"] for r in results], intent="search", lang=lang)

    # ------------------------------------------------------------------ #
    def _llm_answer(self, query: str, books: list[dict], lang: str) -> str | None:
        """Міндетті емес: OPENAI_API_KEY болса, LLM табылған кітаптар негізінде жауап береді."""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        try:
            from openai import OpenAI  # type: ignore

            context = "\n".join(
                f"- {b['title_kk']} / {b['title_ru']} — {b['author']}: {b['annotation_kk']}" for b in books
            )
            lang_name = "қазақ" if lang == "kk" else "русском"
            system = (
                "Сен кітапхана кеңесшісісің. Тек берілген тізімдегі кітаптарды ұсын, жаңа кітап ойдан шығарма. "
                f"Жауапты {lang_name} тілінде, 3–5 сөйлеммен бер."
            )
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Сұрау: {query}\n\nКаталогтан табылған кітаптар:\n{context}"},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content
        except Exception:
            return None  # LLM қолжетімсіз болса, жергілікті жауаппен жалғасамыз

    # ------------------------------------------------------------------ #
    @staticmethod
    def _t(lang: str, key: str) -> str:
        texts = {
            "greet": ("Сәлеметсіз бе! Мен Smart Library кеңесшісімін. Кітапты тақырыбы, авторы немесе "
                      "мазмұны бойынша сұраңыз, мысалы: «қазақ тарихы туралы роман» немесе «Python үйренуге кітап».",
                      "Здравствуйте! Я консультант Smart Library. Спросите книгу по теме, автору или содержанию, "
                      "например: «роман об истории Казахстана» или «книга для изучения Python»."),
            "empty": ("Сұрағыңызды жазыңыз.", "Напишите ваш вопрос."),
            "found": ("Сізге мына кітаптар сәйкес келуі мүмкін:", "Вам могут подойти эти книги:"),
            "availability": ("Каталогтан табылған кітап:", "Найденная книга в каталоге:"),
            "not_found": ("Өкінішке қарай, каталогтан сәйкес кітап таппадым. Сұрауды басқаша жазып көріңіз.",
                          "К сожалению, подходящей книги в каталоге не нашлось. Попробуйте сформулировать иначе."),
        }
        kk, ru = texts[key]
        return kk if lang == "kk" else ru
