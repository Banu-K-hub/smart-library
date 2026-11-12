"""Мағыналық іздеу модулі / Модуль смыслового поиска.

Екі деңгейлі іздеу:
1) TF-IDF (символдық n-грамдар) — әрқашан жұмыс істейді, интернетсіз де.
   Символдық n-грамдар қазақ тілінің жалғауларына төзімді:
   "тарихы", "тарихи", "тарихтағы" сөздерінің ортақ бөлігі "тарих" сәйкес келеді.
2) LaBSE нейрондық моделі (sentence-transformers) — орнатылған болса,
   сұрау мен кітап сипаттамасын мағынасы бойынша салыстырады (109 тіл, қазақ тілі де бар).

Қорытынды балл = гибридті балл (екі әдістің салмақталған қосындысы).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text import content_words, normalize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CATEGORY_NAMES = {
    "kz_lit": ("Қазақ әдебиеті", "Казахская литература"),
    "epic": ("Батырлар жыры, эпос", "Эпос, героические сказания"),
    "world": ("Әлем әдебиеті", "Мировая литература"),
    "ru_lit": ("Орыс әдебиеті", "Русская литература"),
    "children": ("Балалар әдебиеті", "Детская литература"),
    "history": ("Тарих", "История"),
    "science": ("Ғылыми-көпшілік", "Научно-популярная"),
    "self": ("Өзін-өзі дамыту", "Саморазвитие"),
    "it": ("Бағдарламалау және ЖИ", "Программирование и ИИ"),
}


# "Қазақ халық жыры" сияқты автор жолындағы жалпы сөздер автор ретінде саналмайды
_GENERIC_AUTHOR_WORDS = {"қазақ", "халық", "жыры", "ауыз", "әдебиеті"}


@dataclass
class SearchResult:
    book: dict
    score: float


def load_books(path: Path | None = None) -> list[dict]:
    with open(path or DATA_DIR / "books.json", encoding="utf-8") as f:
        return json.load(f)


def book_document(book: dict) -> str:
    """Кітаптың іздеуге арналған толық мәтіні (екі тілде)."""
    cat_kk, cat_ru = CATEGORY_NAMES.get(book["category"], ("", ""))
    parts = [
        book["title_kk"], book["title_ru"], book["author"],
        book["annotation_kk"], book["annotation_ru"], book.get("keywords", ""),
        cat_kk, cat_ru,
    ]
    return normalize(" ".join(parts))


class SearchEngine:
    """Кітаптарды мағынасы бойынша іздейтін қозғалтқыш."""

    def __init__(self, books: list[dict], use_neural: bool | None = None):
        self.books = books
        self.docs = [book_document(b) for b in books]

        # 1) TF-IDF: символдық n-грамдар (2–5) — жалғауларға төзімді
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.char_matrix = self.char_vec.fit_transform(self.docs)
        # Сөз деңгейіндегі TF-IDF — нақты сөздер сәйкес келгенде балл қосады
        self.word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        self.word_matrix = self.word_vec.fit_transform(self.docs)

        # 2) Нейрондық модель (міндетті емес)
        self.model = None
        self.embeddings = None
        if use_neural is None:
            use_neural = os.getenv("SMART_LIBRARY_NEURAL", "1") == "1"
        if use_neural:
            self._try_load_neural()

    # ------------------------------------------------------------------ #
    def _try_load_neural(self) -> None:
        """LaBSE моделін жүктеуге тырысады; болмаса TF-IDF-пен жалғасады."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer("sentence-transformers/LaBSE")
            texts = [f"{b['title_kk']}. {b['annotation_kk']} {b['annotation_ru']}" for b in self.books]
            self.embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception:  # кітапхана орнатылмаған немесе интернет жоқ
            self.model = None
            self.embeddings = None

    @property
    def mode(self) -> str:
        return "LaBSE + TF-IDF" if self.model is not None else "TF-IDF"

    # ------------------------------------------------------------------ #
    def _lexical_scores(self, query: str) -> np.ndarray:
        q = " ".join(content_words(query)) or normalize(query)
        char_s = cosine_similarity(self.char_vec.transform([q]), self.char_matrix)[0]
        word_s = cosine_similarity(self.word_vec.transform([q]), self.word_matrix)[0]
        return 0.7 * char_s + 0.3 * word_s

    def _neural_scores(self, query: str) -> np.ndarray | None:
        if self.model is None:
            return None
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return (self.embeddings @ q[0]).astype(float)

    def _title_author_boost(self, query: str) -> np.ndarray:
        """Сұрауда кітап атауы немесе автор аталса — қосымша балл."""
        qn = normalize(query)
        words = content_words(query)
        boost = np.zeros(len(self.books))
        for i, b in enumerate(self.books):
            for title in (b["title_kk"], b["title_ru"]):
                t = normalize(title)
                if t and t in qn:
                    boost[i] += 0.5
            # автордың тегі (фамилиясы) сұрауда кездессе
            for part in normalize(b["author"]).split():
                if part in _GENERIC_AUTHOR_WORDS:
                    continue
                if len(part) >= 4 and any(w.startswith(part[:5]) for w in words):
                    boost[i] += 0.3
                    break
        return boost

    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 5, category: str | None = None) -> list[SearchResult]:
        if not normalize(query):
            return []
        lex = self._lexical_scores(query)
        neu = self._neural_scores(query)
        scores = lex if neu is None else 0.4 * lex + 0.6 * neu
        scores = scores + self._title_author_boost(query)

        order = np.argsort(-scores)
        results = []
        for idx in order:
            book = self.books[idx]
            if category and book["category"] != category:
                continue
            results.append(SearchResult(book=book, score=float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def similar(self, book_ids: list[int], top_k: int = 5, exclude: set[int] | None = None) -> list[SearchResult]:
        """Берілген кітаптарға ұқсас кітаптар (ұсыныс жүйесі үшін)."""
        exclude = set(exclude or set()) | set(book_ids)
        idx = [i for i, b in enumerate(self.books) if b["id"] in book_ids]
        if not idx:
            return []
        sims = cosine_similarity(self.char_matrix[idx], self.char_matrix).mean(axis=0)
        sims = np.asarray(sims).ravel()
        if self.embeddings is not None:
            profile = self.embeddings[idx].mean(axis=0)
            sims = 0.4 * sims + 0.6 * (self.embeddings @ profile)
        # бірдей санаттағы кітаптарға шағын бонус
        cats = {self.books[i]["category"] for i in idx}
        sims = sims + np.array([0.05 if b["category"] in cats else 0.0 for b in self.books])
        out = []
        for i in np.argsort(-sims):
            if self.books[i]["id"] in exclude:
                continue
            out.append(SearchResult(book=self.books[i], score=float(sims[i])))
            if len(out) >= top_k:
                break
        return out

    def get(self, book_id: int) -> dict | None:
        return next((b for b in self.books if b["id"] == book_id), None)
