"""Smart Library — ақылды кітапхана жүйесі (Streamlit веб-қосымшасы).

Іске қосу:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from core import db
from core.assistant import Assistant
from core.i18n import t
from core.search import CATEGORY_NAMES, SearchEngine, load_books

st.set_page_config(page_title="Smart Library", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------- #
# Ресурстарды бір рет жүктеу (кэш)
# ---------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Каталог жүктелуде… / Загрузка каталога…")
def get_engine() -> SearchEngine:
    return SearchEngine(load_books())


@st.cache_resource
def get_conn():
    return db.connect()


engine = get_engine()
conn = get_conn()
assistant = Assistant(engine, availability_fn=lambda b: db.available(conn, b))

# ---------------------------------------------------------------------- #
# Сессия күйі
# ---------------------------------------------------------------------- #
st.session_state.setdefault("lang", "kk")
st.session_state.setdefault("reader_id", None)
st.session_state.setdefault("reader_name", "")
st.session_state.setdefault("chat", [])
st.session_state.setdefault("flash", None)


def L() -> str:
    return st.session_state.lang


def cat_name(key: str) -> str:
    kk, ru = CATEGORY_NAMES.get(key, (key, key))
    return kk if L() == "kk" else ru


def title_of(book: dict) -> str:
    return book["title_kk"] if L() == "kk" else book["title_ru"]


# ---------------------------------------------------------------------- #
# Бүйірлік панель: тіл және кіру
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 📚 Smart Library")
    lang_choice = st.radio(
        "Тіл / Язык", options=["kk", "ru"], horizontal=True,
        format_func=lambda x: "Қазақша" if x == "kk" else "Русский",
        index=0 if L() == "kk" else 1,
    )
    if lang_choice != L():
        st.session_state.lang = lang_choice
        st.rerun()

    st.divider()
    if st.session_state.reader_id is None:
        st.markdown(f"**{t('login', L())}**")
        name = st.text_input(t("name_input", L()), key="name_field")
        if st.button(t("login_btn", L()), use_container_width=True) and name.strip():
            st.session_state.reader_id = db.get_or_create_reader(conn, name)
            st.session_state.reader_name = name.strip()
            st.rerun()
    else:
        st.success(f"{t('hello', L())}, {st.session_state.reader_name}! 👋")
        # Мерзім туралы ескертулер (напоминания о сроках)
        for loan in db.reader_loans(conn, st.session_state.reader_id):
            book = engine.get(loan["book_id"])
            due = date.fromisoformat(loan["due"])
            if loan["status"] == "borrowed" and due < date.today():
                st.error(f"{t('overdue', L())} {title_of(book)}")
            elif loan["status"] == "borrowed" and due <= date.today() + timedelta(days=2):
                st.warning(f"{t('due_soon', L())}: {title_of(book)} — {due:%d.%m}")
        if st.button(t("logout_btn", L()), use_container_width=True):
            st.session_state.reader_id = None
            st.session_state.reader_name = ""
            st.rerun()

    st.divider()
    st.caption(f"{t('engine_mode', L())}: {engine.mode}")


# ---------------------------------------------------------------------- #
# Кітап карточкасы
# ---------------------------------------------------------------------- #
def book_card(book: dict, key_prefix: str, score: float | None = None) -> None:
    lang = L()
    free = db.available(conn, book)
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"### {title_of(book)}")
            meta = f"**{t('author', lang)}:** {book['author']}"
            if book.get("year"):
                meta += f" · **{t('year', lang)}:** {book['year']}"
            meta += f" · {cat_name(book['category'])}"
            st.markdown(meta)
            st.write(book["annotation_kk"] if lang == "kk" else book["annotation_ru"])
            if score is not None:
                st.caption(f"{t('relevance', lang)}: {min(score, 1.0):.0%}")
        with c2:
            st.metric(t("available", lang), f"{free}/{book['copies']}")
            st.caption(f"{t('shelf', lang)}: {book['shelf']}")
            if st.button(t("reserve", lang), key=f"{key_prefix}_res_{book['id']}",
                         disabled=free == 0, use_container_width=True):
                if st.session_state.reader_id is None:
                    st.session_state.flash = ("warning", t("need_login", lang))
                else:
                    status = db.reserve(conn, st.session_state.reader_id, book)
                    kind = "success" if status == "ok" else "warning"
                    st.session_state.flash = (kind, t(f"reserve_{status}", lang))
                st.rerun()
            if st.session_state.reader_id is not None:
                is_fav = book["id"] in db.favorites(conn, st.session_state.reader_id)
                label = t("fav_remove", lang) if is_fav else t("fav_add", lang)
                if st.button(label, key=f"{key_prefix}_fav_{book['id']}", use_container_width=True):
                    db.toggle_favorite(conn, st.session_state.reader_id, book["id"])
                    st.rerun()


# ---------------------------------------------------------------------- #
# Негізгі бет
# ---------------------------------------------------------------------- #
st.title(t("app_title", L()))
st.caption(t("app_subtitle", L()))

if st.session_state.flash:
    kind, msg = st.session_state.flash
    getattr(st, kind)(msg)
    st.session_state.flash = None

tabs = st.tabs([t("tab_search", L()), t("tab_chat", L()), t("tab_recs", L()),
                t("tab_my", L()), t("tab_catalog", L()), t("tab_stats", L())])

# --- 1. Іздеу ---------------------------------------------------------- #
with tabs[0]:
    col_q, col_c = st.columns([3, 1])
    with col_q:
        query = st.text_input(t("search_label", L()), placeholder=t("search_placeholder", L()))
    with col_c:
        cats = [None] + list(CATEGORY_NAMES.keys())
        category = st.selectbox(t("category", L()), cats,
                                format_func=lambda c: t("all", L()) if c is None else cat_name(c))
    if query:
        results = engine.search(query, top_k=5, category=category)
        results = [r for r in results if r.score > 0.05]
        st.subheader(t("results", L()))
        if not results:
            st.info(t("no_results", L()))
        for r in results:
            book_card(r.book, "s", r.score)

# --- 2. ЖИ-кеңесші ----------------------------------------------------- #
with tabs[1]:
    st.caption(t("chat_hint", L()))
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📚"):
            st.markdown(msg["text"])
    prompt = st.chat_input(t("chat_placeholder", L()))
    if prompt:
        ans = assistant.answer(prompt)  # тілді сұраудан өзі анықтайды
        st.session_state.chat.append({"role": "user", "text": prompt})
        st.session_state.chat.append({"role": "assistant", "text": ans.text})
        st.rerun()

# --- 3. Ұсыныстар ------------------------------------------------------ #
with tabs[2]:
    st.subheader(t("recs_title", L()))
    history = db.history_book_ids(conn, st.session_state.reader_id) if st.session_state.reader_id else []
    if history:
        names = ", ".join(title_of(engine.get(i)) for i in history[:5])
        st.caption(f"{t('recs_because', L())} {names}")
        for r in engine.similar(history, top_k=5):
            book_card(r.book, "r")
    else:
        st.info(t("recs_empty", L()))
        popular_ids = [row["book_id"] for row in db.popular(conn, 5)] or [1, 3, 27, 33, 42]
        for bid in popular_ids:
            book_card(engine.get(bid), "p")

# --- 4. Менің кітаптарым ----------------------------------------------- #
with tabs[3]:
    if st.session_state.reader_id is None:
        st.info(t("need_login", L()))
    else:
        loans = db.reader_loans(conn, st.session_state.reader_id)
        if not loans:
            st.info(t("my_empty", L()))
        for loan in loans:
            book = engine.get(loan["book_id"])
            due = date.fromisoformat(loan["due"])
            with st.container(border=True):
                c1, c2, c3 = st.columns([4, 1, 1])
                status_key = "status_reserved" if loan["status"] == "reserved" else "status_borrowed"
                c1.markdown(f"**{title_of(book)}** — {book['author']}")
                line = f"{t(status_key, L())}: **{due:%d.%m.%Y}**"
                if loan["status"] == "borrowed" and due < date.today():
                    line += f"  {t('overdue', L())}"
                c1.markdown(line)
                if loan["status"] == "reserved":
                    if c2.button(t("pick_up", L()), key=f"pick_{loan['id']}"):
                        db.pick_up(conn, loan["id"])
                        st.rerun()
                else:
                    if c2.button(t("extend", L()), key=f"ext_{loan['id']}"):
                        db.extend(conn, loan["id"])
                        st.rerun()
                if c3.button(t("return", L()), key=f"ret_{loan['id']}"):
                    db.return_book(conn, loan["id"])
                    st.rerun()

# --- 5. Каталог -------------------------------------------------------- #
with tabs[4]:
    rows = [{
        "ID": b["id"],
        ("Атауы" if L() == "kk" else "Название"): title_of(b),
        t("author", L()): b["author"],
        t("year", L()): b["year"],
        t("category", L()): cat_name(b["category"]),
        t("shelf", L()): b["shelf"],
        t("available", L()): f"{db.available(conn, b)}/{b['copies']}",
    } for b in engine.books]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# --- 6. Статистика (кітапханашыға) ------------------------------------- #
with tabs[5]:
    s = db.stats(conn)
    m = st.columns(5)
    m[0].metric(t("stats_books", L()), len(engine.books))
    m[1].metric(t("stats_readers", L()), s["readers"])
    m[2].metric(t("stats_reserved", L()), s["reserved"])
    m[3].metric(t("stats_borrowed", L()), s["borrowed"])
    m[4].metric(t("stats_overdue", L()), s["overdue"])

    st.subheader(t("stats_by_cat", L()))
    by_cat = pd.Series([cat_name(b["category"]) for b in engine.books]).value_counts()
    st.bar_chart(by_cat)

    st.subheader(t("stats_popular", L()))
    pop = db.popular(conn, 5)
    if pop:
        st.table(pd.DataFrame([{"#": i + 1, ("Кітап" if L() == "kk" else "Книга"): title_of(engine.get(r["book_id"])),
                                "N": r["n"]} for i, r in enumerate(pop)]))
    else:
        st.caption("—")
