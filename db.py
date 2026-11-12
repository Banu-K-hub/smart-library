"""SQLite деректер қоры: оқырмандар, брондар, оқу тарихы.

Құпия деректер сақталмайды: тек оқырманның аты (логин) қолданылады,
құпиясөз жоқ — бұл демо-прототип.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "library.db"

LOAN_DAYS = 14      # кітап беру мерзімі
RESERVE_DAYS = 3    # бронь сақталу мерзімі
MAX_BOOKS = 5       # бір мезгілде алуға болатын кітап саны


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS readers (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name  TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS loans (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            reader_id  INTEGER NOT NULL REFERENCES readers(id),
            book_id    INTEGER NOT NULL,
            status     TEXT NOT NULL,          -- reserved | borrowed | returned
            created    TEXT NOT NULL,
            due        TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS favorites (
            reader_id  INTEGER NOT NULL,
            book_id    INTEGER NOT NULL,
            PRIMARY KEY (reader_id, book_id)
        );
        """
    )
    return conn


def get_or_create_reader(conn: sqlite3.Connection, name: str) -> int:
    name = name.strip()
    row = conn.execute("SELECT id FROM readers WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO readers(name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid


def active_count(conn: sqlite3.Connection, book_id: int) -> int:
    """Қазір бронда немесе оқырманда жүрген дана саны."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM loans WHERE book_id = ? AND status IN ('reserved','borrowed')",
        (book_id,),
    ).fetchone()
    return row["n"]


def available(conn: sqlite3.Connection, book: dict) -> int:
    return max(book["copies"] - active_count(conn, book["id"]), 0)


def reader_loans(conn: sqlite3.Connection, reader_id: int, active_only: bool = True) -> list[sqlite3.Row]:
    sql = "SELECT * FROM loans WHERE reader_id = ?"
    if active_only:
        sql += " AND status IN ('reserved','borrowed')"
    return conn.execute(sql + " ORDER BY due", (reader_id,)).fetchall()


def reserve(conn: sqlite3.Connection, reader_id: int, book: dict, today: date | None = None) -> str:
    """Кітапты брондау. Қайтарады: 'ok' | 'no_copies' | 'limit' | 'already'."""
    today = today or date.today()
    loans = reader_loans(conn, reader_id)
    if any(l["book_id"] == book["id"] for l in loans):
        return "already"
    if len(loans) >= MAX_BOOKS:
        return "limit"
    if available(conn, book) <= 0:
        return "no_copies"
    conn.execute(
        "INSERT INTO loans(reader_id, book_id, status, created, due) VALUES (?,?,?,?,?)",
        (reader_id, book["id"], "reserved", today.isoformat(), (today + timedelta(days=RESERVE_DAYS)).isoformat()),
    )
    conn.commit()
    return "ok"


def pick_up(conn: sqlite3.Connection, loan_id: int, today: date | None = None) -> None:
    """Бронды кітапты алу (демо үшін оқырман өзі басады)."""
    today = today or date.today()
    conn.execute(
        "UPDATE loans SET status='borrowed', due=? WHERE id=?",
        ((today + timedelta(days=LOAN_DAYS)).isoformat(), loan_id),
    )
    conn.commit()


def return_book(conn: sqlite3.Connection, loan_id: int) -> None:
    conn.execute("UPDATE loans SET status='returned' WHERE id=?", (loan_id,))
    conn.commit()


def extend(conn: sqlite3.Connection, loan_id: int, days: int = 7) -> None:
    row = conn.execute("SELECT due FROM loans WHERE id=?", (loan_id,)).fetchone()
    new_due = date.fromisoformat(row["due"]) + timedelta(days=days)
    conn.execute("UPDATE loans SET due=? WHERE id=?", (new_due.isoformat(), loan_id))
    conn.commit()


def history_book_ids(conn: sqlite3.Connection, reader_id: int) -> list[int]:
    """Оқырманның барлық алған/брондаған және таңдаулы кітаптары — ұсыныс үшін."""
    rows = conn.execute(
        "SELECT book_id FROM loans WHERE reader_id=? UNION SELECT book_id FROM favorites WHERE reader_id=?",
        (reader_id, reader_id),
    ).fetchall()
    return [r["book_id"] for r in rows]


def toggle_favorite(conn: sqlite3.Connection, reader_id: int, book_id: int) -> bool:
    """Таңдаулыға қосу/алып тастау. True — қосылды."""
    exists = conn.execute(
        "SELECT 1 FROM favorites WHERE reader_id=? AND book_id=?", (reader_id, book_id)
    ).fetchone()
    if exists:
        conn.execute("DELETE FROM favorites WHERE reader_id=? AND book_id=?", (reader_id, book_id))
        conn.commit()
        return False
    conn.execute("INSERT INTO favorites(reader_id, book_id) VALUES (?,?)", (reader_id, book_id))
    conn.commit()
    return True


def favorites(conn: sqlite3.Connection, reader_id: int) -> set[int]:
    rows = conn.execute("SELECT book_id FROM favorites WHERE reader_id=?", (reader_id,)).fetchall()
    return {r["book_id"] for r in rows}


def popular(conn: sqlite3.Connection, limit: int = 5) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT book_id, COUNT(*) AS n FROM loans GROUP BY book_id ORDER BY n DESC LIMIT ?", (limit,)
    ).fetchall()


def stats(conn: sqlite3.Connection) -> dict:
    q = lambda sql: conn.execute(sql).fetchone()[0]
    return {
        "readers": q("SELECT COUNT(*) FROM readers"),
        "reserved": q("SELECT COUNT(*) FROM loans WHERE status='reserved'"),
        "borrowed": q("SELECT COUNT(*) FROM loans WHERE status='borrowed'"),
        "overdue": conn.execute(
            "SELECT COUNT(*) FROM loans WHERE status='borrowed' AND due < ?", (date.today().isoformat(),)
        ).fetchone()[0],
    }
