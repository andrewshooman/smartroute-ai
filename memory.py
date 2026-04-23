import os
import sqlite3
from typing import Dict, List

from config import MEMORY_DB_PATH


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


_db_initialized = False


def init_db() -> None:
    global _db_initialized
    if _db_initialized:
        return
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                meta TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    _db_initialized = True


def load_messages() -> List[Dict[str, str]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, COALESCE(meta, '') AS meta FROM messages ORDER BY id ASC"
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "meta": r["meta"]} for r in rows]


def save_message(role: str, content: str, meta: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (role, content, meta) VALUES (?, ?, ?)",
            (role, content, meta),
        )


def clear_messages() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages")
