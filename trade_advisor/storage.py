"""
决策存储模块 — 基于 SQLite 持久化历史决策
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_NAME = Path(__file__).parent / "decisions.db"


def get_conn():
    conn = sqlite3.connect(str(DB_NAME))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            strategy TEXT NOT NULL,
            action TEXT NOT NULL DEFAULT 'buy',
            code TEXT NOT NULL,
            name TEXT DEFAULT '',
            price REAL DEFAULT 0,
            shares INTEGER DEFAULT 0,
            reason TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            strategy TEXT NOT NULL,
            full_result TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()
    conn.close()


def save_decision(date: str, strategy: str, action: str, code: str,
                  name: str = "", price: float = 0, shares: int = 0,
                  reason: str = "", notes: str = ""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO decisions (date, strategy, action, code, name, price, shares, reason, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (date, strategy, action, code, name, price, shares, reason, notes)
    )
    conn.commit()
    conn.close()


def save_decision_batch(decisions: list):
    """批量保存"""
    conn = get_conn()
    conn.executemany(
        "INSERT INTO decisions (date, strategy, action, code, name, price, shares, reason, notes) "
        "VALUES (:date, :strategy, :action, :code, :name, :price, :shares, :reason, :notes)",
        decisions
    )
    conn.commit()
    conn.close()


def save_decision_log(date: str, strategy: str, full_result: dict, notes: str = ""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO decision_logs (date, strategy, full_result, notes) VALUES (?, ?, ?, ?)",
        (date, strategy, json.dumps(full_result, ensure_ascii=False, default=str), notes)
    )
    conn.commit()
    conn.close()


def get_decisions(limit: int = 200, offset: int = 0):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM decisions ORDER BY date DESC, id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_decisions_by_date(date: str):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM decisions WHERE date = ? ORDER BY id", (date,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_decision_logs(limit: int = 100):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM decision_logs ORDER BY date DESC, id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_decision(did: int, **kwargs):
    allowed = {"action", "code", "name", "price", "shares", "reason", "notes", "date", "strategy"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [did]
    conn = get_conn()
    conn.execute(f"UPDATE decisions SET {sets} WHERE id = ?", vals)
    conn.commit()
    conn.close()


def delete_decision(did: int):
    conn = get_conn()
    conn.execute("DELETE FROM decisions WHERE id = ?", (did,))
    conn.commit()
    conn.close()


def clear_decisions():
    conn = get_conn()
    conn.execute("DELETE FROM decisions")
    conn.commit()
    conn.close()
