"""
db.py
-----
All SQL database logic lives here. Uses SQLite (a real SQL database,
just file-based) so the project runs with zero setup. If you want to
point this at MySQL/Postgres later, only this file needs to change —
swap sqlite3.connect(...) for e.g. mysql.connector / psycopg2 and keep
the same function signatures.
"""

import sqlite3
from datetime import datetime, date

DB_PATH = "attendance.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't exist yet."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def add_user(name):
    """Insert a new user and return their auto-generated id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (name, created_at) VALUES (?, ?)",
        (name, datetime.now().isoformat()),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def get_all_users():
    """Return list of (id, name) for every registered user."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM users ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_user_name(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def already_marked_today(user_id):
    """Prevent duplicate attendance rows for the same person, same day."""
    conn = get_connection()
    cur = conn.cursor()
    today_str = date.today().isoformat()
    cur.execute(
        """SELECT 1 FROM attendance
           WHERE user_id = ? AND DATE(timestamp) = ?
           LIMIT 1""",
        (user_id, today_str),
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def log_attendance(user_id, confidence):
    """Insert an attendance record. Returns True if a new row was written."""
    if already_marked_today(user_id):
        return False

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO attendance (user_id, timestamp, confidence) VALUES (?, ?, ?)",
        (user_id, datetime.now().isoformat(), confidence),
    )
    conn.commit()
    conn.close()
    return True


def get_attendance_for_today():
    conn = get_connection()
    cur = conn.cursor()
    today_str = date.today().isoformat()
    cur.execute(
        """SELECT u.name, a.timestamp, a.confidence
           FROM attendance a
           JOIN users u ON u.id = a.user_id
           WHERE DATE(a.timestamp) = ?
           ORDER BY a.timestamp""",
        (today_str,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
