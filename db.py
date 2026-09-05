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

    # Cache of per-image trait analysis (see traits.py). Keyed by file path
    # plus mtime so an edited or replaced sample is re-analysed automatically.
    # Purely a cache: deleting this table costs time, not data.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sample_traits (
            path TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            mtime REAL NOT NULL,
            sharpness REAL,
            brightness REAL,
            contrast REAL,
            quality REAL,
            face_px INTEGER,
            yaw REAL,
            roll REAL,
            detected INTEGER,
            flags TEXT,
            embedding BLOB,
            age_label TEXT,
            age_conf REAL,
            gender_label TEXT,
            gender_conf REAL,
            analyzed_at TEXT NOT NULL,
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


# ---------------------------------------------------------------------------
# Trait cache (see traits.py / analytics.py)
# ---------------------------------------------------------------------------

TRAIT_COLUMNS = (
    "path, user_id, mtime, sharpness, brightness, contrast, quality, face_px, "
    "yaw, roll, detected, flags, embedding, age_label, age_conf, gender_label, "
    "gender_conf, analyzed_at"
)


def get_cached_traits(path, mtime):
    """Return the cached row for this exact file version, or None."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        f"SELECT {TRAIT_COLUMNS} FROM sample_traits WHERE path = ? AND mtime = ?",
        (path, mtime),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def save_traits(row):
    """Insert or replace one sample's trait row. `row` is a plain dict."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"""INSERT OR REPLACE INTO sample_traits ({TRAIT_COLUMNS})
            VALUES ({', '.join('?' * 18)})""",
        (row["path"], row["user_id"], row["mtime"], row["sharpness"],
         row["brightness"], row["contrast"], row["quality"], row["face_px"],
         row["yaw"], row["roll"], row["detected"], row["flags"],
         row["embedding"], row["age_label"], row["age_conf"],
         row["gender_label"], row["gender_conf"], datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_traits_for_user(user_id):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        f"SELECT {TRAIT_COLUMNS} FROM sample_traits WHERE user_id = ? ORDER BY path",
        (user_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def delete_user(user_id):
    """Remove a person entirely: attendance, cached traits, then the user.

    The dataset/ folder is deleted by the caller — this only clears the
    database side. Ordered so the foreign keys stay satisfied throughout.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM sample_traits WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
