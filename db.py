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
from contextlib import contextmanager
from datetime import datetime, date

import paths

# Kept as a module attribute because tests and the CLI scripts patch it, but
# the default now comes from paths so it moves with dataset/ rather than
# being separately redirectable.
DB_PATH = paths.db_path()


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection(row_factory=None):
    """Borrow a connection that is always closed, even on an exception.

    Every function here used to open a connection and close it on the success
    path only. Any error in between -- a failed foreign key, a bad value --
    leaked the connection with its transaction still open, and SQLite then
    refused every later write with "database is locked". One bad row poisoned
    the whole process.

    The failure was invisible until something actually raised, which is why it
    survived until a dataset/ folder referenced a user id that no longer
    existed. Rolling back explicitly keeps a half-finished write from sitting
    on the lock.
    """
    conn = get_connection()
    if row_factory is not None:
        conn.row_factory = row_factory
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist yet."""
    with connection() as conn:
        _create_tables(conn)


def _create_tables(conn):
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


def add_user(name):
    """Insert a new user and return their auto-generated id."""
    with connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (name, created_at) VALUES (?, ?)",
            (name, datetime.now().isoformat()),
        )
        conn.commit()
        return cur.lastrowid


def get_all_users():
    """Return list of (id, name) for every registered user."""
    with connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM users ORDER BY id")
        return cur.fetchall()


def get_user_name(user_id):
    with connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return row[0] if row else None


def user_exists(user_id):
    """Used before caching traits keyed to a dataset/ folder's user id."""
    with connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE id = ? LIMIT 1", (user_id,))
        return cur.fetchone() is not None


def already_marked_today(user_id):
    """Prevent duplicate attendance rows for the same person, same day."""
    with connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT 1 FROM attendance
               WHERE user_id = ? AND DATE(timestamp) = ?
               LIMIT 1""",
            (user_id, date.today().isoformat()),
        )
        return cur.fetchone() is not None


def log_attendance(user_id, confidence):
    """Insert an attendance record. Returns True if a new row was written."""
    if already_marked_today(user_id):
        return False

    with connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO attendance (user_id, timestamp, confidence) VALUES (?, ?, ?)",
            (user_id, datetime.now().isoformat(), confidence),
        )
        conn.commit()
        return True


def get_attendance_for_today():
    with connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT u.name, a.timestamp, a.confidence
               FROM attendance a
               JOIN users u ON u.id = a.user_id
               WHERE DATE(a.timestamp) = ?
               ORDER BY a.timestamp""",
            (date.today().isoformat(),),
        )
        return cur.fetchall()


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
    with connection(sqlite3.Row) as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {TRAIT_COLUMNS} FROM sample_traits WHERE path = ? AND mtime = ?",
            (path, mtime),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def save_traits(row):
    """Insert or replace one sample's trait row. `row` is a plain dict.

    Returns False when the row references a user id that has no row in
    `users` -- a dataset/ folder copied between machines, or a database that
    was reset while the images stayed put. The trait analysis is still valid,
    it just cannot be cached against a person who does not exist, so this
    reports rather than raising.
    """
    if not user_exists(row["user_id"]):
        return False

    with connection() as conn:
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
        return True


def get_traits_for_user(user_id):
    with connection(sqlite3.Row) as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {TRAIT_COLUMNS} FROM sample_traits WHERE user_id = ? ORDER BY path",
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def delete_user(user_id):
    """Remove a person entirely: attendance, cached traits, then the user.

    The dataset/ folder is deleted by the caller — this only clears the
    database side. All three deletes share one transaction, so a failure
    part-way cannot leave attendance rows pointing at a deleted user.
    """
    with connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM sample_traits WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
