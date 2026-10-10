"""
core/db.py
----------
All SQL database logic lives here. Uses SQLite (a real SQL database,
just file-based) so the project runs with zero setup. If you want to
point this at MySQL/Postgres later, only this file needs to change —
swap sqlite3.connect(...) for e.g. mysql.connector / psycopg2 and keep
the same function signatures.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, date

from core import paths

# Resolved per call, not captured at import.
#
# paths.use() exists so a script or a test can point the whole set at another
# root and have the database and the dataset move together. A module-level
# `X = paths.x()` reads the root once, at import, and then never moves --
# which reintroduces exactly the split paths.py was written to prevent: after
# use(), the dataset is the copy and the database is still the real one.


def get_connection():
    conn = sqlite3.connect(paths.db_path(), timeout=10.0)
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
            method TEXT NOT NULL DEFAULT 'lbph',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # `confidence` is whatever the recogniser that made the mark reports: an
    # LBPH distance (lower is closer, ~0-100) or an SFace cosine similarity
    # (higher is closer, 0-1). Rows from before SFace decided attendance have
    # no `method`; every one of them was LBPH, which is what the default says.
    columns = [row[1] for row in cur.execute("PRAGMA table_info(attendance)")]
    if "method" not in columns:
        cur.execute("ALTER TABLE attendance ADD COLUMN method TEXT NOT NULL DEFAULT 'lbph'")
    _one_mark_per_day(cur)

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


# One attendance row per person per day, held by the database itself.
#
# log_attendance used to check and then insert, on two connections, so two
# processes marking the same person at the same instant could both find no
# row and both write one (notes/CODE_AUDIT.md, third pass).
#
# A unique index on the expression DATE(timestamp) rather than a `day` column.
# `timestamp` is written as datetime.now().isoformat(): local time, no
# offset, and DATE() of it is that local date unchanged -- the same day
# get_attendance_for_today compares against date.today(). The index therefore means exactly what every existing query
# means by "today", needs no ALTER TABLE or backfill, and cannot drift from
# the timestamp the way a separately written column could. (DATE(timestamp)
# is deterministic; only DATE('now') is refused in an index.) A timestamp
# DATE() cannot parse is NULL, and NULLs never collide, so such a row is
# simply not constrained -- as it was not counted before.
ONE_MARK_PER_DAY_INDEX = "attendance_one_per_person_per_day"


def _one_mark_per_day(cur):
    """Create the index, first removing duplicates an older database may hold.

    Runs from init_db on every startup, so it has to be idempotent: once the
    index exists there is nothing to de-duplicate and nothing is scanned.
    Of each person's marks on one day the earliest is kept -- by julianday(),
    not string order, since "2026-10-01 10:00" sorts before "2026-10-01T09:00"
    as text -- with the lower id breaking an exact tie. The delete and the
    index share the caller's transaction.
    """
    cur.execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
                (ONE_MARK_PER_DAY_INDEX,))
    if cur.fetchone() is not None:
        return
    cur.execute("""
        DELETE FROM attendance WHERE EXISTS (
            SELECT 1 FROM attendance AS earlier
            WHERE earlier.user_id = attendance.user_id
              AND DATE(earlier.timestamp) = DATE(attendance.timestamp)
              AND (julianday(earlier.timestamp) < julianday(attendance.timestamp)
                   OR (julianday(earlier.timestamp) = julianday(attendance.timestamp)
                       AND earlier.id < attendance.id)))
    """)
    cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {ONE_MARK_PER_DAY_INDEX} "
                "ON attendance (user_id, DATE(timestamp))")


def add_user(name):
    """Insert a new user and return their auto-generated id.

    Never an id that a dataset/ folder already claims. Folders are matched to
    users by their id prefix, and a folder outlives its user row (a reset
    database, a removed demo entry), so a reused id quietly files the old
    folder's face under the new person's name -- found in the October 2026
    audit, where two folders shared id 7 (notes/CODE_AUDIT_2026-10.md).
    """
    claimed = paths.folder_ids()
    with connection() as conn:
        cur = conn.cursor()
        if claimed:
            # AUTOINCREMENT hands out max(sqlite_sequence, max(id)) + 1, so
            # raising the sequence past every claimed id is enough.
            top = max(claimed)
            cur.execute("SELECT seq FROM sqlite_sequence WHERE name = 'users'")
            row = cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('users', ?)", (top,))
            elif row[0] < top:
                cur.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'users'", (top,))
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


def log_attendance(user_id, confidence, method="lbph"):
    """Insert an attendance record. Returns True if a new row was written,
    False if the person already has one today.

    The one-per-day rule is the unique index (see _one_mark_per_day), not a
    check before the insert. That check was already_marked_today(), removed
    with it: a check and an insert are two steps, and another process can
    write in between. ON CONFLICT DO NOTHING covers uniqueness
    only, so a user id with no user row still raises IntegrityError.

    `method` says how to read `confidence`: "sface" (a similarity, higher is
    closer) or "lbph" (a distance, lower is closer)."""
    with connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO attendance (user_id, timestamp, confidence, method) "
            "VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (user_id, datetime.now().isoformat(), confidence, method),
        )
        conn.commit()
        return cur.rowcount == 1


def get_all_attendance():
    """Every attendance row, newest first.

    Lives here rather than in the two callers that had it inline. Both of them
    opened a raw connection and closed it on the success path only -- the
    exact leak the connection() context manager above exists to prevent, in
    the two places the fix never reached. The same SQL in two files was how
    that happened.
    """
    with connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT u.name, a.timestamp, a.confidence
               FROM attendance a
               JOIN users u ON u.id = a.user_id
               ORDER BY a.timestamp DESC"""
        )
        return cur.fetchall()


def get_all_attendance_with_method():
    """Every attendance row, newest first, as (name, timestamp, confidence,
    method) -- for a reader that prints the number and so has to know which
    way round it goes."""
    with connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT u.name, a.timestamp, a.confidence, a.method
               FROM attendance a
               JOIN users u ON u.id = a.user_id
               ORDER BY a.timestamp DESC"""
        )
        return cur.fetchall()


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
    print(f"Database initialized at {paths.db_path()}")
