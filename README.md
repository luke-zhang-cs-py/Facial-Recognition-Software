# Face Recognition Attendance System

A simple, local attendance system: OpenCV captures webcam frames, detects
and recognizes faces, and logs each recognized person into a SQL database
(SQLite by default — no server setup needed).

## How it works

1. **Register** — capture ~30 face photos of each person via webcam and
   create their record in the `users` table.
2. **Train** — build an LBPH (Local Binary Patterns Histogram) face
   recognizer model from all registered faces.
3. **Run attendance** — webcam feed detects faces every frame; recognized
   faces get an `attendance` row inserted (once per person per day).

```
Webcam frame → Haar cascade (face detection) → LBPH recognizer (face ID)
    → confidence check → SQL INSERT into attendance table
```

## Files

| File | Purpose |
|---|---|
| `db.py` | SQLite schema + all SQL queries (users, attendance) |
| `register_user.py` | Capture face samples for a new person |
| `train_model.py` | Train the LBPH recognizer on captured faces |
| `attendance.py` | Live webcam recognition + attendance logging |
| `view_report.py` | Print all users / attendance records from the DB |

## Setup

Requires a machine with a webcam (this won't work in a cloud sandbox —
run it locally).

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Register each person (repeat per person)
python register_user.py "Jane Doe"
python register_user.py "John Smith"

# 2. Train the recognizer on everyone registered so far
python train_model.py

# 3. Run live attendance
python attendance.py
# press 'q' to quit the video window

# 4. Check what's in the database
python view_report.py
```

## Database schema (SQLite, `attendance.db`)

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    confidence REAL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

You can open `attendance.db` with any SQLite browser (e.g. DB Browser for
SQLite) or query it directly:

```bash
sqlite3 attendance.db "SELECT * FROM attendance;"
```

## Notes, limits, and tuning

- **Detection**: Haar cascades (built into OpenCV) — fast but not as
  accurate as deep-learning detectors in poor lighting or side angles.
  For better accuracy, swap in a DNN face detector or `dlib`/
  `face_recognition` (128-d embeddings) — the DB and attendance logic
  here don't need to change, just what feeds `recognizer.predict()`.
- **Recognition confidence**: LBPH's `confidence` is a *distance* —
  lower means more sure. `CONFIDENCE_THRESHOLD = 70` in `attendance.py`
  is a starting point; tighten it (e.g. 50) if you get false positives,
  loosen it if real matches are being marked "Unknown."
- **One mark per day**: `db.already_marked_today()` stops duplicate rows
  from being inserted every frame someone's face is on camera.
- **Switching to MySQL/Postgres**: only `db.py` needs to change — swap
  the `sqlite3` connection/queries for `mysql.connector` or `psycopg2`
  and keep the same function signatures; nothing else in the project
  depends on SQLite specifically.
- **Privacy**: this stores raw face images in `dataset/` and a trained
  model in `trainer.yml`. Treat both as sensitive biometric data —
  don't commit them to a public repo, and delete a person's folder +
  retrain if they ask to be removed.
