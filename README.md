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
| `camera.py` | Shared webcam manager used by the web UI |
| `app.py` | Flask web UI — all four steps in the browser |
| `facemodels.py` | Lazy loader for the pretrained analysis models |
| `traits.py` | Per-face feature extraction (quality, pose, embedding, demographics) |
| `analytics.py` | Dataset-wide enrollment quality + threshold sweeps |
| `analyze_faces.py` | CLI report — the console version of the Analysis panel |
| `fetch_models.py` | Downloads the pretrained weights into `models/` |

There are two ways to drive the same pipeline: the **CLI scripts** above, or
the **web UI** (`app.py`). They share `db.py` and `train_model.py` and read
and write the same `dataset/`, `trainer.yml`, and `attendance.db`, so you can
mix and match. Only run one at a time — they compete for the one webcam.

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

## Web UI

The same four steps, in a browser:

```bash
python app.py
# then open http://127.0.0.1:5001
```

Start the camera, type a name and capture 30 samples, hit **Retrain model**,
then **Start attendance**. Registered users, today's attendance, and a live
activity log update in the sidebar once a second.

The webcam is opened by the *server* process, not by the browser — OpenCV
annotates each frame and Flask streams them out as MJPEG. That means the
machine running `app.py` must be the machine with the camera, which is the
same constraint the CLI scripts have. It also means this is not something to
deploy: it binds to `127.0.0.1` on purpose, since it exposes a live camera
feed and everyone's attendance records.

| Endpoint | Purpose |
|---|---|
| `GET /` | dashboard |
| `GET /video_feed` | MJPEG stream of annotated frames |
| `GET /api/status` | camera state, mode, capture progress, recent events |
| `POST /api/camera/start` · `/stop` | open / release the camera |
| `POST /api/register` | `{name}` → start capturing samples |
| `POST /api/train` | rebuild `trainer.yml` from `dataset/` |
| `POST /api/attendance/start` · `/stop` | toggle recognition + logging |
| `GET /api/report` | users + today's and all-time attendance |
| `GET /api/models` | which pretrained analysis models are present |
| `POST /api/analysis/start` | kick off a dataset scan on a worker thread |
| `GET /api/analysis` | scan progress, then the finished report |
| `POST /api/traits` | `{enabled}` → toggle the live trait readout |

## Face trait analysis

Optional layer that measures what is actually in your enrolled samples, so
enrollment problems and threshold choices stop being guesswork.

```bash
python fetch_models.py     # ~134 MB of pretrained weights, once
python analyze_faces.py    # full report
```

Or open the **Analysis** panel in the web UI, which runs the same thing on a
background thread with a progress bar. The **Live face traits** sidebar shows
the same measurements for whoever is in front of the camera right now, read
off the full-resolution colour frame.

### What it measures

**Enrollment quality**, per person: sharpness, brightness, contrast, face
size in pixels, head pose, and eDifFIQA's learned 0–1 quality score. It names
the specific files to recapture and why (`13.jpg — too dark, flat contrast`).

Blur is judged *relative to that person's own samples*, not against a fixed
number. Laplacian variance has no absolute meaning — it scales with camera,
face, and crop — so one person's sharp sample can measure 1100 while
another's measures 90. A fixed cutoff either misses real blur or condemns a
whole enrollment.

**Recognition analytics**: how separable the enrolled people actually are,
and what confidence threshold your data supports. `attendance.py` ships
`CONFIDENCE_THRESHOLD = 70` as a guess; this replaces it with a sweep of
measured accept and false-match rates, and names the most confusable pair of
people. Both recognisers are scored held-out — LBPH by 5-fold
cross-validation, SFace by leave-one-out — so no image is ever scored by a
model that already saw it.

**Face embeddings** (SFace, 128-d): unlike LBPH's histogram distance, these
live in a metric space, so cosine similarity is comparable across people and
thresholds transfer between datasets. This is what makes the separability
numbers meaningful.

**Demographic estimates**: an 8-bucket age estimate and a gender estimate,
both from Levi & Hassner (2015). Read the caveats below before using either.

### Caveats that matter

- **Age** is coarse and dated. Being off by a whole bucket is common,
  especially outside 25–45. Reported with its full probability distribution,
  because the margin is the interesting part — a 0.34/0.31 split is a coin
  flip wearing a label.
- **Gender** is a binary classifier guessing at apparent presentation from
  pixels. It is not a statement about anyone's identity, and it is
  materially less accurate for some groups than others. Treat it as weak
  evidence or leave it off.
- Both classifiers have a softmax over a fixed label set, so they return a
  confident label for *anything* — including a black frame. The live readout
  therefore suppresses them entirely when no face is detected.
- `dataset/` stores greyscale crops, but all three DNN models expect colour.
  Live camera reads are more reliable than re-analysed stored samples, and
  anything derived from a grey source is tagged as such.
- This measures **image and recogniser properties**. Inferring character,
  personality, honesty, or intent from face geometry is physiognomy; it does
  not work, and nothing here does it.

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
