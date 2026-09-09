# Face Recognition Attendance System

[![CI](https://github.com/luke-zhang-cs-py/Facial-Recognition-Software/actions/workflows/python-package.yml/badge.svg)](https://github.com/luke-zhang-cs-py/Facial-Recognition-Software/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)](https://www.python.org/)

A simple, local attendance system: OpenCV captures webcam frames, detects
and recognizes faces, and logs each recognized person into a SQL database
(SQLite by default — no server setup needed).

## How it works

1. **Register** — capture ~30 face photos of each person via webcam and
   create their record in the `users` table.
2. **Train** — happens automatically: the LBPH model is rebuilt as soon as a
   registration finishes, and again at startup if `dataset/` has changed since
   `trainer.yml` was written. There is still a manual button, but reaching for
   it should not be necessary.
3. **Run attendance** — webcam feed detects faces every frame; recognized
   faces get an `attendance` row inserted (once per person per day).

While the camera is on, the app tells the person in front of it what to fix —
one instruction at a time, drawn onto the video itself and mirrored in the
sidebar. "Look straight into the camera", "Move closer", "Take off sunglasses
or anything with a brim that shades your eyes". See
[Positioning guidance](#positioning-guidance).

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
| `calibration.py` | False-match rates measured on ~98k identities, and gallery-size maths |
| `fairness_benchmark.py` | Stratified benchmark: does the quality gate treat groups equally? |
| `guidance.py` | Turns a live trait read into one instruction to act on |
| `liveness.py` | Presentation-attack detection (is this a person or a photo) |
| `recognition.py` | Identification via SFace embeddings + gallery-size threshold |
| `seed_demo.py` | Enrolls well-known faces from LFW so recognition can be tested |

## Testing recognition without registering anyone

```bash
python seed_demo.py --people 40 --samples 10   # enroll from LFW
python seed_demo.py --list
python seed_demo.py --remove                   # take them all out again
```

LFW is the standard academic face-recognition benchmark, built from press
photographs of public figures and published for this kind of evaluation.
Entries are prefixed `[demo]` so they can never be mistaken for a real
person, and `--remove` deletes the database rows and the image folders.

Identify a still through the **Identify from an image** panel, or:

```bash
curl -X POST -F "image=@someone.jpg" http://127.0.0.1:5001/api/identify
```

That path is deliberately separate from the camera and is not liveness-gated.
Holding a photo up to the webcam is refused as a presentation attack, which is
correct — but it would also make still images impossible to test. Nothing in
the identify path writes to the attendance table.

### Measured effectiveness (40 enrolled, held-out LFW images)

| | |
|---|---|
| Correct identification | **92.9%** (223/240) |
| Rejected as unknown | 6.7% |
| **Misidentified** | **0.4%** (1/240) |
| Impostor rejection (300 non-enrolled) | **100%**, zero false accepts |

The failure mode is the safe one: it far more often declines to answer than
names the wrong person. Correct matches average 0.677 similarity against an
impostor mean of 0.231, so the two populations barely overlap.

### Operating range

Video-like degradations applied to held-out stills — a controlled sweep
answers "where does it stop working" better than a handful of clips. Face
*detection* held at 100% throughout; it is recognition that degrades.

| Condition | Holds until | Breaks at |
|---|---|---|
| Distance (downscale) | 0.25x (83%) | 0.15x (23%) |
| Motion blur | 9 px (83%) | 13 px (47%) |
| Lighting (gamma 0.4–2.2) | **no measurable loss** | — |
| In-plane rotation | 15° (80%) | 30° (37%) |
| JPEG compression | q20 (83%) | q10 (77%) |

Lighting invariance is the standout — the embeddings are essentially
unaffected across a five-fold gamma range. Rotation is the weakest axis, so a
tilted camera costs more than a dim room.



## Positioning guidance

`guidance.py` converts the live measurements into a single instruction, in
priority order: is there a face at all, is there exactly one, is it close
enough, is it facing forward, is it upright, is the exposure clipping, is it
sharp. The first failing check is what gets shown, on a coloured bar burned
into the video frame — people being registered are looking at the camera, not
at a sidebar. The sidebar carries the full checklist so a failure is never a
mystery.

Two deliberate choices:

- **Nothing gates on skin tone.** Exposure advice fires only on clipped
  pixels, never on average brightness. Telling someone their face is "too
  dark" because of their complexion is the same defect as the old quality
  gate ([BENCHMARK.md](BENCHMARK.md)), just phrased more politely.
- **Head coverings are not mentioned.** The honest failure is "no face
  detected", not a guess about what somebody is wearing. A hijab, turban or
  kippah does not interfere with detection and there is no reason to ask
  anyone to remove one. Brims and dark lenses genuinely occlude, so those are
  named — and only when detection is actually failing.

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

**Enrollment quality**, per person: sharpness, head pose, face size, exposure
(clipping and dynamic range), and eDifFIQA's learned 0–1 quality score. It
names the specific files to recapture and why
(`12.jpg — low quality, soft focus`).

Nothing gates on absolute brightness or contrast. Both track skin tone, so
thresholding them rejects people rather than photographs — measured at 2.15x
and 1.61x disparity across race groups. They are still reported as
diagnostics. Exposure is judged by clipping and dynamic range instead, which
is skin-tone independent: a dark face that is well lit still spans a wide
range; an underexposed one has its shadows crushed flat whoever is in it.

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

- **Age** is reported as a single year figure with a range, rather than one of
  eight wide buckets. The figure is the probability-weighted mean over all
  eight, which measured better than reading off the winning bucket (MAE
  **12.4y** vs 13.4y on 10,946 FairFace faces).

  The range is deliberately narrow — ±6 years — **and always shown with how
  often it is actually right, which is ~40%.** That pairing is the point.
  Measured coverage: ±5y → 33%, ±10y → 54%, ±15y → 69%. A tighter band is
  available by lowering `AGE_BAND_YEARS` in `calibration.py`, but it buys
  the look of precision and nothing else; the underlying error does not
  shrink because the display does.
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

### Measured, not assumed

Everything above was benchmarked against all 97,698 images of FairFace.
See **[BENCHMARK.md](BENCHMARK.md)** for the full results. The short version:

- **Detection is even** — 99.95%, widest race-group gap 0.08pp.
- **The quality gate used to be biased and was fixed.** Absolute brightness
  and contrast thresholds encoded skin tone (2.15x and 1.61x disparity) and
  flagged 38.8% of Black faces vs 18.5% of White faces as "too dark". Gating
  now uses scale-free and learned signals only; disparity is 1.24x.
- **Thresholds depend on how many people are enrolled.** A threshold swept on
  a few identities cannot see false matches. `calibration.py` carries the
  curve measured over 4.77 billion impostor pairs, and the recommendation
  scales with gallery size — 0.425 at 10 people, 0.725 at 1,000, and nothing
  sufficient past ~10,000.
- **The gender estimator fails badly for Black women (43.7%, worse than
  chance).** Leave it off unless you have a reason not to.

Re-check any of it after a change:

```bash
python fairness_benchmark.py --corpus <dir-of-parquet> --per-group 800
```

It exits non-zero if any check exceeds the disparity budget, so it works as a
CI gate.

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

## License

[MIT](LICENSE) — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup and test
conventions.
