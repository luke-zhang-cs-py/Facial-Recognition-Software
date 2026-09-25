# Code audit

Static analysis (flake8, radon), a 162-test suite, and a coverage report.

```bash
python -m pytest tests/ --cov=. --cov-report=term-missing
python -m flake8 . --select=E9,F63,F7,F82,F401,F402,F811,F841,E722 --exclude=dataset,models
python -m radon cc . -s -n C --exclude "dataset/*,models/*"
```

## Coverage

145 tests, **60%** overall, from 0% at the start of the audit.

| Module | Cover | | Module | Cover |
|---|---|---|---|---|
| `db.py` | **100%** | | `traits.py` | 68% |
| `paths.py` | **100%** | | `analytics.py` | 67% |
| `view_report.py` | **100%** | | `train_model.py` | 64% |
| `landmarks.py` | 94% | | `camera.py` | 31% |
| `liveness.py` | 94% | | `analyze_faces.py` | 19% |
| `guidance.py` | 92% | | `attendance.py` | 0% |
| `recognition.py` | 90% | | `register_user.py` | 0% |
| `calibration.py` | 84% | | | |
| `facemodels.py` | 83% | | | |
| `app.py` | 72% | | | |

What is left uncovered genuinely needs hardware. `camera.py` at 31% is its
decision logic tested and its capture loop not; `attendance.py` and
`register_user.py` are camera loops end to end. Everything that can be tested
without a webcam now is.

## Findings

### Dispensables

**Duplicate code, `face_attendance.py` -- FIXED (462 -> 64 lines).** It re-implements
**12 functions** that already exist in the modules: all 8 of `db.py`, plus
`register_user`, `load_training_data`, `run_attendance`, and `view_report`
`main`. Two copies of the schema and the attendance rules, and only one has
this year of bug fixes -- the connection-leak repair, the detection-threshold
split, the yaw clamp. **The largest maintenance liability in the repo**, and a
textbook shotgun-surgery source: every future pipeline change must be made
twice or silently diverge. *Fixed:* rewritten as one argument parser over the real modules. The
convenience of a single command is kept; the duplication is gone. A test
(`test_face_attendance_defines_no_duplicated_logic`) now fails if logic
creeps back in.

**Unused imports** in `recognition.py`, `seed_demo.py`, `tools_trials.py`.
*Fixed.*

### Bloaters

Fourteen functions over 55 lines. Worst:

| Function | Lines | Complexity |
|---|---|---|
| `analytics.lbph_analysis` | 80 | **E (34)** |
| `guidance.instruction` | 94 | **E (32)** |
| `analytics.summarize_user` | -- | **D (30)** |
| `fairness_benchmark.main` | 83 | D (26) |
| `camera._build_report` | 94 | D (25) |
| `analytics.sface_analysis` | 105 | C (19) |

`guidance.instruction` is a deliberate long if-chain -- the ordering *is* the
feature and splitting it would hide the priority it encodes -- but 32 is high
enough that a table of (predicate, severity, message) driven by a loop would
read better and test more easily. The `analytics` pair are genuine bloaters
mixing extraction, statistics and presentation.

### Abusers

**Primitive obsession / primitives for state.** `guidance` severity is a bare
string (`block`/`warn`/`ok`) compared by literal in Python, JS *and* CSS.
`liveness` verdicts (`live`/`spoof`/`unknown`) and camera modes are the same.
A typo in any of the three places fails silently. These want an `Enum` on the
Python side with the strings generated for the wire.

**Magic numbers -- FIXED.** `camera.pose_matches` had `12`, `13`, `38`, `22`
inline as yaw gates; they are now `FRONT_YAW`, `TURN_MIN`, `TURN_MAX`,
`TILT_MAX_YAW` with the reasoning above them. `analytics.summarize_user` had
`0.75`, `0.15`, `0.3` and now has `USABLE_FRACTION`, `SOFT_FRACTION`,
`DARK_FRACTION`, `UNDETECTED_FRACTION`, `FLAT_POSE_DEGREES`,
`EXPECTED_SAMPLES`.

### Couplers

**Inappropriate intimacy: `DATASET_DIR` resolved in three places -- FIXED.**
`analytics`, `train_model` and `camera` each worked out their own paths at
import time, so a test could redirect the database and still read the real
dataset off disk. That is the same split-brain that let a `dataset/` folder
reference a user id with no matching row. New `paths.py` is the single source;
`paths.use(root)` moves the whole set together so the two stores cannot drift
apart. 100% covered.

**Feature envy:** `camera._build_report` reaches into `analytics`, `db`,
`calibration` and numpy to assemble its result. It belongs in `analytics`.

### Global data

`app._analysis`, `camera.camera`, and the `_cache`/`_lock` singletons in
`facemodels`, `landmarks`, `liveness`. All deliberate -- one process owns one
camera and one copy of each model -- and all lock-guarded. Noted, not a
defect. `app._analysis` as a bare module dict mutated from a worker thread is
the shakiest.

### Naming

Consistent: `snake_case` in Python, `camelCase` at the JSON boundary, which is
the right seam. `_f()` in `landmarks.py` was uncommunicative -- **renamed to
`_plain_float()`**, which is what it does and why it exists.

## Bug classes

| Class | Found |
|---|---|
| Syntax | none -- flake8 E9/F63/F7/F82 clean |
| Runtime | **fixed:** numpy float32 500ing `/api/status`; connection leak locking the DB; ParquetFile handle blocking unlink on Windows |
| Logical | **fixed:** yaw unbounded to +/-689 degrees; traits measured on the drawn-on frame; one detection threshold doing two jobs |
| Integration | **fixed:** `camera` and `traits` ran separate detectors that could disagree |
| Out-of-bounds | **fixed:** negative crop origins in the register/attendance handlers |
| Security | see below |

**Security posture.** Everything binds to `127.0.0.1`. Endpoints are
unauthenticated, defensible only because of that binding -- the moment this is
exposed, `/api/identify` and `/api/report` leak biometric matching and
attendance records to anyone who can reach the port. SQL is parameterised
throughout; no injection surface found. `/api/identify` accepted an arbitrary upload with
no size limit, a trivial memory-exhaustion vector -- **fixed** with a 12 MB
`MAX_CONTENT_LENGTH` and a 413 handler that returns JSON rather than an HTML
error page. Verified: a 13 MB upload returns 413.

## Maintenance classification

**Corrective** -- the eight bugs above, all fixed and now covered by
regression tests.

**Adaptive** -- model weights come from third-party URLs (`fetch_models.py`,
`seed_demo.py`) and will rot. The LFW mirror already moved once, and Python
SSL rejects a CA chain that `curl` accepts, which is worked around rather than
solved.

**Perfective** -- the bloaters and primitive obsession above. None affect
behaviour; all affect the cost of the next change.

**Preventive** -- this suite. The two most valuable tests pin down bugs that
already happened: `test_a_rejected_write_does_not_lock_the_database` and
`test_landmark_metrics_are_json_serialisable`. Neither failure was reachable
by a test that only checked return values.


## Status after the audit

Fixed in this pass:

- `face_attendance.py` duplication, 462 -> 64 lines, twelve duplicated
  functions to zero
- `DATASET_DIR` split-brain, via `paths.py`
- magic numbers in `camera.pose_matches` and `analytics.summarize_user`
- `/api/identify` unbounded upload
- `_f()` renamed to `_plain_float()`
- three unused imports

Deliberately not changed, with reasons:

- **`guidance.instruction` complexity (E/32).** The long if-chain *is* the
  priority ordering, and it is the most-tested function in the project. A
  predicate table would read better; it would also make the ordering
  implicit, which is the one property that must stay obvious.
- **Primitives for state.** Severity and verdict strings cross into JS and
  CSS, so an `Enum` only helps the Python third of the problem and adds a
  translation layer at the wire. Worth doing alongside a typed API, not
  before it.
- **`camera._build_report` feature envy.** Moving it into `analytics` is
  right, but it is live code with no coverage of its own; it should move
  after it has tests, not before.

## Second pass

Three more, found by widening the net rather than by re-reading.

**A person with one usable image broke the whole analysis report.**
Leave-one-out has nothing to match a lone sample against, and that was
recorded as a similarity of `-inf` and averaged into the genuine
distribution, taking its mean and minimum with it. `-Infinity` is not valid
JSON: Python writes it and reads it back without complaint, so it looks fine
from the server, and the browser's `JSON.parse` rejects it -- the report
simply never appeared, with no field named. The unmatchable samples are now
counted and reported separately, since it is a fact about the dataset rather
than a score, and accuracy is measured over what could actually be
evaluated.

**`json_safe` did not catch it, and `/api/analysis` was not using it.**
The helper coerced numpy types and passed non-finite floats straight
through, and the one endpoint carrying the most measured numbers on it was
the one endpoint not calling the helper at all. Both fixed; a non-finite
value now degrades one field to `null` instead of the page.

**Two callers leaked a database connection.** `db.connection()` was added in
the first pass precisely because closing on the success path only turned any
single error into "database is locked" for the rest of the process.
`app.api_report` and `view_report` still opened a raw `get_connection()` and
closed it on the success path only -- running the same query, inline, in both
files, which is how the fix failed to reach them. Now one `db.get_all_attendance()`,
and a test asserts no module outside `db.py` opens a raw connection.

Also: a loop variable named `paths` shadowed the imported `paths` module
inside `lbph_analysis`. Nothing used the module after it, so nothing was
broken -- but the next person to reach for `paths.DATASET` in that function
would have got an `AttributeError` on a list. `F402` is in the flake8
selection above now, which is what found it.
