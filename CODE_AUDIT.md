# Code audit

Static analysis (flake8, radon), an 84-test suite, and a coverage report.

```bash
python -m pytest tests/ --cov=. --cov-report=term-missing
python -m flake8 . --select=E9,F63,F7,F82,F401,F811,F841,E722 --exclude=dataset,models
python -m radon cc . -s -n C --exclude "dataset/*,models/*"
```

## Coverage

84 tests, **45%** overall (30% before the app-route and camera-logic tests).

| Module | Cover | Note |
|---|---|---|
| `db.py` | 96% | includes the connection-leak regression |
| `liveness.py` | 94% | |
| `guidance.py` | 92% | every branch of the instruction ordering |
| `facemodels.py` | 83% | |
| `calibration.py` | 80% | pure arithmetic, easiest to get subtly wrong |
| `app.py` | 71% | routes via Flask test client |
| `traits.py` | 62% | |
| `landmarks.py` | 61% | |
| `recognition.py` | 57% | |
| `camera.py` | 30% | decision logic only; the rest needs hardware |
| `analytics.py` | 12% | **largest untested surface** |
| CLI entry points | 0% | `attendance`, `register_user`, `view_report`, `analyze_faces` |

Most uncovered lines need a webcam or a populated dataset. `analytics.py` at
12% is the real gap: 217 statements of pure computation with no hardware
dependency.

## Findings

### Dispensables

**Duplicate code, `face_attendance.py` (462 lines).** It re-implements
**12 functions** that already exist in the modules: all 8 of `db.py`, plus
`register_user`, `load_training_data`, `run_attendance`, and `view_report`
`main`. Two copies of the schema and the attendance rules, and only one has
this year of bug fixes -- the connection-leak repair, the detection-threshold
split, the yaw clamp. **The largest maintenance liability in the repo**, and a
textbook shotgun-surgery source: every future pipeline change must be made
twice or silently diverge. *Not fixed* -- deleting a deliverable is the
owner call. Recommended: delete, or reduce to a thin script that imports.

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

**Magic numbers** in `camera.pose_matches`: `12`, `13`, `38`, `22` are the yaw
gates, unnamed and repeated. Contrast `traits.py` and `calibration.py`, where
every threshold is a named constant with its measurement written above it.
`analytics.summarize_user` has the same problem with `0.75`, `0.15`, `0.3`.

### Couplers

**Inappropriate intimacy: `DATASET_DIR` is resolved at module level in three
places** (`analytics`, `train_model`, `camera`) and is not injectable. A test
can redirect the database and still read the real dataset off disk -- which is
exactly what happened while writing `test_app_routes.py`. It is the same
split-brain that let a `dataset/` folder reference a user id with no matching
row: the two stores are independently redirectable and nothing keeps them in
step. *Not fixed* -- wants a config object threaded through.

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
the right seam. `_f()` in `landmarks.py` is uncommunicative -- it rounds to a
plain Python float and should say so.

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
throughout; no injection surface found. `/api/identify` accepts an arbitrary
upload with **no size limit**, a trivial memory-exhaustion vector that wants a
`MAX_CONTENT_LENGTH`.

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
