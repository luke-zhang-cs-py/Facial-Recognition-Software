# Code audit

## 2026-10-10: fourth pass

The fourth pass: the third pass's open items, coverage of the code that was
never counted, then making the tests prove more than "this line ran". Suite:
**443 pass, 3 skip before (446 tests); 824 pass, 3 skip after (827 tests)** on
this machine. Under CI's conditions (no weights, no `dataset/`, no pyarrow)
it is **809 pass, 18 skip**. Each bug fix below came with a test that fails
on the code as it was.

```bash
python -m coverage run -m pytest -q -p no:cacheprovider
python -m coverage report -m
python -m flake8 . --select=E9,F63,F7,F82,F401,F402,F811,F841,E722 --exclude=dataset,models,.venv
HYPOTHESIS_PROFILE=thorough python -m pytest tests/test_properties.py   # 3,000 examples a property
python tools/refresh_figures.py      # keeps docs/index.html, README, CONTRIBUTING true
```

The offline figures were measured in a clean copy of the working tree
(`git ls-files -co --exclude-standard`, so no `models/` or `dataset/`) with
pyarrow blocked, which is what the push job has. "With the weights" is that
copy again with `models/` added, combined with the offline data the way
`coverage-with-weights.yml` does it.

### Bugs fixed

| # | Class | Where | What happened | Fix | Test |
|---|---|---|---|---|---|
| 1 | design (shotgun surgery) | `pipeline/camera.py`, `cli/attendance.py` | Each front end ran its own copy of "score, identify, vote, mark". The third pass's two liveness fixes had to be made in both, and the next one could have landed in only one. | `decision.decide()` is that sequence once. Both front ends call it and only draw and report the result. | `test_both_front_ends_decide_through_the_same_function`, the `test_decide_*` tests in `test_fourth_pass_phase1.py` |
| 2 | data integrity (race) | `core/db.py` `log_attendance` | Check-then-insert on two connections. Two processes marking the same person at the same instant could both find no row and both write one. | `UNIQUE(user_id, DATE(timestamp))` as an expression index, and `INSERT ... ON CONFLICT DO NOTHING`. `init_db` migrates an old database first, keeping each person's earliest mark of the day (by `julianday`, not string order). `already_marked_today` is gone. | `test_two_connections_marking_one_person_at_once_write_one_row`, `test_the_database_itself_refuses_a_second_mark_on_one_day`, `test_the_migration_keeps_the_earliest_mark_of_each_day`, `test_log_attendance_keeps_its_return_values` |
| 3 | consistency | `pipeline/train_model.py`, `core/paths.py` | Training read a folder's id with `int()`, which takes `" 5"` and `"-1"`. `folder_ids` used `isdigit()`, which refuses those but takes `"²"`, which `int()` then rejects. The two disagreed about which folders hold a person. | `paths.folder_id()`: ASCII digits before the first underscore, or no id. Both use it. | `test_one_rule_decides_a_folders_id`, `test_training_and_folder_ids_agree_on_every_folder` |
| 4 | input validation | `app.py` `/api/traits`, `/api/analysis/start` | The string `"false"` is truthy. `{"enabled": "false"}` turned the trait panel on, and `{"refresh": "false"}` re-read every image instead of using the cache. | `json_flag()` accepts a real boolean only. Anything else is a 400. | `test_traits_toggle_refuses_anything_but_a_boolean`, `test_traits_toggle_passes_a_boolean_through`, `test_analysis_start_refuses_a_refresh_that_is_not_a_boolean` |
| 5 | input validation | `cli/face_attendance.py` | `register` called `register_user()` directly and skipped `main()`'s strip-and-refuse check, so `register "  "` created a user called `"  "` and trained on nothing. | Goes through `register_user.main(["--", name])`. | `test_register_refuses_a_name_register_user_refuses`, `test_main_refuses_a_blank_or_flag_like_name` |
| 6 | data integrity | `cli/seed_demo.py` | Only the `--people` path passed the already-enrolled names to the picker, so `--keep --all-with N` enrolled everyone already enrolled a second time, under a second id. | The exclusion is worked out once, for both paths. | `test_keep_with_all_with_does_not_enroll_anyone_twice` |
| 7 | output format | `analysis/fairness_benchmark.py` `--json` | An infinite disparity (best group at 0%) was written as the bare token `Infinity`, which no strict JSON parser accepts. `default=float` also wrote booleans as `1.0`. | `json_ready()` turns non-finite numbers into null and numpy scalars into Python values. `allow_nan=False`. | `test_an_infinite_disparity_is_written_as_valid_json`, `test_json_ready` |
| 8 | measurement | `analysis/fairness_benchmark.py` | An image the decoder could not read counted as a detection failure, and as an unflagged image, for its group. A few corrupt files in one group read as a biased detector. | Only analysed images are in any denominator. The report records `analysed`. | `test_images_that_could_not_be_analysed_are_not_detection_failures` |
| 9 | workflow | `cli/seed_demo.py` `load_lfw` | pyarrow, which is not in `requirements.txt`, was imported before the corpus check. A fresh install got `ImportError` instead of the download instructions. | Check for the corpus first. A missing pyarrow is then a one-line `pip install pyarrow` message, not a traceback. | `test_load_lfw_without_the_corpus_says_how_to_get_it`, `test_load_lfw_with_the_corpus_but_no_pyarrow_says_what_to_install` |
| 10 | reporting | `pipeline/recognition.py` `refresh_gallery` | Counted every analysed sample as embedded. Without the SFace weights, seed_demo printed "N samples embedded" over a gallery that could recognise nobody. | Counts only samples that came out with an embedding. seed_demo names the missing weights. | `test_refresh_gallery_does_not_count_a_sample_with_no_embedding` |
| 11 | functional (found fixing 10) | `pipeline/recognition.py` `refresh_gallery` | Skipped every cached sample. A sample analysed before the weights were fetched was cached with no embedding and then skipped for good, so after `fetch_models` that person never got a centroid and SFace never decided for them. | A cached row with no embedding is redone once the SFace weights are present (`use_cache=False`), and left alone until then. | `test_refresh_gallery_embeds_samples_cached_before_the_weights_arrived`, guarded by `test_refresh_gallery_without_the_weights_does_not_redo_cached_samples` |

**Hypothesis** (`tests/test_properties.py`) found nothing to fix. All ten
properties held at the default 100 examples and at 3,000 examples each. The
properties:

- The calibrated threshold never loosens as the gallery grows, and an unreachable target never becomes reachable.
- A reachable threshold meets its target and is the loosest that does.
- Gallery risk grows with gallery size.
- `guidance.instruction` returns exactly one instruction, equal to the first rule in `RULES` that fires, or Ready when none does.
- Worsening the lowest-priority input never changes an instruction already given.
- `json_safe` never lets NaN or Infinity through, at any nesting of dicts, lists, tuples, numpy scalars and arrays (0-d included), and leaves plain finite data unchanged.
- Under any sequence of `follow`/`push`/`reset`, `LivenessVote` says live only on passing frames pushed since it last started over.
- A photograph, every frame below the threshold, never passes on the previous person's frames, whether through the vote directly or through `decision.decide`.

### Security

- **.gitignore.** Now ignores video files (`*.mp4`, `*.mov`, `*.avi`, `*.mkv`, `*.webm`): a presentation-attack session or footage for `tools/sample_frames.py` records a real person, the same as an image. Also ignored: `.hypothesis/`, mutmut's `mutants/`, and the parallel coverage files `.coverage.*` and `coverage.json`. Nothing tracked is newly ignored (`git ls-files -ci --exclude-standard` is empty). No face image, embedding, database or `trainer.yml` is tracked or untracked-but-unignored. `tests/golden/analytics_golden.json` is computed from grey squares.
- **Liveness measurement.** The protocol in `notes/BENCHMARK.md` uses consenting adults only, and only a participant's own face as the attack on them. Recordings stay off the repository and are deleted after analysis or on request. The CSV `tools/pad_eval.py` reads holds outcomes keyed by participant code, never images or names.
- **Workflows.** All three now run with `permissions: contents: read`. The weights job downloads only through `cli.fetch_models`, which refuses any file whose SHA-256 differs from its pin. `models/` is cached, never uploaded as an artifact. The artifacts hold coverage data (file paths and line numbers) and mutmut results, nothing derived from a face. Actions are pinned to major tags, not commit SHAs (see Left for later).
- **Tests.** Still no network: the property tests generate their own inputs, and `test_cov_cli_small.py` refuses `urlopen` outright. Real-face tests still skip without a sample frame.
- No secrets in tracked files. The cross-site, DNS-rebinding and `esc()` guards are intact. Nothing new builds HTML from a name.

### Checklist

- **Couplers.** Fixed: the per-frame sequence (bug 1), the third pass's "Left for later".
- **Bloaters.** Fixed: `analytics.summarize_user` and `lbph_analysis` mixed extraction, statistics and presentation. Each is now three functions, and `tests/golden/analytics_golden.json`, captured before the split, proves the output did not change. `camera.py` is 865 lines (878), with the decision moved out. Left: `camera.py` is still one `CameraManager`, cohesive and 99% covered.
- **Dispensables.** `db.already_marked_today` was removed with the check-then-insert it served.
- **Abusers, global data, names.** Nothing new. `decide()`'s outcomes are named constants, like `record()`'s.
- **Out of bounds.** Folder ids (bug 3). `decide()` on a box with no pixels leaves the vote alone. `pad_eval.wilson(0, 0)` returns no interval rather than dividing by zero.
- **Test smells.** `pytest.ini` now has `addopts = -rs --strict-markers`, so a misspelt marker fails instead of silently doing nothing, and every run says what skipped. The suite passes under both.
- **Lint.** CI's flake8 is widened to `E9,F63,F7,F82,F401,F402,F811,F841,E722`, the list this file's first audit used. CONTRIBUTING matches. Clean.

### Coverage

Both runs use `branch = True` and `source_dirs` (coverage 7.10+, now the
floor in `requirements.txt`). Every shipped module is counted, including
`cli/seed_demo.py`, `cli/fetch_models.py`, `cli/face_attendance.py` and
`analysis/fairness_benchmark.py`, which were omitted before the baseline and
show at 0% in it. Only `tools/` is omitted.

| | Lines | Branches | Lines + branches |
|---|---|---|---|
| Before (Phase 0, HEAD with this `.coveragerc`) | 77.5% (2,278/2,941) | 68.8% (647/940) | 75.4% |
| After, same conditions (weights and pyarrow present) | **99.2%** (2,995/3,019) | **97.7%** (954/976) | **98.8%** |
| After, offline (CI: no weights, no `dataset/`, no pyarrow) | 97.2% (2,933/3,019) | 94.5% (922/976) | 96.5% |
| After, offline + weights (what `coverage-with-weights.yml` combines) | 97.2% (2,933/3,019) | 94.9% (926/976) | 96.6% |

The baseline was re-measured from `HEAD` with this `.coveragerc` and the
weights present. It reproduces the Phase 0 report exactly (2,941 statements,
663 missed, 940 branches). `fail_under = 96` in `.coveragerc` is the offline
figure rounded down. CI runs `coverage report` after the tests, so a drop
fails the push job.

| File | Lines before | Branches before | Lines after | Branches after | Lines offline | Branches offline |
|---|---|---|---|---|---|---|
| `analysis/analytics.py` | 76.0% | 63.5% | 100% | 100% | 100% | 100% |
| `analysis/calibration.py` | 82.7% | 76.7% | 100% | 100% | 100% | 100% |
| `analysis/fairness_benchmark.py` | 0.0% | 0.0% | 99.4% | 97.9% | 77.8% | 72.9% |
| `app.py` | 75.4% | 75.0% | 100% | 100% | 100% | 100% |
| `cli/analyze_faces.py` | 47.0% | 37.5% | 100% | 100% | 100% | 100% |
| `cli/attendance.py` | 86.7% | 78.6% | 87.5% | 81.8% | 87.5% | 81.8% |
| `cli/face_attendance.py` | 0.0% | 0.0% | 100% | 90.0% | 100% | 90.0% |
| `cli/fetch_models.py` | 55.8% | 36.4% | 100% | 100% | 100% | 100% |
| `cli/register_user.py` | 80.5% | 55.0% | 100% | 100% | 100% | 100% |
| `cli/seed_demo.py` | 26.8% | 10.0% | 99.4% | 98.1% | 93.9% | 94.2% |
| `cli/view_report.py` | 100% | 100% | 100% | 100% | 100% | 100% |
| `core/corpus_paths.py` | 75.0% | (none) | 100% | (none) | 100% | (none) |
| `core/db.py` | 99.1% | 92.9% | 99.2% | 92.9% | 99.2% | 92.9% |
| `core/facemodels.py` | 93.6% | 81.2% | 100% | 100% | 100% | 100% |
| `core/paths.py` | 100% | 100% | 100% | 100% | 100% | 100% |
| `core/vision.py` | 100% | (none) | 100% | (none) | 100% | (none) |
| `pipeline/camera.py` | 98.7% | 97.1% | 98.9% | 97.1% | 98.9% | 95.6% |
| `pipeline/decision.py` | 100% | 93.8% | 100% | 96.9% | 100% | 96.9% |
| `pipeline/enrollment.py` | 79.3% | 70.8% | 100% | 100% | 100% | 100% |
| `pipeline/guidance.py` | 95.1% | 88.1% | 100% | 100% | 100% | 100% |
| `pipeline/landmarks.py` | 85.9% | 62.1% | 100% | 98.3% | 100% | 98.3% |
| `pipeline/liveness.py` | 65.8% | 50.0% | 100% | 100% | 100% | 100% |
| `pipeline/readout.py` | 86.8% | 78.6% | 100% | 100% | 100% | 100% |
| `pipeline/recognition.py` | 93.8% | 87.5% | 100% | 100% | 100% | 100% |
| `pipeline/sampleframes.py` | 100% | 97.3% | 100% | 97.3% | 86.6% | 75.7% |
| `pipeline/train_model.py` | 94.6% | 83.3% | 94.4% | 85.7% | 94.4% | 85.7% |
| `pipeline/traits.py` | 89.8% | 75.0% | 100% | 100% | 100% | 100% |

The gap between "after" and "offline" is almost entirely pyarrow. The
parquet paths in `fairness_benchmark.py`, `seed_demo.py` and
`sampleframes.py` are tested, but those tests skip without pyarrow, and CI
installs only `requirements.txt`. The weights now change little: four
branches. The networks are mocked in the `test_cov_*` files, so their
callers run either way. What the weights job adds is the real networks
themselves.

### Mutation score

Pending: run the mutmut workflow (`.github/workflows/mutation.yml`, on demand)
over `pipeline/decision.py`, `pipeline/liveness.py`, `core/db.py` and
`analysis/calibration.py`. The workflow writes the score into the job summary.

### Maintenance types

- **Corrective:** bugs 1 to 11.
- **Adaptive:** `coverage>=7.10` (`source_dirs`). mutmut 3.8's renamed keys (`source_paths`, `only_mutate`, `pytest_add_cli_args_test_selection` in place of `paths_to_mutate` and `tests_dir`). The node24 majors of `upload-artifact` (v7), `download-artifact` (v8) and `cache` (v6). Runners stay on `ubuntu-24.04` with timeouts.
- **Perfective:** seed_demo says what to install or fetch instead of failing or overstating. `tools/pad_eval.py`. A side-by-side coverage report with and without the weights. CONTRIBUTING's skip sentence now says "here" instead of claiming the run had no weights.
- **Preventive:** `fail_under`, `--strict-markers`, the wider flake8, the property tests, the weekly weights job, the mutation workflow, the wider `.gitignore`, and 25 new tests on top of Phases 1 and 2.

### Left for later

- **Kill the mutation survivors** once the workflow has run.
- **Measure liveness against real attacks** per the protocol in `notes/BENCHMARK.md` (APCER/BPCER, ISO/IEC 30107-3). Not yet measured.
- **pyarrow in CI.** A job with pyarrow installed would cover the parquet paths offline (the gap above). It is optional on purpose, so this would be a second job, not a requirement.
- **`refresh_figures.py` measures whatever machine runs it.** Here that means the weights, `dataset/` and pyarrow, so README and the page say 99% of statements where the push job measures 97.2%. It also publishes lines only, not branches. It should either measure in a clean copy, as this pass did by hand, or label which figure it is.
- **`json_safe` and `np.longdouble`.** On Linux, `longdouble.item()` returns a `longdouble`, which would recurse without end. Not verifiable here, where `longdouble` is a `double`. Nothing produces one today.
- **Actions pinned by tag, not commit SHA.** Tags can move. Pinning SHAs (with a bot to bump them) is the stricter choice.
- **Python 3.10.** The offline figure was measured on 3.14 only. CI's 3.10 run may differ by a fraction, and `fail_under` leaves 0.5pp of room.
- Still uncovered offline: the CLI's camera-opening and model-loading branches and a few loop exits (`cli/attendance.py`), five lines of `camera.py`, `train_model.py`'s unreadable-image and non-folder skips, and one branch of `db.add_user`'s id reservation.

---

## 2026-10-05: third pass

The third pass over the checklist, after the first audit below and
`notes/CODE_AUDIT_2026-10.md`. Suite: **436 pass, 3 skip before; 443 pass,
3 skip after** (7 new tests, each failing on the code as it was).

```bash
python -m coverage run --branch -m pytest
python -m coverage report -m
python tools/refresh_figures.py      # keeps docs/index.html, README, CONTRIBUTING true
```

### Bugs fixed

| # | Class | Where | What happened | Fix | Test |
|---|---|---|---|---|---|
| 1 | security (presentation attack) | `liveness.LivenessVote`, `camera.py`, `cli/attendance.py` | The liveness vote (4 of the last 7 frames must pass) was never reset between people, or when the frame emptied. Once a live person had filled the window, a photograph of somebody else held up next read "live" on its first frame and was marked present. | `LivenessVote.follow(user_id)` starts the vote over when a different person is recognised. A frame with no face resets it. A frame recognising nobody keeps it, so a frame of poor recognition does not throw the evidence away. | `test_the_vote_starts_over_for_a_new_person`, `test_the_cli_does_not_pass_a_photograph_on_the_last_persons_frames`, `test_the_camera_does_not_pass_a_photograph_on_the_last_persons_frames`, `test_the_camera_forgets_the_vote_when_nobody_is_in_front_of_it` |
| 2 | functional / integration | `cli/attendance.py` | The CLI drew the other faces' boxes and labels onto the frame first, then scored liveness and embedded the presenting face. The liveness crop is 2.7x the face, so it took in a neighbour's overlay. The web camera already decided first and drew afterwards. | Decide on the primary face first, then draw the others. | `test_the_cli_scores_and_embeds_the_face_before_drawing_anyone_else` |
| 3 | workflow | `cli/attendance.py` | The CLI loaded `trainer.yml` before anything else and quit without it, even when SFace (which never reads it) would decide. Anyone who had registered but not run `train_model` was turned away. | Ask for the SFace gallery first. The LBPH model is required only when there is no gallery. | `test_the_cli_runs_by_sface_without_a_trained_lbph_model` |
| 4 | runtime | `app.py` | `get_json(...) or {}` let a valid JSON body that is not an object (a list, a string, a number) through to `.get()`. `/api/register`, `/api/traits` and `/api/analysis/start` raised, which is a 500. `{"name": 42}` raised on `.strip()`. | `json_body()` returns the object, or `{}` for anything else. A name that is not text is a 400. | `test_a_body_that_is_not_a_json_object_is_a_400_not_a_500` |

### Security

- **.gitignore.** It covered `dataset/`, `attendance.db`, `trainer.yml` and `models/`, but not SQLite's side files (`attendance.db-journal`, `-wal`, `-shm`). Those hold the same names and attendance rows. It also did not cover face images or embeddings saved anywhere outside `dataset/`. It now ignores `*.db`, the journal files, `*.sqlite*`, and images, `.npz`/`.npy` and `.parquet` files everywhere except `docs/`. Nothing tracked is newly ignored (`git ls-files -ci --exclude-standard` is empty). No face image, database or model file is tracked, and none appears in history.
- **Local path.** `tests/test_ci_portability.py` gave its example of a machine path with the author's own Windows user name. It now uses a placeholder. The test checks the pattern, not the name.
- **Presentation attack.** See bug 1.
- No secrets in tracked files. The cross-site and DNS-rebinding guards from the last pass are intact.

### Checklist

- **Dispensables.** Fixed: `BASE_DIR` in `app.py` and `camera.py` was never read (dead code). The comment above it in `app.py` said paths resolve "against this file's folder", but `core/paths.py` resolves against the project root. Left: the `# noqa: E402` markers in `app.py` no longer suppress anything. They are harmless, and removing them would only reformat the file.
- **Bloaters.** Left: `camera.py` (878 lines, one `CameraManager`) and `analytics.py` (687). Both are large, but they are cohesive and covered (98% and 72% of statements), and splitting them is a refactor the tests do not ask for. `_handle_attendance` stays long because of its comments, not its branches.
- **Abusers.** Nothing to fix. The modes are string constants, and the decision outcomes (`LOGGED`/`ALREADY`/`NO_USER`) are named.
- **Couplers.** The camera and the CLI still each run their own copy of the per-frame "score, identify, vote, mark" sequence around `pipeline/decision.py`. Bugs 1 and 2 had to be fixed in both places (shotgun surgery). Moving the loop body into `decision.py` would end that. It is left for later because the two front ends draw and report differently.
- **Global data / magic numbers / names.** Nothing new. The thresholds and window sizes are named constants with their reasons.
- **Out of bounds.** Checked the crop clipping, the empty-face paths, `primary([])` and the dates (`DATE(timestamp)` against `date.today()`, both local time). No new issue.
- **Left, noted.** `train_model.load_training_data` parses a folder id with `int()`, which accepts `" 5"` and `"-1"`, while `paths.folder_ids` uses `isdigit()`. No code path writes such a folder. `db.log_attendance` checks and then inserts on two connections, so two processes marking the same person at the same instant could write two rows. That is harmless for a report, but a `UNIQUE(user_id, day)` constraint would close it.

### Coverage (after; `coverage run --branch`)

Total: lines **87%** (2,195 of 2,529), branches **77%** (634 of 820), combined 84%. Baseline: 84% combined.

| File | Lines | Branches |
|---|---|---|
| `analysis/analytics.py` | 76% | 63% |
| `analysis/calibration.py` | 83% | 77% |
| `app.py` | 76% | 75% |
| `cli/analyze_faces.py` | 47% | 38% |
| `cli/attendance.py` | 87% | 79% |
| `cli/register_user.py` | 80% | 55% |
| `cli/view_report.py` | 100% | 100% |
| `core/corpus_paths.py` | 75% | 100% |
| `core/db.py` | 99% | 93% |
| `core/facemodels.py` | 94% | 81% |
| `core/paths.py` | 100% | 100% |
| `core/vision.py` | 100% | 100% |
| `pipeline/camera.py` | 99% | 97% |
| `pipeline/decision.py` | 100% | 94% |
| `pipeline/enrollment.py` | 79% | 71% |
| `pipeline/guidance.py` | 95% | 88% |
| `pipeline/landmarks.py` | 86% | 62% |
| `pipeline/liveness.py` | 66% | 50% |
| `pipeline/readout.py` | 87% | 79% |
| `pipeline/recognition.py` | 94% | 88% |
| `pipeline/sampleframes.py` | 100% | 97% |
| `pipeline/train_model.py` | 95% | 83% |
| `pipeline/traits.py` | 90% | 75% |

The `liveness.py` misses are `score()` running the real network, which needs the weights (CI skips them). The voting logic is fully covered. The browser pages (`docs/`, `static/js`) are checked for escaping and shared constants by `test_frontend_escaping.py` and `test_shared_constants.py`. They are not driven in a browser.

### Maintenance types

- **Corrective:** bugs 1 to 4.
- **Adaptive:** nothing needed. The requirements and the SHA-pinned model downloads are unchanged.
- **Perfective:** the CLI now works straight after `register_user` when SFace is available, as the web camera does.
- **Preventive:** the wider `.gitignore`, and seven regression tests.

### Left for later

- One per-frame decision function shared by the camera and the CLI (see Couplers).
- A `UNIQUE` constraint on one mark per person per day.
- Liveness has still not been validated against real printed or replayed attacks (see `pipeline/liveness.py`).

---

## First audit

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
