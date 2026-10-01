# Code audit, October 2026

This covers the second pass after `notes/CODE_AUDIT.md`, against the same checklist: code smells, bug classes, coverage, and maintenance type.

Each bug below was first reproduced by a test on the code as it was before this pass. All 12 of those tests failed there. Both regression tests that check the normal path ("the page itself can still post" and "the CLI still marks a live face") passed there and still pass.

Suite: 421 tests (418 pass, 3 skip without the pretrained weights). Statement coverage is 86% of 2,409, up from 84% of 2,338.

## The finding behind most of the others

There are four paths into or out of the LBPH model:

- the web camera (`pipeline/camera.py`)
- `cli.register_user`
- `cli.attendance`
- `cli.seed_demo`

They had drifted apart. LBPH compares texture histograms grid cell by grid cell, so it is not mirror-invariant and it is sensitive to where the box is drawn. Each difference between the paths therefore cost accuracy, with no visible symptom. The attendance decision now lives once, in `pipeline/decision.py`, and the web camera and the CLI both call it.

These were measured on held-out LFW crops from this checkout:

| | rank-1 at threshold 70 |
|---|---|
| unmirrored query vs unmirrored gallery | 9/16 |
| mirrored query (what the web camera sent) | 4/16 |

Querying a YuNet-cropped gallery with Haar crops (what the CLI sent) cut correct accepts at the threshold from 42 to 28 of 92.

## Bugs fixed

| # | Class | Where | What happened | Fix | Test |
|---|---|---|---|---|---|
| 1 | functional / integration | camera.py | The preview is mirrored, and the crops matched and stored were taken from that mirrored frame. Galleries from the CLI and seed_demo are unmirrored. | `decision.crop(..., mirrored=True)` flips the crop back. Only the preview stays mirrored. | `test_the_camera_matches_…`, `test_the_camera_stores_…` |
| 2 | integration | cli/register_user.py | It ran its own Haar cascade. Everything else detects with `traits.detect` (YuNet, with Haar as a fallback). | `decision.boxes`, using the largest face | `test_cli_registration_crops_with_the_detector_…` |
| 3 | out-of-bounds | cli/attendance.py, register_user.py | A box starting left of or above the frame was sliced with a negative start. That counts from the far edge, so the crop came from the wrong side, or was empty and crashed `cv2.resize`. | `decision.crop` clips to the frame and returns None when nothing is left | `test_a_box_off_the_edge_…` |
| 4 | security (presentation attack) | cli/attendance.py | The CLI had no liveness check. A photograph the web camera refuses was marked present by `python -m cli.attendance`, using the same model and threshold. | It now uses the same `LivenessVote` and marks only on `live`. Without the liveness model it marks nobody and says to run `cli.fetch_models`. | `test_the_cli_refuses_a_photograph_…`, `test_the_cli_marks_nobody_without_…`, `test_the_cli_still_marks_a_live_face` |
| 5 | logical | cli/attendance.py | It marked every face in the frame. One liveness vote can't vouch for two people. | Only the primary (largest) face is recognised. Other faces are drawn but not decided on. | `test_the_cli_decides_on_the_largest_face_only` |
| 6 | runtime | cli/attendance.py, camera.py | `train_model` trains every `dataset/` folder, including ones whose user row is gone. Logging such a label raised `IntegrityError`, which crashed the CLI and the camera loop. | `decision.record` returns `NO_USER`. The CLI prints the id, and the camera logs an error event telling you to retrain. | `test_…_label_with_no_user` (×3) |
| 7 | security (CSRF) | app.py | `get_json(force=True)` reads a `text/plain` body, and any web site can POST one without a preflight. So any page open in the browser could start the camera, register a user, or retrain. | A `before_request` hook refuses state-changing requests with a foreign or `null` Origin, or with `Sec-Fetch-Site: cross-site`. | `test_a_cross_site_page_cannot_drive_the_camera`, `test_the_page_itself_can_still_post` |
| 8 | security (DNS rebinding) | app.py | A hostile name that re-resolves to 127.0.0.1 becomes same-origin with the server. It could then read `/api/report` (names and attendance) and `/video_feed`. | Requests whose Host is not 127.0.0.1, localhost or [::1] get a 403. | `test_a_rebound_hostname_reads_nothing` |
| 9 | logical | analysis/analytics.py | `best_threshold` counted "admits no false match" as clean. The strictest threshold admits nobody, so it is always clean, and when it was the only clean one it became the recommendation. That threshold marks nobody present. | A clean setting must also accept somebody. | `test_best_threshold_never_recommends_accepting_nobody` |
| 10 | workflow | cli/register_user.py | Any one argument was taken as the name, so `--help` registered a user called "--help". The folder `dataset/9_--help` in this checkout dates from 2026-09-24 and is evidence of it. | argparse, plus refusing names that start with `-`. Exit code 1 on bad usage, as before. | `test_register_user_help_prints_usage_and_creates_nobody` |

## Smells fixed

- **Duplicated code:**
  - The crop → resize → predict → threshold → name → log sequence was written out in both `camera.py` and `cli/attendance.py`, and bugs 1 and 3 to 6 are what came of the two copies drifting. Both now call `pipeline/decision.py`.
  - The centroid was computed twice, in `enrollment.centroid_of` and `recognition._centroid`, and the two disagreed. For embeddings that average to the origin, one returned None and the other returned the zero vector, which is 0.0 similar to everyone. `recognition.centroid` is now the only copy and returns None in that case. The test that pinned the zero-vector behaviour was updated.
- **Dead code:** removed `FACE_CASCADE_PATH` from both CLI modules, and the `vision` import that `register_user` no longer needs.
- **Uncommunicative names:** `crop` in `camera.py` had meant three different things across the function. It is now `sample` and `face_img`.

## Not changed, and why

- **Attendance still uses LBPH, not SFace.** On the same held-out LFW photos, SFace named the right person 15/16 times with 0 wrong names and 0 strangers accepted. LBPH at threshold 70 got 9/16 right, named the wrong person 5 times, and accepted 12/16 strangers. Switching is the biggest accuracy gain available. But it changes the recognizer behind every attendance row, and it needs the SFace weights and an embedding for every enrolled sample, so it is a design decision rather than a bug fix. Recommended as the next perfective change.
- `dataset/9_--help` is left on disk. It's personal data, so removing it is the owner's call.

## Maintenance classification

| Type | Changes |
|---|---|
| Corrective | bugs 1–6, 9, 10 |
| Preventive | bugs 7, 8 (no exploit observed); the single `decision` module, so the paths can't drift again; the tests that hold each path to the same orientation and detector |
| Perfective | centroid de-duplication; the dead-code and naming cleanup |
| Adaptive | none |
