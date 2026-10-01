# Code audit, October 2026

This covers the second pass after `notes/CODE_AUDIT.md`, against the same checklist: code smells, bug classes, coverage, and maintenance type.

Each bug below was first reproduced by a test on the code as it was before this pass. All 12 of those tests failed there. Both regression tests that check the normal path ("the page itself can still post" and "the CLI still marks a live face") passed there and still pass.

Suite: 434 tests after the SFace follow-up (431 pass, 3 skip without the pretrained weights). Statement coverage is 87% of 2,506, up from 84% of 2,338.

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

## Attendance now decided by SFace (follow-up, same day)

On the same held-out LFW photos, SFace named the right person 15/16 times with 0 wrong names and 0 strangers accepted. LBPH at threshold 70 got 9/16 right, named the wrong person 5 times, and accepted 12/16 strangers. The owner approved the switch after the first pass.

- **`decision.sface_gallery()`.** The web camera and the CLI decide by SFace whenever the weights are present and someone enrolled has an embedding. Otherwise they fall back to LBPH, and `decision.lbph_reason()` says which of the two conditions caused it.
- **The decision itself.** It is `recognition.match_vector`, the rule `/api/identify` already used: the calibrated threshold for the gallery size, plus a margin over the runner-up. Liveness, the primary-face rule and one mark a day are unchanged.
- **The mirrored preview.** The camera embeds the unmirrored frame using `decision.unmirror_row`, which also swaps the eyes and mouth corners, because SFace aligns the face by those points. On a real face, unmirroring matched the straight-on embedding at 0.96 similarity, against 0.94 for embedding the mirrored frame directly.
- **`recognition.refresh_gallery()`.** It embeds samples that have no embedding yet, such as those from `cli.register_user` and from older enrollments. Results are cached by path and mtime, so a refresh takes about 0.3 s on this checkout. When someone has no usable embedding, attendance names them instead of silently never marking them.
- **`attendance.method`.** This new column records `sface` or `lbph`, because the two numbers run in opposite directions: SFace gives a similarity, where higher is closer, and LBPH gives a distance, where lower is closer. Existing databases gain the column on `init_db`, and older rows read as `lbph`, which is what they were. `cli.view_report` prints each number with its direction.
- **Tests.** `tests/test_sface_attendance.py` holds 13 tests. Each was checked against a deliberate break: no eye swap, accepting below the threshold, or embedding the mirrored frame. Each break failed at least one test.

## Data cleanup

- **`--help` user removed.** The user row "--help" (id 9, no attendance rows) and its empty `dataset/9_--help` folder were deleted at the owner's request.
- **Still open: two folders share id 7.** In the local `dataset/`, the demo folder for id 7 (12 LFW samples) sits beside a 30-sample folder for id 7 left over from an earlier database. Folders are matched to users by their id prefix, so both sets train as user 7 and one person's face is filed under another's name, in both LBPH and the SFace centroid. The owner chose to keep it. The guard below stops it from happening again.
- **New ids skip claimed folders.** `db.add_user` never hands out an id that a `dataset/` folder already claims (`paths.folder_ids()`). This checkout has leftover folders up to id 372, so the next user gets id 373.

## More people enrolled from LFW

Running `cli.seed_demo --keep` exposed two problems, both now fixed:
- **Duplicates.** It picked the most-photographed people first, who are the ones already enrolled, and enrolled them a second time under new ids. `load_lfw(exclude=...)` now skips them.
- **Its own crop, and a slow re-scan.** It cropped with its own code instead of `pipeline/decision.py`, and it ran a full `analytics.scan(use_cache=False)` over every folder, including ones with no user. It now uses `decision` for the crop and `recognition.refresh_gallery()` for the embeddings.

`python -m cli.seed_demo --people 20 --samples 20 --keep` added 20 people with 20 samples each (ids 373–392). That makes 28 people in the SFace gallery, all of them recognisable. On 60 LFW photos of the new people that enrollment did not use, SFace named 57 correctly, named nobody wrongly, and refused 3.

Then about 1,000 more scans: `--people 83 --samples 12 --keep` wrote 996 samples. That makes 111 people in the gallery, all of them recognisable, with 1,522 samples on disk for enrolled people. At 111 people the calibrated threshold rises to 0.55 (gallery false-match risk 0.0068):

| at 111 people | result |
|---|---|
| held-out photos of enrolled people (past the 20th, so never enrolled) | 158 tested: 141 right, **0 wrong**, 17 refused |
| strangers (LFW people never enrolled, one photo each) | 300 tested: **1 wrongly accepted** |

The refusals come from the threshold rising with the gallery. A refusal is a retry, and a wrong name is somebody else's attendance record, which is the trade the threshold is set to make.

**CI fix.** The `--keep` test had built a parquet file with `pyarrow`, which the CI runner doesn't install (it's optional and only needed for the corpus). It failed on both 3.10 and 3.12. The choice of people is now `seed_demo.pick_people()`, which takes plain lists, and the test exercises that directly. It was checked with `pyarrow` hidden.

## Maintenance classification

| Type | Changes |
|---|---|
| Corrective | bugs 1–6, 9, 10 |
| Preventive | bugs 7, 8 (no exploit observed); the single `decision` module, so the paths can't drift again; the tests that hold each path to the same orientation and detector |
| Perfective | attendance by SFace instead of LBPH; centroid de-duplication; the dead-code and naming cleanup |
| Adaptive | none |
