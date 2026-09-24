# Contributing

## Setup

```bash
pip install -r requirements.txt
python -m cli.fetch_models   # ~134 MB of third-party weights
python app.py            # http://127.0.0.1:5001
```

`opencv-contrib-python` is the one to install, not `opencv-python`. Contrib is
a superset and is the build that actually ships `cv2.face`, which is the LBPH
recogniser this project uses. Installing both side by side makes the two
packages shadow each other.

On a headless machine — including CI — use
`opencv-contrib-python-headless` instead. Same APIs, no `libGL` dependency.

## What is not in the repository, and why

| Path | Why |
|---|---|
| `dataset/` | Captured webcam images of real people |
| `attendance.db` | Their attendance records |
| `trainer.yml` | Trained from the above, so it encodes it |
| `models/` | Third-party weights, ~134 MB, reproducible |
| `logs/` | Benchmark output, appended forever by design |

The first three are the important ones. **This is a public repository and
face images are biometric data**; keep them out of it. If you add a feature
that writes anything derived from a captured face, add it to `.gitignore` in
the same commit.

A fresh clone therefore has no faces and no weights. The handful of tests
that need the pretrained weights skip rather than fail, which is what makes
the suite runnable in CI at all — keep that property. `pytest -q -rs` shows
you what skipped and why.

## Benchmark corpora

LFW, FairFace, faceage and the CASIA gallery are downloaded on demand and
cached under the system temporary directory. `core/corpus_paths.py` is the only
place that decides where:

| Variable | Effect |
|---|---|
| `FACE_CORPORA` | base directory for every corpus; defaults to the system temp dir |
| `CASIA_DIR` | the CASIA shards and gallery on their own, since they are the largest |

Worth setting `FACE_CORPORA` somewhere permanent — these take a long time to
fetch and some systems clear `/tmp` on boot.

Read paths from `corpus_paths`, never from `os.environ["TEMP"]`. That is
unset outside Windows, and `os.environ.get("TEMP", ".")` is worse: it does
not fail, it just looks in whatever directory the process is standing in and
reports the corpus missing. Both spellings were in here, along with a
FairFace path pointing inside a long-dead editor session.

## Tests

```bash
pytest -q -rs
python -m flake8 . --select=E9,F63,F7,F82
```

382 tests locally; 379 pass and 3 skip without the weights.

`python tools/refresh_figures.py` rewrites that sentence, the figures on
`docs/index.html` and the one in the README from a real measured run — do
not edit any of them by hand. The sentence above was four times out of date
before the script owned it.

## The published demo

`docs/app/` is generated. Edit `tools/static_src/`, then:

```bash
python tools/build_static.py            # rebuild
python tools/build_static.py --prove    # and check the checks still bite
```

The build runs the JavaScript it is about to publish in a real browser and
compares it against the Python it was ported from, over a matrix of
measurement combinations; a single disagreement and it writes nothing. It
needs `pip install playwright && python -m playwright install chromium`,
which is a build-time dependency only — `pytest` does not need it.

The demo covers `pipeline/guidance.py` and `analysis/calibration.py` and
nothing else, because those two are the only modules that import neither
`cv2` nor `numpy`. **Do not extend it to anything that would imply the
browser is recognising faces.** It is not, it cannot, and a page that
suggested otherwise would be worse than no page.

## Conventions

Comments explain *why*. See [CODE_AUDIT.md](CODE_AUDIT.md) for the code smell
and complexity state, and [BENCHMARK.md](BENCHMARK.md) for the accuracy and
fairness numbers and how they were measured.

Names registered through the UI are user input and reach the page — they go
through `esc()`. That was a stored XSS hole once; do not reintroduce it by
building HTML with string concatenation.
