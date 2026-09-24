# Face Recognition Attendance System

[![CI](https://github.com/luke-zhang-cs-py/Facial-Recognition-Software/actions/workflows/python-package.yml/badge.svg)](https://github.com/luke-zhang-cs-py/Facial-Recognition-Software/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)](https://www.python.org/)

A local attendance system: OpenCV reads the webcam, recognises enrolled faces,
and logs each one to SQLite. Everything stays on the machine — no cloud, no
face data leaving the room.

### ▶ [Try the guidance & calibration demo →](https://luke-zhang-cs-py.github.io/Facial-Recognition-Software/app/)

![Pushing one measurement at a time past its threshold: the instruction changes from Ready to Move closer to Look at the centre of the camera, and the priority chain shows which rules are firing underneath the one being shown](docs/demo.gif)

*Fourteen rules, tried in a fixed order; only the first to fire is shown. Watch
row 3 fire while row 5 is **also firing** underneath — that's the design: the
person gets one thing to do, not a list.*

**It does not recognise faces, and doesn't pretend to.** That needs a Haar
cascade, a 328 MB `trainer.yml` and a camera the server owns — none of which
fit in a tab. What's there is the two modules that import neither `cv2` nor
`numpy`: `pipeline/guidance.py` and `analysis/calibration.py`.
`tools/build_static.py` runs the JavaScript against the Python over a matrix of
measurements and refuses to publish if they disagree.

**[Read the full write-up →](https://luke-zhang-cs-py.github.io/Facial-Recognition-Software/)**
— what the 97,698-face benchmark found, where recognition stops working, and
every bug this has had. (Or open [`docs/index.html`](docs/index.html) locally.)

## Run it

Needs a machine with a webcam — this won't work in a cloud sandbox.

```bash
pip install -r requirements.txt
python app.py                      # http://127.0.0.1:5001
```

Register a person, capture ~30 samples, and the LBPH model retrains itself.
Then start attendance: recognised faces get one `attendance` row per person per
day. The CLI does the same four steps (`python -m cli.register_user`,
`cli.attendance`, `cli.view_report`).

To try recognition without registering anyone, enroll from LFW:

```bash
python -m cli.seed_demo --people 40 --samples 10
python -m cli.seed_demo --remove
```

## Measured, not assumed

Benchmarked against all 97,698 images of FairFace. Full results in
**[BENCHMARK.md](BENCHMARK.md)**. The short version:

- **Detection is even** — 99.95%, widest race-group gap 0.08pp.
- **The quality gate used to be biased, and was fixed.** Absolute brightness and
  contrast thresholds encode skin tone (2.15× and 1.61× disparity) and flagged
  **38.8% of Black faces vs 18.5% of White faces** as "too dark". Gating now
  uses scale-free and learned signals only; disparity is 1.24×.
- **Thresholds depend on how many people are enrolled.** A threshold swept on a
  few identities cannot see false matches. `analysis/calibration.py` carries the
  curve measured over 4.77 billion impostor pairs — 0.425 at 10 people, 0.725 at
  1,000, and **nothing sufficient past ~1,646**. That's the birthday problem,
  and it's why the recommendation takes a gallery size instead of returning a
  constant. The demo above plots it.
- **The gender estimator fails badly for Black women (43.7%, worse than
  chance).** Leave it off unless you have a reason not to.

### Effectiveness and operating range

40 enrolled, held-out LFW images: **92.9% correct** (223/240), 6.7% rejected as
unknown, **0.4% misidentified**, and 100% impostor rejection over 300
non-enrolled faces. The failure mode is the safe one — it declines far more
often than it names the wrong person.

Where it stops working, under a controlled sweep (detection held at 100%
throughout; it's recognition that degrades):

| Condition | Holds until | Breaks at |
|---|---|---|
| Distance (downscale) | 0.25× (83%) | 0.15× (23%) |
| Motion blur | 9 px (83%) | 13 px (47%) |
| Lighting (gamma 0.4–2.2) | **no measurable loss** | — |
| In-plane rotation | 15° (80%) | 30° (37%) |
| JPEG compression | q20 (83%) | q10 (77%) |

Lighting invariance is the standout; rotation is the weakest axis, so a tilted
camera costs more than a dim room.

## Two deliberate choices in the guidance

**Nothing gates on skin tone.** Exposure advice fires only on clipped pixels,
never on average brightness. Telling someone their face is "too dark" because
of their complexion is the same defect as the old quality gate, just phrased
more politely.

**Head coverings are not mentioned.** The honest failure is "no face detected",
not a guess about what somebody is wearing. A hijab, turban or kippah doesn't
interfere with detection. Brims and dark lenses genuinely occlude, so those are
named — and only when detection is actually failing.

## Layout

Grouped by responsibility. Thirty-one modules used to sit flat in the root,
which said nothing about which the recogniser needs and which is developer
scripting:

```
app.py        the Flask entry point, and the only module left in the root
core/         paths, vision constants, the database, the model loaders
pipeline/     the capture path: camera, traits, landmarks, liveness,
              guidance, recognition, training
analysis/     measuring the result: enrollment quality, threshold sweeps,
              calibration, the fairness benchmark
cli/          the command-line entry points
tools/        developer scripts, corpus benchmarks, the demo build
```

`tests/layout.py` writes that down once, and the structural tests read it from
there. They used to spell it `os.listdir(ROOT)` — exactly right while
everything was flat, and silently wrong the moment it wasn't: `listdir` still
returns a list, the loop still runs, and the assertion passes over a set of one.

## Caveats that matter

- **Age** is a single figure with a ±6-year band, shown *with how often that
  band is actually right, which is ~40%*. That pairing is the point. MAE is
  12.4 years on 10,946 FairFace faces.
- **Gender** is a binary classifier guessing apparent presentation from pixels.
  It is not a statement about anyone's identity, and it is materially less
  accurate for some groups than others.
- Both classifiers softmax over a fixed label set, so they return a confident
  label for *anything*, including a black frame. The live readout suppresses
  them when no face is detected.
- This measures **image and recogniser properties**. Inferring character,
  personality or intent from face geometry is physiognomy; it does not work,
  and nothing here does it.
- **Privacy:** `dataset/` holds raw face images and `trainer.yml` a trained
  model. Treat both as sensitive biometric data — don't commit them, and delete
  a person's folder and retrain if they ask to be removed.

## Tests

```bash
pytest -q
```

382 tests, 81% of 2,315 statements. The uncovered part is mostly the frame loop
itself — opening the device, grabbing, annotating — and the suite needs no
webcam, no pretrained weights and no corpus: tests that need a real face use a
sample frame if one is present and **skip** rather than asserting against a
synthetic one, because a face a detector accepts cannot be faked convincingly
enough to be evidence.

## License

[MIT](LICENSE) — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup and conventions.
