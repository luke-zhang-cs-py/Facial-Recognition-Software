# Face Recognition Attendance System

[![CI](https://github.com/luke-zhang-cs-py/Facial-Recognition-Software/actions/workflows/python-package.yml/badge.svg)](https://github.com/luke-zhang-cs-py/Facial-Recognition-Software/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)](https://www.python.org/)

A local attendance system: OpenCV reads the webcam, recognises enrolled faces,
and logs each one to SQLite. Nothing leaves the machine.

### ▶ [Point your own webcam at it →](https://luke-zhang-cs-py.github.io/Facial-Recognition-Software/camera/) · [or drive the same rules with sliders →](https://luke-zhang-cs-py.github.io/Facial-Recognition-Software/app/)

![A face tracked live: the mesh follows it in, the instruction changes from No face detected to Move closer to Head upright to Ready, then a second person appears and the verdict becomes One person only](docs/demo.gif)

*Tracked live, judged live. The mesh is MediaPipe's; the instruction, checklist
and priority chain are this project's `pipeline/guidance.py` — fourteen rules in
a fixed order, of which only the first to fire is shown, so a person gets one
thing to do instead of a list. `tools/build_static.py` refuses to publish the
JavaScript if it disagrees with the Python.*

> **The recording** handed a still photograph to the browser as if it were a
> camera, and paints the frame over *after* the measurements are read — so the
> GIF carries the tracking and nobody's face. Open the page and it is your own
> camera; no frame leaves the tab.

**Neither page recognises anyone.** That needs a 328 MB `trainer.yml` and a
camera the server owns; a tab gets detection, landmarks and guidance. MediaPipe
is not this project's detector either — the app uses YuNet or Haar and OpenCV's
68-point LBF model — so the camera page labels every input as the same
measurement, an approximation, or not measured at all.

**[Read the full write-up →](https://luke-zhang-cs-py.github.io/Facial-Recognition-Software/)**
— the 97,698-face benchmark, where recognition stops working, and every bug this
has had. (Or [`docs/index.html`](docs/index.html) locally.)

## Run it

```bash
pip install -r requirements.txt
python -m cli.fetch_models         # ~134 MB of pretrained weights, once
python app.py                      # http://127.0.0.1:5001
```

The camera is opened by the *server*, not the browser, so this runs on the
machine it is plugged into — which is why the demos cover only the arithmetic.
**Don't skip the middle line:** attendance works without those weights, but
traits, liveness, quality scoring and SFace identification degrade quietly,
which looks like the app being broken rather than incomplete.
`GET /api/models` names the missing ones.

Then register a person, capture ~30 samples, and the LBPH model retrains
itself; recognised faces get one `attendance` row per person per day. The same
on the CLI — `cli.register_user`, `cli.attendance`, `cli.view_report` — and
`cli.seed_demo --people 40` enrolls from LFW to try recognition without
registering anyone. Every feature, with the endpoint and command behind it, is
[listed on the camera page](https://luke-zhang-cs-py.github.io/Facial-Recognition-Software/camera/#features).

## Measured, not assumed

All 97,698 images of FairFace. Full results in **[BENCHMARK.md](BENCHMARK.md)**.

- **Detection is even** — 99.95%, widest race-group gap 0.08pp.
- **The quality gate used to be biased, and was fixed.** Absolute brightness and
  contrast encode skin tone, and flagged **38.8% of Black faces vs 18.5% of
  White faces** as "too dark". Gating is now scale-free and learned signals
  only; disparity fell from 2.15× to 1.24×.
- **The right threshold depends on how many people are enrolled** — one swept
  over a few identities cannot see false matches. Over 4.77 billion impostor
  pairs: 0.425 at 10 people, 0.725 at 1,000, **nothing sufficient past ~1,646**.
  The birthday problem, and why `analysis/calibration.py` takes a gallery size
  instead of returning a constant.
- **The gender estimator fails badly for Black women** (43.7%, worse than
  chance). Leave it off unless you have a reason not to.

**Effectiveness.** 40 enrolled, held-out LFW images: **92.9% correct**
(223/240), 6.7% rejected as unknown, **0.4% misidentified**, 100% impostor
rejection over 300 non-enrolled faces. The failure mode is the safe one.

**Operating range.** Detection holds at 100% throughout; recognition degrades.

| Condition | Holds until | Breaks at |
|---|---|---|
| Distance (downscale) | 0.25× (83%) | 0.15× (23%) |
| Motion blur | 9 px (83%) | 13 px (47%) |
| Lighting (gamma 0.4–2.2) | **no measurable loss** | — |
| In-plane rotation | 15° (80%) | 30° (37%) |
| JPEG compression | q20 (83%) | q10 (77%) |

A tilted camera costs more than a dim room.

## Layout

```
app.py        the Flask entry point, and the only module left in the root
core/         paths, vision constants, the database, the model loaders
pipeline/     the capture path: camera, traits, landmarks, liveness,
              guidance, recognition, training
analysis/     enrollment quality, threshold sweeps, calibration, fairness
cli/          the command-line entry points
tools/        developer scripts, corpus benchmarks, the demo build
```

Thirty-one modules used to sit flat in the root. `tests/layout.py` writes the
grouping down once, and the structural tests read it from there.

## Caveats that matter

- **Nothing gates on skin tone**: exposure advice fires on clipped pixels, not
  average brightness. **Head coverings are not mentioned** either — the honest
  failure is "no face detected", not a guess at what somebody is wearing. Brims
  and dark lenses do occlude, so those are named, and only when detection is
  actually failing.
- **Age** is one figure with a ±6-year band, shown *with how often that band is
  right, which is ~40%* — MAE 12.4 years over 10,946 faces. **Gender** guesses
  apparent presentation from pixels and says nothing about identity. Both
  softmax over a fixed label set, so they label *anything* confidently,
  including a black frame; the readout suppresses them when no face is found.
- This measures **image and recogniser properties**. Inferring character from
  face geometry is physiognomy: it does not work, and nothing here does it.
- **Privacy:** treat `dataset/` and `trainer.yml` as biometric data — don't
  commit them, and delete a person's folder and retrain if they ask to be
  removed.

## Tests

```bash
pytest -q
```

382 tests, 81% of 2,315 statements, needing no webcam, no weights and no
corpus. Tests that need a real face use a sample frame if one is present and
**skip** rather than asserting against a synthetic one, because a face a
detector accepts cannot be faked convincingly enough to be evidence.

## License

[MIT](LICENSE) — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup and conventions.
