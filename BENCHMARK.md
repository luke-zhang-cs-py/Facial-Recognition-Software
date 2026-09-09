# Benchmark: 97,698 faces

The engine was run over the whole of [FairFace](https://github.com/joojs/fairface)
— 97,698 images, balanced across seven race groups (~14k each), both genders,
and nine age bands. 39.8 minutes at 41 img/s on 10 cores, followed by
4,767,224,190 identity comparisons.

Reproduce the fairness half with:

```bash
python fairness_benchmark.py --corpus <dir-of-parquet> --per-group 800
```

FairFace was chosen because it is openly licensed, demographically labelled,
and built for exactly this measurement. Scraping faces off the web would have
been the wrong way to get a corpus, and the openly available million-face
datasets are not usable: MS-Celeb-1M was withdrawn by Microsoft in 2019 over
consent, and WebFace260M/VGGFace2 need institutional licences.

Scale note: at ~14,000 samples per group the 95% confidence intervals here are
about ±0.4pp. The disparities found were 8–25pp — fifty times wider. More
images would have narrowed error bars that were already far too small to
change any conclusion.

## What held up

**Detection: 99.95%**, 53 misses out of 97,698. Widest race-group gap 0.08pp,
gender gap 0.01pp. YuNet sees everyone.

## What did not, and what changed

### The quality gate was rejecting people, not photographs

The shipped gate used absolute thresholds on mean brightness and contrast.
Both track skin tone, so both encoded it. Measured disparity at a matched 25%
overall flag rate:

| Gate | Disparity | Status |
|---|---|---|
| mean brightness (absolute) | **2.15x** | removed |
| contrast / std (absolute) | **1.61x** | removed |
| `MIN_SHARPNESS = 25` absolute floor | 1.32x | lowered to 6.0 |
| sharpness, relative to the person | 1.20x | kept |
| eDifFIQA learned quality | 1.21x | kept, now the main gate |

In production terms the old gate flagged **38.8% of Black faces and 18.5% of
White faces as "too dark"** — a 2.1x gap that was skin tone, not lighting.
Mean face-crop brightness by group: White 102.4, East Asian 94.9, Indian 84.7,
Black 83.8.

The `MIN_SHARPNESS` floor was documented as catching only catastrophic blur
but was flagging 27% of the corpus; measured p2 of face-crop sharpness is 6.2,
so the floor moved to 6.0 and now catches ~1.7%.

Brightness and contrast are still measured and reported — they are useful
diagnostics. They no longer decide anything. Exposure is judged by clipping
and dynamic range instead, which is skin-tone independent: a dark face that is
well lit still spans a wide range, an underexposed one has its shadows crushed
flat whoever is in it.

**After the fix**, at 800 per group: overall rejection disparity **1.24x
(within budget)**, gender **1.07x**, and the "too dark" flag no longer exists.
Two residuals remain over the 1.25x budget and are documented rather than
tuned away: `turned away` at 1.34x and gender-sliced rejection at 1.26x. Both
track head pose in the source photographs rather than skin tone, and FairFace
is in-the-wild imagery with pose varying by source. Tuning `MAX_YAW` to
equalise them would be fitting the corpus, not fixing a defect.

### The recommended threshold did not survive scale

Every FairFace image is a different person, so all 4.77 billion pairs are
impostor pairs. That makes this a direct false-match measurement:

| Threshold | FMR per pair | Gallery of 100 | Gallery of 1,000 | Gallery of 10,000 |
|---|---|---|---|---|
| 0.363 (SFace reference) | 0.547% | 41.9% | 99.6% | ~100% |
| 0.500 | 0.017% | 1.6% | 15.4% | 81.3% |
| 0.725 | 0.0009% | 0.08% | 0.85% | 8.2% |

A per-pair rate reads as harmless and is not. Against N others the chance of
*some* collision is `1 - (1 - FMR)^(N-1)`, so it compounds with gallery size.
The earlier recommendation of 0.5 came from a sweep over three enrolled people
— with three identities there are three ways to be wrong, so almost any
threshold scores zero false matches and the sweep recommends the loosest one.

`calibration.py` now carries the measured curve, and `analytics.py` refuses to
trust a local sweep below 15 enrolled people, deferring to it instead. The
recommendation is a function of gallery size:

| Enrolled | Threshold for <1% risk |
|---|---|
| 10 | 0.425 |
| 100 | 0.525 |
| 1,000 | 0.725 |
| 10,000 | unreachable |

Past ~10,000 no measured threshold holds the line, and the honest answer is
that SFace embeddings alone do not scale that far — add a second factor rather
than tightening further.

**The risk is also not evenly shared.** At threshold 0.5, the share of people
who falsely match somebody ranges from 51.2% (White) to 76.0% (Indian), a
1.48x gap. Same-race impostor similarity averages 0.1022 against 0.0777
cross-race: the other-race effect, measured.

### The demographic estimators are worse than their reputation

**Gender**, scored against ground truth (72.88% overall):

| | Male | Female |
|---|---|---|
| Middle Eastern | 90.2% | 56.6% |
| White | 87.3% | 61.4% |
| Indian | 88.5% | 53.3% |
| **Black** | 81.2% | **43.7%** |

Male 86.5% vs female 57.6% — a 28.9pp gap — and Black women at 43.7%, worse
than a coin flip on a binary task. This reproduces Buolamwini & Gebru's
*Gender Shades* finding. Recommendation: leave it off, or delete
`models/gender_net.caffemodel`.

**Age**: MAE 13.6 years. Usable on young faces (20-29: 61.6% within 10 years),
useless on older ones (50-59: 11.1%; 70+: MAE 38.4 years). Race differences
are minor (12.8–15.0).

## Bugs this surfaced

Running at scale broke things that a 45-image test never would:

- **`db.py` leaked connections on any error.** Every function closed its
  connection only on the success path, so one failed write left a connection
  open with its transaction, and SQLite then refused every later write with
  "database is locked". One bad row poisoned the process. All access now goes
  through a context manager that closes and rolls back.
- **A `dataset/` folder naming a user id with no `users` row crashed the whole
  analysis** on a foreign key. `save_traits` now reports it and `scan()`
  returns those folders under `orphanFolders`.
- **The fairness benchmark itself called noise "bias"** — a 0.20% vs 0.00%
  detection gap is an infinite ratio and means nothing. Verdicts now require
  both an effect over budget and non-overlapping confidence intervals,
  otherwise they read INCONCLUSIVE.


# Recognition at scale

Identification uses SFace embeddings (`recognition.py`) with the gallery-size
threshold from `calibration.py`. Two questions matter: how often is it right,
and does that hold as more people enroll.

Reproduce with `tools_trials.py` (robustness) and `tools_build_gallery.py` +
`tools_scale_benchmark.py` (scale).

## 4,500 randomised trials, 45 enrolled people

100 trials each. Every trial randomises distance, motion blur, rotation,
gamma, sensor noise and JPEG quality together, so the run samples the
operating space rather than one clean point in it.

| | |
|---|---|
| Correct | **82.2%** |
| Unknown (declined) | 17.7% |
| **Misidentified** | **0.0%** — 1 in 4,500 |

That distribution is the design working. Under degradation it gives up rather
than guesses. A refusal costs somebody a badge swipe; a confident wrong name
puts them in another person's attendance record.

Best: Roh Moo-hyun 99%, Ricardo Lagos 97%, Yao Ming 94%, Michael Jordan 93%.
Worst: LeBron James 72%, Laura Bush 57%, Jennifer Capriati 45% — all enrolled
from the fewest images, which argues for capturing more, not for loosening
the threshold.

## Gallery size: 100 to 10,576 identities

Built from CASIA-WebFace, 10,590 identities, embeddings only. Each person's
centroid comes from 3 embeddings, probed with a held-out 4th.

| Gallery | Threshold | Rank-1 | **Usable** | Declined | Misidentified |
|---|---|---|---|---|---|
| 100 | 0.525 | 89.0% | **70.0%** | 30.0% | 0.0% |
| 500 | 0.650 | 89.0% | **40.8%** | 59.2% | 0.0% |
| 1,000 | 0.725 | 88.6% | **23.2%** | 76.8% | 0.0% |
| 2,500 | 0.750 | 86.6% | **16.5%** | 83.5% | 0.0% |
| 10,576 | 0.750 | 84.2% | **16.4%** | 83.5% | 0.1% |

**Rank-1 barely moves — 89% to 84% — while usable accuracy collapses from 70%
to 16%.** Rank-1 asks only "was the right person nearest", and that is the
number normally quoted as accuracy. It stays flat while the system becomes
unusable underneath it, because the threshold has to keep rising to hold false
matches down, and past a few hundred people it rises faster than the genuine
scores do.

Past 2,500 enrolled, no measured threshold keeps the false-match risk under
1% at all. `calibration.py` predicted this arithmetically from FairFace
impostor pairs before any of it was measured here; the measurement agrees.

**What this means in practice.** The system is sound for a class, a team, an
office floor — up to a few hundred. It is not an identification system for
ten thousand people, and no threshold tuning makes it one. At that scale it
needs a second factor: a badge, a PIN, a name typed in, with the face
confirming rather than searching.

The one reassurance is that it fails safely at every size. Misidentification
never exceeds 0.1%: what grows is refusal, not error.


## How many samples does enrollment need?

Registration captures 30. That number was inherited, never measured. Swept on
LFW: 24 identities deep enough to build a 30-sample centroid and still have
probes left, against a 610-identity gallery.

| Samples | Correct | Unknown | Wrong |
|---|---|---|---|
| 1 | 42.0% | 58.0% | 0% |
| 2 | 72.3% | 27.7% | 0% |
| 3 | 79.0% | 21.0% | 0% |
| 5 | **89.1%** | 10.9% | 0% |
| 8 | 90.8% | 9.2% | 0% |
| 12 | 91.6% | 8.4% | 0% |
| 16 | **92.4%** | 7.6% | 0% |
| 20 | 92.4% | 7.6% | 0% |
| 30 | 92.4% | 7.6% | 0% |

**Almost everything is bought by the fifth image, and returns stop entirely at
sixteen** — 17 through 30 are worth literally nothing here. A single sample is
a bad idea at 42%; two is worth more than the following fourteen combined.

Capture still takes 30, deliberately. These were LFW press photographs:
decently lit, in focus, framed by someone whose job it was. A webcam
enrollment produces duds, and the pose stages spend samples on angles that are
individually worse but collectively necessary. The headroom exists to be spent
on bad frames. Drop it only alongside a measurement of what your own camera
produces.

The enrollment report now states the measured identification rate for however
many usable samples a person ended up with, and what more would buy.
