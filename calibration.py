"""
calibration.py
---------------
False-match rates measured on a large corpus, and the arithmetic for turning
them into a threshold you can actually use.

Why this file exists
--------------------
A threshold swept on a small enrollment is misleading in a specific and
dangerous way. With three enrolled people there are three ways to be wrong,
so almost any threshold scores 0% false matches and the sweep happily
recommends the loosest one. With a thousand people there are half a million
ways to be wrong.

The numbers below come from running the engine over all 97,698 images of
FairFace -- every one a different person, so all 4,767,224,190 pairs are
impostor pairs -- and counting how many exceeded each threshold.

The per-pair rate is not the number that matters. What matters is the chance
that *someone in your gallery* gets mistaken for someone else, which grows
with gallery size:

    P(at least one false match) = 1 - (1 - FMR) ** (N - 1)

At threshold 0.5 the per-pair rate is 0.017%, which sounds harmless. In a
gallery of 1,000 it means a 15% chance of a false match; at 10,000 it is 81%.
That is the birthday problem, and it is why `recommend_threshold` takes the
gallery size as an argument rather than returning a constant.

Caveats
-------
These are impostor rates only. FairFace has one image per person, so it
cannot measure the genuine-match side: a threshold that never confuses two
people may also fail to recognise the right one. Use `analytics.py`'s
leave-one-out numbers on your own enrollment for that half, and treat these
as the ceiling on how loose the threshold can safely be.

FairFace is race-balanced but is still a specific corpus of mostly frontal,
reasonably lit photographs. Your camera in your room will differ.
"""

import bisect

# Measured on 97,645 distinct identities, 4.77e9 impostor pairs.
# (threshold, false match rate per pair)
SFACE_FMR = [
    (0.300, 0.01653983),
    (0.325, 0.00957929),
    (0.350, 0.00546767),
    (0.375, 0.00308784),
    (0.400, 0.00172866),
    (0.425, 0.00096123),
    (0.450, 0.00053202),
    (0.475, 0.00029584),
    (0.500, 0.00016737),
    (0.525, 0.00009863),
    (0.550, 0.00006213),
    (0.575, 0.00004239),
    (0.600, 0.00003127),
    (0.625, 0.00002420),
    (0.650, 0.00001907),
    (0.675, 0.00001498),
    (0.700, 0.00001152),
    (0.725, 0.00000857),
    (0.750, 0.00000611),
]

CORPUS = "FairFace (97,645 identities, 4.77e9 impostor pairs)"

# SFace's own published operating point. Included because it is what most
# examples use, and because at any real gallery size it is far too loose:
# at 0.363 essentially every person in a 1,000-person gallery collides with
# somebody.
SFACE_REFERENCE = 0.363

# Measured disparity in per-person false-match rate across FairFace's seven
# race groups, at threshold 0.5. Kept here so the UI can show that the risk
# is not evenly distributed.
FALSE_MATCH_DISPARITY = {
    "threshold": 0.5,
    "worst": ("Indian", 0.7602),
    "best": ("White", 0.5121),
    "ratio": 1.48,
}


def fmr_at(threshold):
    """Interpolate the measured per-pair false match rate at a threshold."""
    xs = [t for t, _ in SFACE_FMR]
    ys = [f for _, f in SFACE_FMR]
    if threshold <= xs[0]:
        return ys[0]
    if threshold >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_left(xs, threshold)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    frac = (threshold - x0) / (x1 - x0)
    return y0 + frac * (y1 - y0)


def gallery_risk(threshold, gallery_size):
    """Chance that a given person falsely matches anyone else in the gallery."""
    if gallery_size < 2:
        return 0.0
    return 1.0 - (1.0 - fmr_at(threshold)) ** (gallery_size - 1)


def recommend_threshold(gallery_size, max_risk=0.01):
    """Loosest measured threshold whose gallery-wide risk stays under max_risk.

    Returns (threshold, achieved_risk, reachable). `reachable` is False when
    even the strictest measured threshold cannot hit the target -- at which
    point the honest answer is that this recogniser does not scale to that
    gallery, not that you should pick a number anyway.
    """
    for threshold, _ in SFACE_FMR:
        risk = gallery_risk(threshold, gallery_size)
        if risk <= max_risk:
            return threshold, risk, True
    strictest = SFACE_FMR[-1][0]
    return strictest, gallery_risk(strictest, gallery_size), False


def risk_table(gallery_size):
    """Every measured threshold with its risk at this gallery size."""
    return [{"threshold": t,
             "fmrPerPair": f,
             "galleryRisk": gallery_risk(t, gallery_size)}
            for t, f in SFACE_FMR]


def describe(gallery_size, max_risk=0.01):
    """One-line summary for the CLI and the web UI."""
    threshold, risk, ok = recommend_threshold(gallery_size, max_risk)
    if not ok:
        return (f"With {gallery_size} people enrolled, even the strictest measured "
                f"threshold ({threshold}) leaves a {risk:.0%} chance of a false "
                f"match. SFace embeddings alone are not enough at this scale -- "
                f"add a second factor (PIN, badge) rather than tightening further.")
    return (f"With {gallery_size} people enrolled, threshold {threshold} keeps the "
            f"chance of any false match at {risk:.2%} (target {max_risk:.0%}). "
            f"Calibrated on {CORPUS}.")


# ---------------------------------------------------------------------------
# Age estimator, calibrated on the 10,946 FairFace validation faces that
# detected. Ages are compared against the midpoint of FairFace's own band,
# which adds a little error of its own; treat these as slightly optimistic.
# ---------------------------------------------------------------------------

# Adience bucket midpoints, in years.
AGE_MIDPOINTS = [1.0, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 70.0]

# Taking the probability-weighted mean over all eight buckets beats reading
# off the argmax bucket: MAE 12.4y against 13.4y. The distribution carries
# information the winning bucket throws away -- a face split 0.4/0.35 between
# "25-32" and "38-43" is a statement about someone around 34, and the argmax
# reports 28.5.
AGE_MAE_EXPECTED = 12.4
AGE_MAE_ARGMAX = 13.4

# How often the true age actually falls within +/- k years of the estimate.
# This is the number that stops a narrow band from being a lie: a +/-5y range
# looks confident and is wrong two times in three.
AGE_COVERAGE = {
    1: 0.056, 2: 0.100, 3: 0.143, 4: 0.237, 5: 0.332, 6: 0.398,
    8: 0.486, 10: 0.543, 12: 0.584, 15: 0.686, 20: 0.800,
}

# Displayed half-width, set to 2 years by request.
#
# Be clear about what this means: a +/-2 year window contains the true age
# 10% of the time. Nine readings in ten are wrong. The underlying error did
# not change when the window shrank -- MAE is still 12.4 years -- so this is
# a display choice, not an accuracy improvement, and every surface that shows
# the range also shows the coverage so it cannot be mistaken for confidence.
#
# For reference: +/-10y is right 54% of the time, +/-20y is right 80%.
AGE_BAND_YEARS = 2


def age_band_coverage(half_width):
    """Measured share of faces whose true age falls within +/- half_width."""
    if not AGE_COVERAGE:
        return None
    keys = sorted(AGE_COVERAGE)
    if half_width <= keys[0]:
        return AGE_COVERAGE[keys[0]]
    if half_width >= keys[-1]:
        return AGE_COVERAGE[keys[-1]]
    lo = max(k for k in keys if k <= half_width)
    hi = min(k for k in keys if k >= half_width)
    if lo == hi:
        return AGE_COVERAGE[lo]
    frac = (half_width - lo) / (hi - lo)
    return AGE_COVERAGE[lo] + frac * (AGE_COVERAGE[hi] - AGE_COVERAGE[lo])


def describe_age(years, half_width=None):
    """Render an age estimate with its real confidence attached."""
    k = AGE_BAND_YEARS if half_width is None else half_width
    cov = age_band_coverage(k)
    lo, hi = max(0, round(years - k)), round(years + k)
    return f"{round(years)} yrs ({lo}-{hi}, right ~{cov:.0%} of the time)"


# ---------------------------------------------------------------------------
# How evenly this system performs across demographic groups, measured on all
# 97,698 FairFace images. See BENCHMARK.md.
#
# This is the legitimate use of race labels: auditing whether a system works
# equally for everyone. It is the opposite of classifying an individual's
# ethnicity -- it measures the software, not the person, and the labels come
# from the corpus rather than from a model guessing about somebody.
# ---------------------------------------------------------------------------

FAIRNESS = {
    "corpus": "FairFace, 97,698 images, 7 race groups (~14k each)",
    "checks": [
        {"name": "Face detection",
         "value": "99.95%", "disparity": 1.00,
         "detail": "Widest gap between race groups 0.08pp. Even."},
        {"name": "Quality gate rejection",
         "value": "1.24x", "disparity": 1.24,
         "detail": "Within the 1.25x budget, after removing the absolute "
                   "brightness and contrast thresholds that encoded skin tone "
                   "(2.15x and 1.61x)."},
        {"name": "False-match burden",
         "value": "1.48x", "disparity": 1.48,
         "detail": "At threshold 0.5, 76.0% of Indian faces falsely match "
                   "somebody against 51.2% of White faces. Not evenly shared."},
        {"name": "Gender estimator",
         "value": "43.7% worst cell", "disparity": 2.07,
         "detail": "Black women, against 90.2% for Middle Eastern men. Worse "
                   "than chance on a binary task. Recommend leaving it off."},
    ],
}
