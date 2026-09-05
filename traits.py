"""
traits.py
----------
Per-face feature extraction. Given one image, work out what can actually be
measured about the face in it:

    geometry      box, 5 landmarks, approximate yaw/roll, detector score
    quality       sharpness, brightness, contrast, face size in pixels, and
                  eDifFIQA's learned 0..1 quality score
    embedding     128-d SFace vector (a real metric space, unlike LBPH)
    demographics  age bucket and a gender estimate, both with caveats

Everything degrades rather than fails. No YuNet weights -> no landmarks, so
no pose, but sharpness and brightness still compute. No SFace -> no
embedding. The caller checks for None.

Two notes on inputs, because they change what the numbers mean:

*   `dataset/` holds 200x200 GRAYSCALE crops, but the age/gender nets and
    eDifFIQA were trained on colour. Analysing a stored sample therefore
    gives a worse answer than analysing the live colour frame. Anything
    produced from a grey source is tagged `grayscaleSource: True` so the
    report can say so instead of quietly pretending otherwise.
*   A pre-cropped face often will not re-detect (YuNet wants context around
    the head). When detection fails on a small square image we fall back to
    treating the whole image as the face box, and mark `detected: False`.
"""

import math

import cv2
import numpy as np

import calibration
import facemodels
from facemodels import AGE_BUCKETS, GENDER_LABELS, CAFFE_MEAN

# Thresholds used to turn raw numbers into "is this sample usable?".
#
# These were re-derived after benchmarking the engine on all 97,698 images of
# FairFace (race-balanced, ~14k per group). The headline result: an absolute
# threshold on any quantity that tracks skin tone measures the person, not the
# photograph. Measured disparity between the worst- and best-affected race
# group, at a matched 25% overall flag rate:
#
#     mean brightness (absolute)   2.15x     <- was shipped, now removed
#     contrast / std  (absolute)   1.61x     <- was shipped, now removed
#     sharpness (relative)         1.20x     <- kept
#     eDifFIQA learned quality     1.21x     <- kept, now the main gate
#
# The shipped absolute brightness gate flagged 38.8% of Black faces and 18.5%
# of White faces as "too dark" -- a 2.1x gap that is skin tone, not lighting.
# Brightness and contrast are still measured and reported, because they are
# genuinely useful diagnostics; they just no longer decide anything.
#
# What survives gating is scale-free or learned: relative sharpness, the
# eDifFIQA score, geometry, and a clipping backstop for exposures that are
# broken beyond argument.

# Absolute floor for *catastrophic* blur only. Laplacian variance has no
# absolute meaning -- it scales with camera, face, and crop resizing -- so
# real blur detection is per-person and relative, in
# analytics._mark_relative_blur (SOFT_RATIO).
#
# This was 25.0, which flagged 27% of a 97k-image corpus -- a real gate
# pretending to be a backstop, and the last skin-tone-coupled check left
# (1.32x across race groups). Against the measured distribution of face-crop
# sharpness, p2 is 6.2, so 6.0 catches the bottom ~1.7%: frames that are
# blurred beyond argument, which is what this constant claims to be for.
MIN_SHARPNESS = 6.0
SOFT_RATIO = 0.4          # flag a sample under 40% of that person's median

# Main quality gate. eDifFIQA is a learned face-image-quality model, and it
# was the fairest signal measured (1.21x) as well as the most meaningful.
MIN_QUALITY = 0.25

# Exposure backstop. Not a brightness band -- a clipping check. A face can be
# dark and perfectly exposed; it cannot be half crushed to pure black and
# still carry detail. These are deliberately extreme so they fire only on
# genuinely destroyed frames (lens cap, blown highlight, sensor failure).
MAX_SHADOW_CLIP = 0.50    # fraction of pixels at/near 0
MAX_HIGHLIGHT_CLIP = 0.35  # fraction of pixels at/near 255
MIN_DYNAMIC_RANGE = 25.0   # p99 - p1; below this there is no signal at all

MIN_FACE_PX = 90
MAX_YAW = 30.0            # degrees off-centre before it stops being frontal
MAX_ROLL = 20.0

# Reported but no longer used for gating -- see the note above.
BRIGHT_RANGE = (75.0, 180.0)
MIN_CONTRAST = 25.0

GENDER_CAVEAT = (
    "Model guess at apparent presentation from pixels, not a statement about "
    "identity. Levi & Hassner (2015), binary by construction, and materially "
    "less accurate for some groups. Treat as weak evidence."
)
AGE_CAVEAT = (
    "Levi & Hassner (2015), trained on Adience. Measured on FairFace: mean "
    "absolute error 12.4 years, and a +/-6 year range contains the true age "
    "only ~40% of the time. Accurate enough to say roughly young or roughly "
    "old, not to act on."
)


# ------------------------------------------------------------------ helpers

def to_bgr(img):
    """Accept grey or colour, always hand back 3-channel BGR."""
    if img is None:
        return None, False
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), True
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR), True
    # A colour-format image whose channels are identical is still grey data.
    b, g, r = cv2.split(img)
    is_grey = bool(np.array_equal(b, g) and np.array_equal(g, r))
    return img, is_grey


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ----------------------------------------------------------------- geometry

def detect(bgr):
    """Run YuNet. Returns a list of 15-element rows, best score first."""
    net = facemodels.get("yunet")
    if net is None:
        return []
    h, w = bgr.shape[:2]
    with facemodels.lock_for("yunet"):
        net.setInputSize((w, h))
        _, faces = net.detect(bgr)
    if faces is None:
        return []
    return sorted([f for f in faces], key=lambda r: -float(r[14]))


def geometry(row):
    """Box, landmarks and approximate pose from one YuNet row.

    Yaw is estimated from how far the nose sits off the midpoint of the eyes,
    scaled by eye separation. That is a rough proxy, not a calibrated pose
    solver -- good enough to tell "looking at the camera" from "looking away",
    which is all the quality check needs.
    """
    x, y, w, h = (float(v) for v in row[:4])
    pts = np.array(row[4:14], dtype=np.float32).reshape(5, 2)
    right_eye, left_eye, nose, mouth_r, mouth_l = pts

    eye_mid = (right_eye + left_eye) / 2.0
    eye_dist = float(np.linalg.norm(left_eye - right_eye)) or 1.0

    roll = math.degrees(math.atan2(float(left_eye[1] - right_eye[1]),
                                   float(left_eye[0] - right_eye[0])))
    yaw_ratio = float(nose[0] - eye_mid[0]) / eye_dist
    yaw = yaw_ratio * 75.0  # empirical scaling to something degree-ish

    mouth_mid = (mouth_r + mouth_l) / 2.0
    vertical = float(np.linalg.norm(mouth_mid - eye_mid)) or 1.0
    pitch_ratio = float(nose[1] - eye_mid[1]) / vertical

    return {
        "box": [round(x), round(y), round(w), round(h)],
        "facePx": int(round(max(w, h))),
        "landmarks": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts],
        "yaw": round(yaw, 1),
        "roll": round(roll, 1),
        "pitchRatio": round(pitch_ratio, 3),
        "eyeDist": round(eye_dist, 1),
        "score": round(float(row[14]), 3),
    }


# ------------------------------------------------------------------ quality

def sharpness(gray):
    """Variance of the Laplacian — the standard cheap blur measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def learned_quality(bgr_face):
    """eDifFIQA score in 0..1. Higher is a more recognisable face image."""
    net = facemodels.get("ediffiqa")
    if net is None:
        return None
    blob = cv2.dnn.blobFromImage(bgr_face, 1 / 255.0, (112, 112), (0, 0, 0), swapRB=True)
    blob = (blob - 0.5) / 0.5
    with facemodels.lock_for("ediffiqa"):
        net.setInput(blob)
        out = net.forward()
    return round(float(np.ravel(out)[0]), 4)


def exposure_metrics(gray):
    """Exposure judged by clipping and usable range, not by average level.

    This is the skin-tone-independent way to ask "is this photo exposed
    properly". A dark face that is well lit still spans a wide range with
    little clipping; an underexposed one has its shadows crushed flat
    against zero whoever is in it.
    """
    total = gray.size or 1
    p1, p99 = np.percentile(gray, [1, 99])
    return {
        "shadowClip": round(float((gray <= 8).sum()) / total, 4),
        "highlightClip": round(float((gray >= 247).sum()) / total, 4),
        "dynamicRange": round(float(p99 - p1), 1),
    }


def quality_metrics(bgr_face):
    gray = to_gray(bgr_face)
    return {
        "sharpness": round(sharpness(gray), 1),
        # Reported for diagnostics; deliberately not gated on. See the note
        # at the top of this file for the measured bias.
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(gray.std()), 1),
        "qualityScore": learned_quality(bgr_face),
        **exposure_metrics(gray),
    }


def quality_flags(metrics, geom, grayscale_source):
    """Turn raw metrics into the specific reasons a sample is weak.

    Every check here is either scale-free, learned, or an extreme backstop.
    Nothing gates on absolute brightness or contrast -- benchmarking showed
    both encode skin tone (2.15x and 1.61x disparity across race groups),
    so gating on them rejects people rather than photographs.
    """
    flags = []

    if metrics["sharpness"] < MIN_SHARPNESS:
        flags.append("blurry")

    q = metrics.get("qualityScore")
    if q is not None and q < MIN_QUALITY:
        flags.append("low quality")

    # Exposure backstops: broken frames, not dark ones.
    if metrics.get("shadowClip", 0) > MAX_SHADOW_CLIP:
        flags.append("underexposed")
    if metrics.get("highlightClip", 0) > MAX_HIGHLIGHT_CLIP:
        flags.append("blown highlights")
    if metrics.get("dynamicRange", 255) < MIN_DYNAMIC_RANGE:
        flags.append("no tonal range")

    if geom:
        if geom["facePx"] < MIN_FACE_PX:
            flags.append("face too small")
        if abs(geom["yaw"]) > MAX_YAW:
            flags.append("turned away")
        if abs(geom["roll"]) > MAX_ROLL:
            flags.append("head tilted")
    return flags


# ---------------------------------------------------------------- embedding

def embed(bgr, row=None):
    """128-d SFace vector, L2-normalised so cosine similarity is a dot product.

    With a YuNet row we use alignCrop, which warps the face to the canonical
    112x112 layout the network expects. Without one (a pre-cropped dataset
    image that would not re-detect) we just resize, which is measurably worse
    but still usable.
    """
    net = facemodels.get("sface")
    if net is None:
        return None
    with facemodels.lock_for("sface"):
        if row is not None:
            aligned = net.alignCrop(bgr, row)
        else:
            aligned = cv2.resize(bgr, (112, 112))
        feat = net.feature(aligned)
    vec = np.ravel(np.asarray(feat, dtype=np.float32))
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def cosine(a, b):
    """Similarity in [-1, 1]. Same person is typically > 0.36 for SFace."""
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


# ------------------------------------------------------------- demographics

def _caffe_predict(name, bgr_face, labels):
    net = facemodels.get(name)
    if net is None:
        return None
    blob = cv2.dnn.blobFromImage(bgr_face, 1.0, (227, 227), CAFFE_MEAN, swapRB=False)
    with facemodels.lock_for(name):
        net.setInput(blob)
        probs = np.ravel(net.forward())
    order = probs.argsort()[::-1]
    return {
        "label": labels[int(order[0])],
        "confidence": round(float(probs[order[0]]), 3),
        "runnerUp": labels[int(order[1])],
        "runnerUpConfidence": round(float(probs[order[1]]), 3),
        "distribution": {labels[i]: round(float(probs[i]), 3) for i in range(len(labels))},
    }


def _age_point_estimate(distribution):
    """Probability-weighted age in years, plus a calibrated range.

    Reading off the winning bucket throws away everything the other seven say.
    A face split 0.40/0.35 between "25-32" and "38-43" is a statement about
    someone around 34, and the argmax reports 28.5. Weighting all eight
    midpoints by their probability measured better on FairFace: MAE 12.4y
    against 13.4y.

    The range that comes back is deliberately narrow *and* carries how often
    it is actually right. A +/-5y band looks authoritative and contains the
    truth a third of the time; showing the width without the coverage would
    be the more precise-looking of two wrong answers.
    """
    mids = calibration.AGE_MIDPOINTS
    total = sum(distribution.get(b, 0.0) for b in AGE_BUCKETS) or 1.0
    years = sum(distribution.get(b, 0.0) * m
                for b, m in zip(AGE_BUCKETS, mids)) / total
    spread = math.sqrt(sum(distribution.get(b, 0.0) * (m - years) ** 2
                           for b, m in zip(AGE_BUCKETS, mids)) / total)
    k = calibration.AGE_BAND_YEARS
    return {
        "years": round(years, 1),
        "range": [max(0, round(years - k)), round(years + k)],
        "halfWidth": k,
        "coverage": round(calibration.age_band_coverage(k), 3),
        "modelSpread": round(spread, 1),
        "maeYears": calibration.AGE_MAE_EXPECTED,
    }


def demographics(bgr_face, grayscale_source=False):
    """Age estimate + gender estimate, each with its caveat attached.

    Both nets are reported with their full probability distribution rather
    than a bare label, because the margin is usually the interesting part:
    a 0.34/0.31 split is a coin flip dressed up as an answer.
    """
    age = _caffe_predict("age", bgr_face, AGE_BUCKETS)
    gender = _caffe_predict("gender", bgr_face, GENDER_LABELS)
    if age is None and gender is None:
        return None

    out = {"grayscaleSource": grayscale_source}
    if age:
        age["caveat"] = AGE_CAVEAT
        age["uncertain"] = age["confidence"] < 0.5
        age["estimate"] = _age_point_estimate(age["distribution"])
        out["age"] = age
    if gender:
        gender["caveat"] = GENDER_CAVEAT
        gender["uncertain"] = gender["confidence"] < 0.75
        out["gender"] = gender
    if grayscale_source:
        out["warning"] = ("Estimated from a greyscale crop; both nets expect "
                          "colour, so treat these as weaker than a live read.")
    return out


# ------------------------------------------------------------------ top API

def analyze(img, want_embedding=True, want_demographics=True,
            require_detection=False):
    """Full trait read for the most prominent face in `img`.

    `require_detection` suppresses the demographic estimate when no face was
    actually found. Both Caffe nets are classifiers with a softmax over a
    fixed label set: hand them a black frame and they will still return a
    label, and often a high-confidence one. That is not a detection failure
    they can report — it is just what a classifier does when asked about
    something outside its domain. Live camera callers should pass True so a
    covered lens does not produce a confident readout about nobody.

    Returns None only if the image itself could not be read.
    """
    bgr, is_grey = to_bgr(img)
    if bgr is None:
        return None

    rows = detect(bgr)
    row = rows[0] if rows else None
    geom = geometry(row) if row is not None else None

    if row is not None:
        x, y, w, h = geom["box"]
        pad = int(0.12 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
        crop = bgr[y0:y1, x0:x1]
    else:
        crop = bgr  # already a cropped face, or nothing was found

    if crop.size == 0:
        crop = bgr

    metrics = quality_metrics(crop)
    result = {
        "detected": row is not None,
        "grayscaleSource": is_grey,
        "faces": len(rows),
        "geometry": geom,
        **metrics,
        "flags": quality_flags(metrics, geom, is_grey),
    }
    if geom is None:
        result["facePx"] = int(min(bgr.shape[:2]))
    else:
        result["facePx"] = geom["facePx"]

    result["usable"] = not result["flags"]

    if want_embedding:
        vec = embed(bgr, row)
        result["embedding"] = vec
        result["hasEmbedding"] = vec is not None

    if want_demographics and (row is not None or not require_detection):
        result["demographics"] = demographics(crop, grayscale_source=is_grey)
    elif want_demographics:
        result["demographics"] = None
        result["demographicsSkipped"] = "no face detected"

    return result
