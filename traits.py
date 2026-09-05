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

import facemodels
from facemodels import AGE_BUCKETS, GENDER_LABELS, CAFFE_MEAN

# Thresholds used to turn raw numbers into "is this sample usable?".
# Tuned for the 200x200 crops register_user.py writes; adjust for your camera.
#
# MIN_SHARPNESS is only a floor for *catastrophic* blur. Laplacian variance
# has no absolute meaning -- it scales with the camera, the face, and how
# much the crop was resized, so one person's sharp sample can measure 1100
# while another's measures 90. Blur is therefore caught per-person and
# relatively, in analytics.summarize_user (SOFT_RATIO below); this constant
# exists only to flag frames that are blurred beyond any doubt.
MIN_SHARPNESS = 25.0
SOFT_RATIO = 0.4          # flag a sample under 40% of that person's median
BRIGHT_RANGE = (75.0, 180.0)
MIN_CONTRAST = 25.0
MIN_FACE_PX = 90
MAX_YAW = 30.0            # degrees off-centre before it stops being frontal
MAX_ROLL = 20.0

GENDER_CAVEAT = (
    "Model guess at apparent presentation from pixels, not a statement about "
    "identity. Levi & Hassner (2015), binary by construction, and materially "
    "less accurate for some groups. Treat as weak evidence."
)
AGE_CAVEAT = (
    "Coarse 8-bucket estimate from a 2015 model trained on Adience. Error of "
    "one whole bucket is common, especially outside 25-45."
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


def quality_metrics(bgr_face):
    gray = to_gray(bgr_face)
    return {
        "sharpness": round(sharpness(gray), 1),
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(gray.std()), 1),
        "qualityScore": learned_quality(bgr_face),
    }


def quality_flags(metrics, geom, grayscale_source):
    """Turn raw metrics into the specific reasons a sample is weak."""
    flags = []
    if metrics["sharpness"] < MIN_SHARPNESS:
        flags.append("blurry")
    if metrics["brightness"] < BRIGHT_RANGE[0]:
        flags.append("too dark")
    elif metrics["brightness"] > BRIGHT_RANGE[1]:
        flags.append("too bright")
    if metrics["contrast"] < MIN_CONTRAST:
        flags.append("flat contrast")
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


def demographics(bgr_face, grayscale_source=False):
    """Age bucket + gender estimate, each with its caveat attached.

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
