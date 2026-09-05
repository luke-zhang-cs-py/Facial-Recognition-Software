"""
landmarks.py
-------------
68-point facial landmarks, and what can be measured from them.

YuNet gives five points -- two eyes, nose tip, two mouth corners -- which is
enough to align a face and estimate pose, and nothing else. A 68-point fit
outlines the individual features, so the parts can be measured separately:
are the eyes open, is the mouth neutral, are both sides of the face equally
visible.

That last one is the useful one for enrollment. A face half in shadow or
partly behind a hand still detects fine and still produces a confident-looking
sample; comparing the left and right halves catches it.

Model: OpenCV's LBF facemark (Ren et al. 2014), models/lbfmodel.yaml. Optional
like everything else -- without it the five-point path still works.

Index map for the standard 68-point layout:
    0-16   jaw line, right ear to left ear
    17-21  right eyebrow      22-26  left eyebrow
    27-30  nose bridge        31-35  nostril line
    36-41  right eye          42-47  left eye
    48-59  outer lip          60-67  inner lip
"""

import os
import threading

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lbfmodel.yaml")

PARTS = {
    "jaw": list(range(0, 17)),
    "browRight": list(range(17, 22)),
    "browLeft": list(range(22, 27)),
    "noseBridge": list(range(27, 31)),
    "nostrils": list(range(31, 36)),
    "eyeRight": list(range(36, 42)),
    "eyeLeft": list(range(42, 48)),
    "lipOuter": list(range(48, 60)),
    "lipInner": list(range(60, 68)),
}

# Eye aspect ratio below this reads as a closed eye (Soukupova & Cech 2016
# use ~0.2 for blink detection; enrollment can afford to be a little stricter).
EAR_CLOSED = 0.19
# Mouth aspect ratio above this is an open mouth, which distorts the lower face.
MAR_OPEN = 0.45
# Left/right size mismatch beyond this suggests one side is occluded or the
# head is turned far enough that half the face is unusable.
ASYMMETRY_LIMIT = 0.28

_model = None
_lock = threading.Lock()


def available():
    return os.path.exists(MODEL_PATH)


def _get():
    global _model
    if not available():
        return None
    with _lock:
        if _model is None:
            m = cv2.face.createFacemarkLBF()
            m.loadModel(MODEL_PATH)
            _model = m
        return _model


def fit(gray, box):
    """68 points for one face box, or None. `box` is (x, y, w, h)."""
    model = _get()
    if model is None:
        return None
    x, y, w, h = (int(v) for v in box)
    if w <= 0 or h <= 0:
        return None
    with _lock:
        ok, shapes = model.fit(gray, np.array([[x, y, w, h]], dtype=np.int32))
    if not ok or not len(shapes):
        return None
    return np.asarray(shapes[0][0], dtype=np.float32)


def parts(points):
    """Split the 68 points into named features."""
    if points is None:
        return {}
    return {name: points[idx] for name, idx in PARTS.items()}


def _aspect(pts):
    """Vertical opening over horizontal width, for an eye or a mouth."""
    width = float(np.linalg.norm(pts[0] - pts[3])) or 1.0
    a = float(np.linalg.norm(pts[1] - pts[5]))
    b = float(np.linalg.norm(pts[2] - pts[4]))
    return (a + b) / (2.0 * width)


def metrics(points):
    """Per-feature measurements plus the flags they imply."""
    if points is None or len(points) < 68:
        return None
    p = parts(points)

    ear_r = _aspect(p["eyeRight"])
    ear_l = _aspect(p["eyeLeft"])
    mouth = p["lipOuter"]
    mar = (float(np.linalg.norm(mouth[3] - mouth[9]))
           / (float(np.linalg.norm(mouth[0] - mouth[6])) or 1.0))

    eye_r_c = p["eyeRight"].mean(axis=0)
    eye_l_c = p["eyeLeft"].mean(axis=0)
    interocular = float(np.linalg.norm(eye_l_c - eye_r_c)) or 1.0

    # Brow raise, normalised by eye separation so it does not scale with
    # distance from the camera.
    brow_r = float(eye_r_c[1] - p["browRight"].mean(axis=0)[1]) / interocular
    brow_l = float(eye_l_c[1] - p["browLeft"].mean(axis=0)[1]) / interocular

    # Symmetry: how far the nose bridge sits from the midpoint of the eyes,
    # and how different the two eyes measure. Either one flags a half-hidden
    # face -- occlusion, harsh side lighting, or an extreme turn.
    nose_top = p["noseBridge"][0]
    eye_mid_x = (eye_r_c[0] + eye_l_c[0]) / 2.0
    centre_offset = abs(float(nose_top[0]) - eye_mid_x) / interocular
    eye_mismatch = abs(ear_r - ear_l) / (max(ear_r, ear_l) or 1.0)

    jaw = p["jaw"]
    jaw_width = float(np.linalg.norm(jaw[0] - jaw[16])) / interocular
    face_height = float(np.linalg.norm(jaw[8] - nose_top)) / interocular

    flags = []
    if ear_r < EAR_CLOSED and ear_l < EAR_CLOSED:
        flags.append("eyes closed")
    elif ear_r < EAR_CLOSED or ear_l < EAR_CLOSED:
        flags.append("one eye closed")
    if mar > MAR_OPEN:
        flags.append("mouth open")
    if eye_mismatch > ASYMMETRY_LIMIT or centre_offset > ASYMMETRY_LIMIT:
        flags.append("face partly obscured")

    return {
        "eyeOpenRight": round(ear_r, 3),
        "eyeOpenLeft": round(ear_l, 3),
        "mouthOpen": round(mar, 3),
        "browRaiseRight": round(brow_r, 3),
        "browRaiseLeft": round(brow_l, 3),
        "interocularPx": round(interocular, 1),
        "jawWidthRatio": round(jaw_width, 3),
        "faceHeightRatio": round(face_height, 3),
        "centreOffset": round(centre_offset, 3),
        "eyeMismatch": round(eye_mismatch, 3),
        "flags": flags,
    }


# Drawing colours per feature group, so the overlay reads as separate parts
# rather than one cloud of dots. BGR, matching static/css/style.css.
PART_COLOURS = {
    "jaw": (165, 150, 139),
    "browRight": (255, 163, 77), "browLeft": (255, 163, 77),
    "noseBridge": (74, 192, 224), "nostrils": (74, 192, 224),
    "eyeRight": (138, 201, 94), "eyeLeft": (138, 201, 94),
    "lipOuter": (94, 106, 223), "lipInner": (94, 106, 223),
}

# Features that form a closed loop, so the outline joins up.
CLOSED = {"eyeRight", "eyeLeft", "lipOuter", "lipInner"}


def draw(frame, points, alpha_parts=None):
    """Trace each feature in its own colour."""
    if points is None or len(points) < 68:
        return
    for name, idx in PARTS.items():
        if alpha_parts and name not in alpha_parts:
            continue
        pts = points[idx].astype(np.int32)
        colour = PART_COLOURS.get(name, (200, 200, 200))
        cv2.polylines(frame, [pts], name in CLOSED, colour, 1, cv2.LINE_AA)
        for q in pts:
            cv2.circle(frame, tuple(q), 1, colour, -1, cv2.LINE_AA)
