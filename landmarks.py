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

# The 68-point model's layout, as half-open ranges. Named once and used
# everywhere, because the alternative -- and what this replaced -- was a
# PARTS dict that named them and then twenty lines of `p[36:42]` and `p[45]`
# further down that did not. Two spellings of the same convention, and the
# raw ones give a reader nothing to check against the diagram above.
POINT_COUNT = 68

JAW = (0, 17)
BROW_RIGHT = (17, 22)
BROW_LEFT = (22, 27)
NOSE_BRIDGE = (27, 31)
NOSTRILS = (31, 36)
EYE_RIGHT = (36, 42)
EYE_LEFT = (42, 48)
LIP_OUTER = (48, 60)
LIP_INNER = (60, 68)

# Both brows together, for the forehead arc.
BROWS = (BROW_RIGHT[0], BROW_LEFT[1])

# Individual points the cheek and chin geometry is built from. The outer eye
# corners and the nose wings are the stable landmarks on each side; the jaw
# indices are the points level with them.
CHIN = 8
EYE_RIGHT_OUTER = 36
EYE_LEFT_OUTER = 45
NOSE_RIGHT = 31
NOSE_LEFT = 35
JAW_RIGHT_HIGH = 2
JAW_LEFT_HIGH = 14
JAW_RIGHT_LOW = 4
JAW_LEFT_LOW = 12

# How far above the brow line the synthetic forehead arc sits, in interocular
# distances -- so it scales with the face rather than with the image.
FOREHEAD_RISE = 0.62

# Offsets *within* a named part, for the measurements that need two specific
# points rather than the whole outline. Within-part offsets rather than
# absolute indices because that is how the code reads them: `mouth[3]` is the
# fourth point of the outer lip, not the fourth point of the face.
LIP_CORNER_RIGHT = 0     # within lipOuter
LIP_TOP_CENTRE = 3
LIP_CORNER_LEFT = 6
LIP_BOTTOM_CENTRE = 9
JAW_START = 0            # within jaw
JAW_CHIN = 8
JAW_END = 16

PARTS = {
    "jaw": list(range(*JAW)),
    "browRight": list(range(*BROW_RIGHT)),
    "browLeft": list(range(*BROW_LEFT)),
    "noseBridge": list(range(*NOSE_BRIDGE)),
    "nostrils": list(range(*NOSTRILS)),
    "eyeRight": list(range(*EYE_RIGHT)),
    "eyeLeft": list(range(*EYE_LEFT)),
    "lipOuter": list(range(*LIP_OUTER)),
    "lipInner": list(range(*LIP_INNER)),
}

# Eye aspect ratio below this reads as a closed eye (Soukupova & Cech 2016
# use ~0.2 for blink detection; enrollment can afford to be a little stricter).
# Features that form a closed loop, so their outline joins up.
CLOSED = {"eyeRight", "eyeLeft", "lipOuter", "lipInner"}

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


def derived_points(points):
    """Points the 68-point model does not label, worked out from ones it does.

    Cheekbones: the 68-point layout stops at the jaw contour and the eyes, and
    never names the zygomatic arch. It sits between the outer eye corner and
    the jaw at cheek height, so that midpoint approximates it, with a second
    point lower down beside the nose for the mid-cheek.

    Forehead: there are no points above the brows at all, which is why an
    outline built from the jaw alone stops at the temples and leaves the face
    open at the top. The arc is the brow line pushed up along the face's own
    vertical axis, scaled by eye separation so it holds at any distance.
    """
    if points is None or len(points) < POINT_COUNT:
        return {}

    p = points
    eye_r_c = p[slice(*EYE_RIGHT)].mean(axis=0)
    eye_l_c = p[slice(*EYE_LEFT)].mean(axis=0)
    interocular = float(np.linalg.norm(eye_l_c - eye_r_c)) or 1.0

    # Face's own "up": perpendicular to the eye line, pointing away from the
    # chin. Using this instead of screen-up keeps the arc correct on a tilted
    # head.
    eye_axis = eye_l_c - eye_r_c
    norm = float(np.linalg.norm(eye_axis)) or 1.0
    up = np.array([eye_axis[1], -eye_axis[0]], dtype=np.float32) / norm
    if float(np.dot(up, p[CHIN] - (eye_r_c + eye_l_c) / 2.0)) > 0:
        up = -up   # "up" must point away from the chin

    cheek_r_high = (p[EYE_RIGHT_OUTER] + p[JAW_RIGHT_HIGH]) / 2.0
    cheek_l_high = (p[EYE_LEFT_OUTER] + p[JAW_LEFT_HIGH]) / 2.0
    cheek_r_low = (p[NOSE_RIGHT] + p[JAW_RIGHT_LOW]) / 2.0
    cheek_l_low = (p[NOSE_LEFT] + p[JAW_LEFT_LOW]) / 2.0

    brow = np.vstack([p[slice(*BROWS)]])
    forehead = brow + up * (FOREHEAD_RISE * interocular)

    return {
        "cheekRightHigh": cheek_r_high, "cheekLeftHigh": cheek_l_high,
        "cheekRightLow": cheek_r_low, "cheekLeftLow": cheek_l_low,
        "forehead": forehead,
        "interocular": interocular,
        "up": up,
    }


def outline(points):
    """Closed contour around the whole face: jaw, then the forehead arc back."""
    d = derived_points(points)
    if not d:
        return None
    jaw = points[slice(*JAW)]
    fore = d["forehead"][::-1]      # left temple back to right
    return np.vstack([jaw, fore]).astype(np.int32)


def mesh_points(points):
    """The 68 landmarks plus the derived cheek and forehead points."""
    d = derived_points(points)
    if not d:
        return points
    extra = np.vstack([d["cheekRightHigh"], d["cheekLeftHigh"],
                       d["cheekRightLow"], d["cheekLeftLow"], d["forehead"]])
    return np.vstack([points, extra]).astype(np.float32)


def _aspect(pts):
    """Vertical opening over horizontal width, for an eye or a mouth."""
    width = float(np.linalg.norm(pts[0] - pts[3])) or 1.0
    a = float(np.linalg.norm(pts[1] - pts[5]))
    b = float(np.linalg.norm(pts[2] - pts[4]))
    return (a + b) / (2.0 * width)


def _plain_float(value, places=3):
    """Round to a plain Python float.

    numpy scalars survive round() as numpy scalars, and Flask's JSON encoder
    refuses them -- one float32 in this dict returned HTTP 500 for the whole
    /api/status response, so the page saw nothing at exactly the moment a face
    was found. Every number leaving this module goes through here.
    """
    return round(float(value), places)


def metrics(points):
    """Per-feature measurements plus the flags they imply.

    Every value is a plain Python float via _plain_float(); this dict is serialised
    straight to JSON.
    """
    if points is None or len(points) < POINT_COUNT:
        return None
    p = parts(points)

    ear_r = _aspect(p["eyeRight"])
    ear_l = _aspect(p["eyeLeft"])
    mouth = p["lipOuter"]
    lip_gap = float(np.linalg.norm(
        mouth[LIP_TOP_CENTRE] - mouth[LIP_BOTTOM_CENTRE]))
    lip_width = float(np.linalg.norm(
        mouth[LIP_CORNER_RIGHT] - mouth[LIP_CORNER_LEFT]))
    mar = lip_gap / (lip_width or 1.0)

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
    jaw_width = (float(np.linalg.norm(jaw[JAW_START] - jaw[JAW_END]))
                 / interocular)
    face_height = (float(np.linalg.norm(jaw[JAW_CHIN] - nose_top))
                   / interocular)

    # Cheekbones, from the derived zygomatic points. Width is the arch-to-arch
    # span; prominence is how far they sit outside the jaw below them, which
    # is what separates a wide face from a high-cheekboned one.
    d = derived_points(points)
    cheek_width = float(np.linalg.norm(
        d["cheekLeftHigh"] - d["cheekRightHigh"])) / interocular
    # The jaw part starts at absolute index 0, so a within-jaw offset and an
    # absolute landmark index are the same number here -- which is why the
    # JAW_*_LOW constants serve both uses.
    lower_jaw_width = (float(np.linalg.norm(jaw[JAW_RIGHT_LOW]
                                            - jaw[JAW_LEFT_LOW]))
                       / interocular)
    cheek_prominence = cheek_width / (lower_jaw_width or 1.0)

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
        "eyeOpenRight": _plain_float(ear_r),
        "eyeOpenLeft": _plain_float(ear_l),
        "mouthOpen": _plain_float(mar),
        "browRaiseRight": _plain_float(brow_r),
        "browRaiseLeft": _plain_float(brow_l),
        "interocularPx": _plain_float(interocular, 1),
        "jawWidthRatio": _plain_float(jaw_width),
        "cheekWidthRatio": _plain_float(cheek_width),
        "cheekProminence": _plain_float(cheek_prominence),
        "faceHeightRatio": _plain_float(face_height),
        "centreOffset": _plain_float(centre_offset),
        "eyeMismatch": _plain_float(eye_mismatch),
        "flags": [str(f) for f in flags],
    }


def _delaunay_edges(pts, w, h):
    """Triangulate the point cloud so every point joins its neighbours.

    Drawing a line between all pairs would be 2,600 lines and a hairball.
    A Delaunay triangulation connects each point only to the ones actually
    adjacent to it, which is what reads as a mesh over the face rather than
    a scribble.
    """
    inset = (0, 0, max(int(w), 2), max(int(h), 2))
    sub = cv2.Subdiv2D(inset)
    ok = []
    for q in pts:
        x, y = float(q[0]), float(q[1])
        if 0 <= x < inset[2] and 0 <= y < inset[3]:
            sub.insert((x, y))
            ok.append((x, y))
    if len(ok) < 3:
        return []

    edges = set()
    for t in sub.getTriangleList():
        corners = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if any(not (0 <= cx < inset[2] and 0 <= cy < inset[3]) for cx, cy in corners):
            continue
        for a in range(3):
            p1, p2 = corners[a], corners[(a + 1) % 3]
            edges.add((p1, p2) if p1 <= p2 else (p2, p1))
    return [((int(a[0]), int(a[1])), (int(b[0]), int(b[1]))) for a, b in edges]


def draw(frame, points, colour=(138, 201, 94)):
    """One colour, every point connected, whole face outlined.

    Three layers, all in the same colour and separated by weight instead of
    hue: the mesh as hairlines, the feature outlines and face contour a little
    brighter, and the landmarks themselves as dots. Colouring features
    differently made the overlay look like a diagram of something else; the
    state of the capture is already carried by which colour is passed in.
    """
    if points is None or len(points) < POINT_COUNT:
        return
    h, w = frame.shape[:2]

    faint = tuple(int(c * 0.55) for c in colour)
    pts_all = mesh_points(points)

    # 1. mesh over every point, including the derived cheek and forehead ones
    for a, b in _delaunay_edges(pts_all, w, h):
        cv2.line(frame, a, b, faint, 1, cv2.LINE_AA)

    # 2. the face contour, closed across the forehead
    contour = outline(points)
    if contour is not None:
        cv2.polylines(frame, [contour], True, colour, 1, cv2.LINE_AA)

    # 3. individual features, so they stay legible through the mesh
    for name, idx in PARTS.items():
        cv2.polylines(frame, [points[idx].astype(np.int32)],
                      name in CLOSED, colour, 1, cv2.LINE_AA)

    # 4. the points themselves
    for q in pts_all.astype(np.int32):
        if 0 <= q[0] < w and 0 <= q[1] < h:
            cv2.circle(frame, (int(q[0]), int(q[1])), 1, colour, -1, cv2.LINE_AA)

    # cheekbones marked a touch larger -- they are derived, not detected, and
    # worth being able to pick out
    d = derived_points(points)
    for key in ("cheekRightHigh", "cheekLeftHigh", "cheekRightLow", "cheekLeftLow"):
        q = d.get(key)
        if q is not None and 0 <= q[0] < w and 0 <= q[1] < h:
            cv2.circle(frame, (int(q[0]), int(q[1])), 3, colour, 1, cv2.LINE_AA)
