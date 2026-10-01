"""
pipeline/decision.py
--------------------
The attendance decision, once, for every front end.

The web camera (pipeline/camera.py) and `python -m cli.attendance` used to
each crop, predict, threshold and log on their own, and drifted apart: the
CLI had no liveness check, logged every face in the frame, ran its own Haar
cascade, and crashed on a label with no user row. These functions are the
one version both now call (notes/CODE_AUDIT_2026-10.md).
"""

import cv2
import numpy as np

from core import db
from core import facemodels
from core import vision
from pipeline import recognition
from pipeline import traits

CONFIDENCE_THRESHOLD = vision.CONFIDENCE_THRESHOLD


def faces(bgr):
    """[(box, row)] from the project's one detector (YuNet, Haar fallback).

    The camera and seed_demo detect with traits.detect; the CLI ran its own
    Haar cascade, and a Haar box and a YuNet box are different crops of the
    same head -- LBPH distances moved by about 10 between them. The row is
    kept because SFace aligns the face by its landmarks.
    """
    return [(tuple(int(v) for v in traits.geometry(row)["box"]), row)
            for row in traits.detect(bgr)]


def boxes(bgr):
    """Just the boxes of faces()."""
    return [box for box, _ in faces(bgr)]


def primary(face_boxes):
    """The largest face: the person presenting. One liveness vote cannot
    vouch for two people, so only this one is recognised and marked."""
    return max(face_boxes, key=lambda b: b[2] * b[3]) if face_boxes else None


def crop(gray, box, mirrored=False):
    """The face at LBPH size, or None when the box holds no pixels.

    The box is clipped to the frame: YuNet can report one starting left of
    or above the image, and a negative slice start counts from the far edge.
    `mirrored`: the box is on a mirrored preview frame, so the crop is
    flipped back -- every gallery is stored as the camera saw the face, and
    LBPH is not mirror-invariant.
    """
    x, y, w, h = (int(v) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    face = gray[y0:y1, x0:x1]
    if mirrored:
        face = cv2.flip(face, 1)
    return cv2.resize(face, vision.LBPH_INPUT_SIZE)


SFACE, LBPH = "sface", "lbph"


def sface_gallery(refresh=True):
    """The SFace gallery attendance should decide against, or None for LBPH.

    SFace whenever its weights are present and somebody enrolled has an
    embedding: on held-out LFW photos it named 15/16 correctly with no wrong
    names and no strangers accepted, where LBPH at its threshold got 9/16,
    named 5 wrongly and accepted 12 of 16 strangers
    (notes/CODE_AUDIT_2026-10.md). `refresh` first embeds any samples that
    have not been yet (cached, so cheap after the first time).
    """
    if not facemodels.have("sface"):
        return None
    db.init_db()
    if refresh:
        recognition.refresh_gallery()
    return recognition.gallery() or None


def lbph_reason():
    """Why attendance is falling back to LBPH, in words a front end can show."""
    if not facemodels.have("sface"):
        return ("SFace weights missing: recognising with LBPH, which is far "
                "less accurate. Run: python -m cli.fetch_models")
    return ("Nobody enrolled has an SFace embedding yet: recognising with LBPH, "
            "which is far less accurate.")


def unenrolled(gal):
    """Names of enrolled people the gallery cannot recognise (no usable
    embedding in any sample), so a front end can say so rather than leave
    them silently never marked."""
    return [name for user_id, name in db.get_all_users() if user_id not in gal]


def unmirror_row(row, width):
    """A YuNet row found on a mirrored frame, as it sits on the unmirrored one.

    Box x -> width - x - w, each landmark x -> width - 1 - x, and the left and
    right eyes and mouth corners swap places: what is the right eye on the
    mirror is the left eye on the face. SFace's alignCrop depends on that order.
    """
    out = np.array(row, dtype=np.float32, copy=True)
    out[0] = width - out[0] - out[2]
    for i in range(4, 14, 2):
        out[i] = width - 1 - out[i]
    out[[4, 5, 6, 7]] = out[[6, 7, 4, 5]]          # right eye <-> left eye
    out[[10, 11, 12, 13]] = out[[12, 13, 10, 11]]  # mouth corners
    return out


def identify(bgr, row, gal):
    """(user_id, similarity, accepted) by SFace, or (None, None, False) when
    the face cannot be embedded. The rule -- calibrated threshold for this
    gallery size, and a margin over the runner-up -- is recognition's, the
    same one /api/identify applies."""
    vec = traits.embed(bgr, row)
    if vec is None:
        return None, None, False
    result = recognition.match_vector(vec, gal)
    chosen = result["match"] or result["best"]
    return int(chosen["userId"]), float(chosen["similarity"]), result["match"] is not None


def match(recognizer, face):
    """(user_id, confidence, accepted). LBPH's "confidence" is a distance:
    lower is closer, so a match is accepted below the threshold."""
    user_id, confidence = recognizer.predict(face)
    return int(user_id), float(confidence), confidence < CONFIDENCE_THRESHOLD


def name_of(user_id):
    """The user's name, or a label saying the id has no user row."""
    return db.get_user_name(user_id) or f"Unknown (id {user_id})"


# What record() did, so each front end can say it in its own words.
LOGGED, ALREADY, NO_USER = "logged", "already", "no-user"


def record(user_id, confidence, method=LBPH):
    """Write the attendance row, if there is a user to write it for.

    train_model trains every folder in dataset/, including folders whose user
    row is gone, so LBPH can predict a label the foreign key refuses; writing
    it raised IntegrityError, which crashed the CLI and failed the camera.
    """
    if not db.user_exists(user_id):
        return NO_USER
    return LOGGED if db.log_attendance(user_id, confidence, method) else ALREADY
