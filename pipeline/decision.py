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

from core import db
from core import vision
from pipeline import traits

CONFIDENCE_THRESHOLD = vision.CONFIDENCE_THRESHOLD


def boxes(bgr):
    """Face boxes from the project's one detector (YuNet, Haar fallback).

    The camera and seed_demo detect with traits.detect; the CLI ran its own
    Haar cascade, and a Haar box and a YuNet box are different crops of the
    same head -- LBPH distances moved by about 10 between them.
    """
    return [tuple(int(v) for v in traits.geometry(row)["box"]) for row in traits.detect(bgr)]


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


def record(user_id, confidence):
    """Write the attendance row, if there is a user to write it for.

    train_model trains every folder in dataset/, including folders whose user
    row is gone, so LBPH can predict a label the foreign key refuses; writing
    it raised IntegrityError, which crashed the CLI and failed the camera.
    """
    if not db.user_exists(user_id):
        return NO_USER
    return LOGGED if db.log_attendance(user_id, confidence) else ALREADY
