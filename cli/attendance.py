"""
cli/attendance.py
-----------------
Step 3 of the pipeline — the actual attendance system. Opens the webcam,
detects faces frame-by-frame, runs each detected face through the trained
LBPH recognizer, and if it's confident about who it sees, logs a row into
the SQL `attendance` table (once per person per day).

Usage:
    python -m cli.attendance

Press 'q' to quit.
"""

import os

import cv2

from core import db
from core import paths
from core import vision

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# LBPH "confidence" is actually a distance: LOWER means more confident.
# The value lives in vision.py, because analytics.py reports it back to
# you as `currentThreshold` and camera.py matches against it too -- three
# copies of one number, and the report was a typed literal.
CONFIDENCE_THRESHOLD = vision.CONFIDENCE_THRESHOLD

# Overlay drawing. BGR, not RGB -- OpenCV's order, and the reason a "red"
# box drawn with (255, 0, 0) comes out blue. These are the plain full
# saturation pair; camera.py uses the muted palette from the web UI's
# stylesheet instead, because there the box sits next to that page.
MATCH_COLOUR = (0, 255, 0)
UNKNOWN_COLOUR = (0, 0, 255)
BOX_THICKNESS = 2
LABEL_SCALE = 0.7
LABEL_OFFSET = 10       # pixels above the box, so the text clears the line


def identify(recognizer, gray, box):
    """Who this face is, and whether the match is close enough to accept.

    Returns (user_id, confidence, name, accepted). Split out of the capture
    loop because it is the actual decision this program makes, and inside a
    `while True:` around a webcam it could not be reached by a test -- so the
    one piece of logic worth verifying was the one piece that was not.

    `accepted` is `confidence < CONFIDENCE_THRESHOLD` and not the other way
    round: LBPH's "confidence" is a distance, so a lower number is a closer
    match. Getting that backwards accepts every stranger and rejects
    everyone enrolled, which is why it is written down once, here.
    """
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], vision.LBPH_INPUT_SIZE)
    user_id, confidence = recognizer.predict(face)

    accepted = confidence < CONFIDENCE_THRESHOLD
    if not accepted:
        return user_id, confidence, "Unknown", False
    return (user_id, confidence,
            db.get_user_name(user_id) or f"Unknown (id {user_id})", True)


def mark_present(user_id, name, confidence, marked_this_session):
    """Log an accepted match once per session, and say what happened.

    The session set is why this exists: without it every frame re-queries
    the database for somebody standing in front of the camera, and the
    console fills with one line per frame.
    """
    if user_id in marked_this_session:
        return
    written = db.log_attendance(user_id, confidence)
    marked_this_session.add(user_id)
    if written:
        print(f"Logged attendance: {name} at confidence {confidence:.1f}")
    else:
        print(f"{name} already marked present today.")


def annotate(frame, box, label, colour):
    """Draw one face's box and label onto the frame."""
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, BOX_THICKNESS)
    cv2.putText(frame, label, (x, y - LABEL_OFFSET),
                cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, colour, BOX_THICKNESS)


def _open_camera():
    """The capture device, or None with a printed reason."""
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap
    cap.release()
    print("ERROR: Could not open webcam.")
    return None


def _load_recognizer():
    """The trained LBPH model, or None with a printed reason."""
    model_path = paths.model_path()
    if not os.path.exists(model_path):
        print("No trained model found. "
              "Run `python -m cli.register_user \"Name\"` then "
              "`python -m pipeline.train_model` first.")
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    return recognizer


def _report_today():
    print("\nToday's attendance:")
    for name, timestamp, confidence in db.get_attendance_for_today():
        print(f"  {name} - {timestamp} (confidence {confidence:.1f})")


def run_attendance():
    recognizer = _load_recognizer()
    if recognizer is None:
        return

    db.init_db()
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cap = _open_camera()
    if cap is None:
        return

    print("Attendance system running. Press 'q' to quit.")
    marked_this_session = set()  # avoid spamming console/DB checks every frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=vision.DETECT_SCALE_FACTOR,
            minNeighbors=vision.DETECT_MIN_NEIGHBOURS,
            minSize=vision.MIN_FACE_SIZE,
        )

        for box in faces:
            user_id, confidence, name, accepted = identify(
                recognizer, gray, box)
            if accepted:
                mark_present(user_id, name, confidence, marked_this_session)
            annotate(frame, box, f"{name} ({confidence:.0f})",
                     MATCH_COLOUR if accepted else UNKNOWN_COLOUR)

        cv2.imshow("Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    _report_today()


if __name__ == "__main__":
    run_attendance()
