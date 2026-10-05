"""
cli/attendance.py
-----------------
Step 3 of the pipeline — the actual attendance system. Opens the webcam,
detects faces frame-by-frame, runs the largest face through the trained
LBPH recognizer and the liveness check, and if it is confident about who it
sees and that they are a person rather than a photograph, logs a row into
the SQL `attendance` table (once per person per day). The decision is
pipeline/decision.py, the same one the web camera makes.

Usage:
    python -m cli.attendance

Press 'q' to quit.
"""

import os

import cv2

from core import db
from core import paths
from core import vision
from pipeline import decision
from pipeline import liveness

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
    face = decision.crop(gray, box)
    if face is None:
        return None, float("inf"), "Unknown", False
    user_id, confidence, accepted = decision.match(recognizer, face)
    if not accepted:
        return user_id, confidence, "Unknown", False
    return user_id, confidence, decision.name_of(user_id), True


def identify_sface(bgr, row, gallery):
    """identify(), by SFace: (user_id, similarity, name, accepted)."""
    user_id, similarity, accepted = decision.identify(bgr, row, gallery)
    if not accepted:
        return user_id, similarity, "Unknown", False
    return user_id, similarity, decision.name_of(user_id), True


def mark_present(user_id, name, confidence, marked_this_session, method=decision.LBPH):
    """Log an accepted match once per session, and say what happened.

    The session set is why this exists: without it every frame re-queries
    the database for somebody standing in front of the camera, and the
    console fills with one line per frame.
    """
    if user_id in marked_this_session:
        return
    outcome = decision.record(user_id, confidence, method)
    marked_this_session.add(user_id)
    if outcome == decision.LOGGED:
        reading = (f"similarity {confidence:.3f}" if method == decision.SFACE
                   else f"distance {confidence:.1f}")
        print(f"Logged attendance: {name} at {reading}")
    elif outcome == decision.ALREADY:
        print(f"{name} already marked present today.")
    else:
        print(f"Recognised label {user_id}, which has no user: its dataset/ "
              "folder outlived the person. Remove it and retrain.")


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
        print(f"  {name} - {timestamp} (score {confidence:.3g})")


def run_attendance():
    db.init_db()
    # SFace first: when it decides, trainer.yml is never read, so a missing
    # one is no reason to refuse. Asking for it first turned away everybody
    # who had registered but not yet run train_model, though SFace could
    # already recognise them.
    gallery = decision.sface_gallery()
    recognizer = None if gallery else _load_recognizer()
    if not gallery and recognizer is None:
        return

    if not liveness.available():
        # The web camera refuses everyone without the liveness model rather
        # than let a held-up photograph through; this path now does the same.
        print("The liveness model is missing, so nobody can be marked present "
              "(a photograph would pass). Run `python -m cli.fetch_models`.")
    vote = liveness.LivenessVote()
    if gallery:
        print(f"Recognising with SFace: {len(gallery)} people in the gallery.")
        missing = decision.unenrolled(gallery)
        if missing:
            print("No usable embedding, so never recognised: " + ", ".join(missing))
    else:
        print(decision.lbph_reason())

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
        found = decision.faces(frame)
        faces = [b for b, _ in found]
        box = decision.primary(faces)
        if box is None:
            vote.reset()        # nobody here: the last person's frames are void
        else:
            # Person, or a picture of one? One vote, for the one face decided
            # on, scored and embedded before anything is drawn on the frame:
            # the liveness crop takes in 2.7x the face, so it reaches the boxes
            # and labels of anybody standing beside them.
            live_score = liveness.score(frame, box) if liveness.available() else None
            if gallery:
                row = next(r for b, r in found if b == box)
                user_id, confidence, name, accepted = identify_sface(frame, row, gallery)
                method, shown = decision.SFACE, "-" if confidence is None else f"{confidence:.2f}"
            else:
                user_id, confidence, name, accepted = identify(recognizer, gray, box)
                method, shown = decision.LBPH, f"{confidence:.0f}"
            vote.follow(user_id if accepted else None)
            vote.push(live_score)
            verdict = vote.verdict() if liveness.available() else "unknown"
            if accepted and verdict == "live":
                mark_present(user_id, name, confidence, marked_this_session, method)
            label = name if verdict != "spoof" or not accepted else f"{name}? photo"
            annotate(frame, box, f"{label} ({shown})",
                     MATCH_COLOUR if accepted and verdict == "live" else UNKNOWN_COLOUR)
        for other in faces:
            if other != box:
                annotate(frame, other, "Other face", UNKNOWN_COLOUR)

        cv2.imshow("Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    _report_today()


if __name__ == "__main__":
    run_attendance()
