"""
cli/register_user.py
--------------------
Step 1 of the pipeline. Opens the webcam, detects your face with a Haar
cascade, and saves ~30 cropped grayscale face images to disk under
dataset/<user_id>_<name>/. Also creates the user's row in the SQL
database so we have an id to associate the images with.

Usage:
    python -m cli.register_user "Jane Doe"
"""

import os
import sys

import cv2

from core import db
from core import paths
from core import vision

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SAMPLES_TO_CAPTURE = 30

# Overlay drawing, same plain pair the CLI attendance view uses. BGR.
CAPTURE_COLOUR = (0, 255, 0)
BOX_THICKNESS = 2
LABEL_SCALE = 0.7
LABEL_OFFSET = 10


def create_user(name):
    """Make the row and the folder, and return (user_id, folder).

    `paths.user_folder`, not the same join written out again -- the folder
    naming rule lived in two places, and a dataset folder the rest of the
    project cannot find is a user who silently never gets recognised.
    """
    db.init_db()
    user_id = db.add_user(name)
    print(f"Created user '{name}' with id={user_id}")

    user_dir = paths.user_folder(user_id, name)
    os.makedirs(user_dir, exist_ok=True)
    return user_id, user_dir


def save_sample(gray, box, user_dir, index):
    """Write one cropped face and return the path it went to.

    The crop is resized to the shared LBPH geometry: this is the gallery
    side of the comparison attendance.py makes, so if the two disagree about
    the size, recognition quietly gets worse with nothing to point at.
    """
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], vision.LBPH_INPUT_SIZE)
    path = os.path.join(user_dir, f"{index}.jpg")
    cv2.imwrite(path, face)
    return path


def annotate(frame, box, index):
    """Draw the capture box and the running count onto the preview frame."""
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), CAPTURE_COLOUR,
                  BOX_THICKNESS)
    cv2.putText(frame, f"Captured {index}/{SAMPLES_TO_CAPTURE}",
                (x, y - LABEL_OFFSET), cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_SCALE, CAPTURE_COLOUR, BOX_THICKNESS)


def report(count, name, user_id):
    """What happened, and what to do next.

    Zero samples is its own message: the registration "succeeded" in that a
    row and a folder exist, and the next step would train on nothing.
    """
    if count == 0:
        print("No face samples captured — registration incomplete. "
              "Try again with better lighting.")
        return
    print(f"Done. Captured {count} samples for '{name}' (user_id={user_id}).")
    print("Next step: run `python -m pipeline.train_model` to (re)train the recognizer.")


def _open_camera():
    """The capture device, or None with a printed reason."""
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap
    cap.release()
    print("ERROR: Could not open webcam. "
          "Check your camera connection/permissions.")
    return None


def register_user(name):
    user_id, user_dir = create_user(name)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    cap = _open_camera()
    if cap is None:
        return

    print("Look at the camera. Capturing face samples... "
          "Press 'q' to stop early.")
    count = 0

    while count < SAMPLES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=vision.DETECT_SCALE_FACTOR,
            minNeighbors=vision.DETECT_MIN_NEIGHBOURS,
            minSize=vision.MIN_FACE_SIZE,
        )

        # Only the first face per frame: a second person in shot would
        # otherwise have their samples filed under this user's id.
        if len(faces):
            count += 1
            save_sample(gray, faces[0], user_dir, count)
            annotate(frame, faces[0], count)

        cv2.imshow("Register User - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    report(count, name, user_id)


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    if len(arguments) != 1:
        print('Usage: python -m cli.register_user "Full Name"')
        return 1
    register_user(arguments[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
