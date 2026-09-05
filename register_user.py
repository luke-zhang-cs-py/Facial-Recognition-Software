"""
register_user.py
-----------------
Step 1 of the pipeline. Opens the webcam, detects your face with a Haar
cascade, and saves ~30 cropped grayscale face images to disk under
dataset/<user_id>_<name>/. Also creates the user's row in the SQL
database so we have an id to associate the images with.

Usage:
    python register_user.py "Jane Doe"
"""

import os
import sys

import cv2

import db
import paths

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SAMPLES_TO_CAPTURE = 30


def register_user(name):
    db.init_db()
    user_id = db.add_user(name)
    print(f"Created user '{name}' with id={user_id}")

    # paths.user_folder, not the same join written out again -- the folder
    # naming rule lived in two places, and a dataset folder the rest of the
    # project cannot find is a user who silently never gets recognised.
    user_dir = paths.user_folder(user_id, name)
    os.makedirs(user_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check your camera connection/permissions.")
        return

    print("Look at the camera. Capturing face samples... Press 'q' to stop early.")
    count = 0

    while count < SAMPLES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            file_path = os.path.join(user_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"Captured {count}/{SAMPLES_TO_CAPTURE}",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            break  # only take one face per frame to avoid duplicates/blur

        cv2.imshow("Register User - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        print("No face samples captured — registration incomplete. Try again with better lighting.")
    else:
        print(f"Done. Captured {count} samples for '{name}' (user_id={user_id}).")
        print("Next step: run `python train_model.py` to (re)train the recognizer.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python register_user.py "Full Name"')
        sys.exit(1)

    register_user(sys.argv[1])
