"""
attendance.py
--------------
Step 3 of the pipeline — the actual attendance system. Opens the webcam,
detects faces frame-by-frame, runs each detected face through the trained
LBPH recognizer, and if it's confident about who it sees, logs a row into
the SQL `attendance` table (once per person per day).

Usage:
    python attendance.py

Press 'q' to quit.
"""

import os

import cv2

import db
import paths

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# LBPH "confidence" is actually a distance: LOWER means more confident.
# Tune this based on your lighting/camera. 60-80 is a reasonable range.
CONFIDENCE_THRESHOLD = 70


def run_attendance():
    model_path = paths.model_path()
    if not os.path.exists(model_path):
        print("No trained model found. Run register_user.py then train_model.py first.")
        return

    db.init_db()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Attendance system running. Press 'q' to quit.")
    marked_this_session = set()  # avoid spamming console/DB checks every frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))

            user_id, confidence = recognizer.predict(face_img)

            if confidence < CONFIDENCE_THRESHOLD:
                name = db.get_user_name(user_id) or f"Unknown (id {user_id})"
                color = (0, 255, 0)
                label = f"{name} ({confidence:.0f})"

                if user_id not in marked_this_session:
                    written = db.log_attendance(user_id, confidence)
                    marked_this_session.add(user_id)
                    if written:
                        print(f"Logged attendance: {name} at confidence {confidence:.1f}")
                    else:
                        print(f"{name} already marked present today.")
            else:
                name = "Unknown"
                color = (0, 0, 255)
                label = f"{name} ({confidence:.0f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        cv2.imshow("Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nToday's attendance:")
    for name, timestamp, confidence in db.get_attendance_for_today():
        print(f"  {name} - {timestamp} (confidence {confidence:.1f})")


if __name__ == "__main__":
    run_attendance()
