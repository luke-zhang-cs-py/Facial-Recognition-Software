"""
face_attendance.py
===================
Single-file Face Recognition Attendance System.

Combines everything from the multi-file version (db.py, register_user.py,
train_model.py, attendance.py, view_report.py) into one script with a
simple menu / CLI.

Pipeline:
    Webcam frame -> Haar cascade (face detection) -> LBPH recognizer (face ID)
        -> confidence check -> SQL INSERT into attendance table (SQLite)

Requires a machine with a real webcam - won't work in a cloud sandbox.

Setup:
    pip install opencv-python opencv-contrib-python numpy

Usage (interactive menu):
    python face_attendance.py

Usage (direct commands):
    python face_attendance.py register "Jane Doe"
    python face_attendance.py train
    python face_attendance.py run
    python face_attendance.py report
"""

import sys
import os
import sqlite3
from datetime import datetime, date

import cv2
import numpy as np

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

DB_PATH = "attendance.db"
DATASET_DIR = "dataset"
MODEL_PATH = "trainer.yml"
SAMPLES_TO_CAPTURE = 30
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# LBPH "confidence" is a distance: LOWER means more confident.
# Tune based on your lighting/camera. 60-80 is a reasonable range.
CONFIDENCE_THRESHOLD = 70


# ----------------------------------------------------------------------
# Database layer (was db.py)
# ----------------------------------------------------------------------

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't exist yet."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def add_user(name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (name, created_at) VALUES (?, ?)",
        (name, datetime.now().isoformat()),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def get_all_users():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM users ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_user_name(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def already_marked_today(user_id):
    conn = get_connection()
    cur = conn.cursor()
    today_str = date.today().isoformat()
    cur.execute(
        """SELECT 1 FROM attendance
           WHERE user_id = ? AND DATE(timestamp) = ?
           LIMIT 1""",
        (user_id, today_str),
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def log_attendance(user_id, confidence):
    """Insert an attendance record. Returns True if a new row was written."""
    if already_marked_today(user_id):
        return False

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO attendance (user_id, timestamp, confidence) VALUES (?, ?, ?)",
        (user_id, datetime.now().isoformat(), confidence),
    )
    conn.commit()
    conn.close()
    return True


def get_attendance_for_today():
    conn = get_connection()
    cur = conn.cursor()
    today_str = date.today().isoformat()
    cur.execute(
        """SELECT u.name, a.timestamp, a.confidence
           FROM attendance a
           JOIN users u ON u.id = a.user_id
           WHERE DATE(a.timestamp) = ?
           ORDER BY a.timestamp""",
        (today_str,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_attendance():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT u.name, a.timestamp, a.confidence
        FROM attendance a
        JOIN users u ON u.id = a.user_id
        ORDER BY a.timestamp DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


# ----------------------------------------------------------------------
# Step 1: Register a new user (was register_user.py)
# ----------------------------------------------------------------------

def register_user(name):
    init_db()
    user_id = add_user(name)
    print(f"Created user '{name}' with id={user_id}")

    user_dir = os.path.join(DATASET_DIR, f"{user_id}_{name.replace(' ', '_')}")
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
        print("No face samples captured - registration incomplete. Try again with better lighting.")
    else:
        print(f"Done. Captured {count} samples for '{name}' (user_id={user_id}).")
        print("Next step: train the recognizer (menu option 2 / `train` command).")


# ----------------------------------------------------------------------
# Step 2: Train the recognizer (was train_model.py)
# ----------------------------------------------------------------------

def load_training_data():
    faces = []
    labels = []

    if not os.path.isdir(DATASET_DIR):
        return faces, labels

    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # folder_name looks like "3_Jane_Doe" -> label/user_id is 3
        try:
            user_id = int(folder_name.split("_")[0])
        except ValueError:
            print(f"Skipping folder with unexpected name: {folder_name}")
            continue

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(user_id)

    return faces, labels


def train_model():
    faces, labels = load_training_data()

    if len(faces) == 0:
        print("No training images found. Register a user first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    unique_users = len(set(labels))
    print(f"Trained on {len(faces)} images across {unique_users} user(s).")
    print(f"Model saved to {MODEL_PATH}")


# ----------------------------------------------------------------------
# Step 3: Run live attendance (was attendance.py)
# ----------------------------------------------------------------------

def run_attendance():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Register user(s) and train first.")
        return

    init_db()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Attendance system running. Press 'q' to quit.")
    marked_this_session = set()  # avoid re-checking DB every frame

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
                name = get_user_name(user_id) or f"Unknown (id {user_id})"
                color = (0, 255, 0)
                label = f"{name} ({confidence:.0f})"

                if user_id not in marked_this_session:
                    written = log_attendance(user_id, confidence)
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
    for name, timestamp, confidence in get_attendance_for_today():
        print(f"  {name} - {timestamp} (confidence {confidence:.1f})")


# ----------------------------------------------------------------------
# Report (was view_report.py)
# ----------------------------------------------------------------------

def view_report():
    init_db()

    print("=== Registered Users ===")
    users = get_all_users()
    if not users:
        print("  (none yet - register a user first)")
    for user_id, name in users:
        print(f"  [{user_id}] {name}")

    print("\n=== All Attendance Records ===")
    rows = get_all_attendance()
    if not rows:
        print("  (no attendance logged yet)")
    for name, timestamp, confidence in rows:
        print(f"  {timestamp}  |  {name}  |  confidence={confidence:.1f}")


# ----------------------------------------------------------------------
# CLI / menu entry point
# ----------------------------------------------------------------------

def print_menu():
    print("""
Face Recognition Attendance System
-----------------------------------
1. Register a new user
2. Train recognizer
3. Run attendance (live webcam)
4. View report
5. Quit
""")


def interactive_menu():
    init_db()
    while True:
        print_menu()
        choice = input("Choose an option (1-5): ").strip()

        if choice == "1":
            name = input("Enter full name: ").strip()
            if name:
                register_user(name)
            else:
                print("Name cannot be empty.")
        elif choice == "2":
            train_model()
        elif choice == "3":
            run_attendance()
        elif choice == "4":
            view_report()
        elif choice == "5":
            print("Goodbye.")
            break
        else:
            print("Invalid choice, try again.")


def main():
    init_db()

    if len(sys.argv) == 1:
        interactive_menu()
        return

    command = sys.argv[1].lower()

    if command == "register":
        if len(sys.argv) != 3:
            print('Usage: python face_attendance.py register "Full Name"')
            sys.exit(1)
        register_user(sys.argv[2])

    elif command == "train":
        train_model()

    elif command == "run":
        run_attendance()

    elif command == "report":
        view_report()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
