"""
camera.py
----------
Shared webcam manager for the web UI.

The CLI scripts each opened their own `cv2.VideoCapture(0)` and their own
`cv2.imshow` window. A web app can't do that: several browser tabs may be
watching at once, and there is only one camera. So this module owns the
capture device exactly once, runs a single background thread that grabs and
annotates frames, and hands the most recent JPEG to whoever asks.

The thread is always doing one of three things (`mode`):

    idle         just show the camera, no detection
    register     capture SAMPLES_TO_CAPTURE cropped faces for a new user
    attendance   recognize faces and log them to the SQL attendance table

Registration and attendance are the same logic as register_user.py and
attendance.py — the difference is that the loop lives here and the frames
go out over HTTP instead of into an OpenCV window.
"""

import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2

import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "trainer.yml")
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Same tuning knobs as the CLI version. LBPH "confidence" is a distance,
# so LOWER means a better match.
CONFIDENCE_THRESHOLD = 70
SAMPLES_TO_CAPTURE = 30

MODE_IDLE = "idle"
MODE_REGISTER = "register"
MODE_ATTENDANCE = "attendance"

GREEN = (0, 255, 0)
RED = (0, 0, 255)
GREY = (160, 160, 160)


class CameraError(Exception):
    """Raised for problems the user can actually act on (no camera, no model)."""


class CameraManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._cap = None
        self._thread = None
        self._running = False
        self._latest_jpeg = None

        self._mode = MODE_IDLE
        self._error = None

        # register-mode state
        self._reg_name = None
        self._reg_user_id = None
        self._reg_dir = None
        self._reg_count = 0
        self._reg_finished = False

        # attendance-mode state
        self._marked_session = set()
        self._model_mtime = None
        self._recognizer = None

        self._events = deque(maxlen=40)

        self._cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if self._cascade.empty():
            raise CameraError(f"Could not load Haar cascade at {FACE_CASCADE_PATH}")

    # ---------------------------------------------------------------- events

    def _log_event(self, kind, message):
        self._events.appendleft({
            "kind": kind,
            "message": message,
            "at": datetime.now().strftime("%H:%M:%S"),
        })

    # ------------------------------------------------------------ lifecycle

    def start(self):
        """Open the camera and start the grab thread. Idempotent."""
        with self._lock:
            if self._running:
                return
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                raise CameraError(
                    "Could not open the webcam. Check that it is connected and "
                    "that no other app (Zoom, Teams, the CLI scripts) is using it."
                )
            self._cap = cap
            self._running = True
            self._error = None
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        self._log_event("info", "Camera started")

    def stop(self):
        with self._lock:
            self._running = False
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._mode = MODE_IDLE
            self._latest_jpeg = None
        self._log_event("info", "Camera stopped")

    # ------------------------------------------------------------ mode API

    def start_register(self, name):
        name = (name or "").strip()
        if not name:
            raise CameraError("Enter a name before registering.")

        self.start()
        user_id = db.add_user(name)
        user_dir = os.path.join(DATASET_DIR, f"{user_id}_{name.replace(' ', '_')}")
        os.makedirs(user_dir, exist_ok=True)

        with self._lock:
            self._reg_name = name
            self._reg_user_id = user_id
            self._reg_dir = user_dir
            self._reg_count = 0
            self._reg_finished = False
            self._mode = MODE_REGISTER
        self._log_event("info", f"Registering '{name}' (id {user_id})")
        return user_id

    def start_attendance(self):
        if not os.path.exists(MODEL_PATH):
            raise CameraError(
                "No trained model yet. Register at least one person, then train."
            )
        self.start()
        self._load_recognizer()
        with self._lock:
            self._marked_session.clear()
            self._mode = MODE_ATTENDANCE
        self._log_event("info", "Attendance mode on")

    def set_idle(self):
        with self._lock:
            self._mode = MODE_IDLE
        self._log_event("info", "Back to idle")

    def _load_recognizer(self):
        """(Re)load trainer.yml, but only when it has actually changed."""
        mtime = os.path.getmtime(MODEL_PATH)
        with self._lock:
            if self._recognizer is not None and self._model_mtime == mtime:
                return
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        with self._lock:
            self._recognizer = recognizer
            self._model_mtime = mtime

    def invalidate_model(self):
        """Called after retraining so the next attendance run picks up the new model."""
        with self._lock:
            self._recognizer = None
            self._model_mtime = None

    # ---------------------------------------------------------------- status

    def status(self):
        with self._lock:
            return {
                "running": self._running,
                "mode": self._mode,
                "error": self._error,
                "register": {
                    "name": self._reg_name,
                    "userId": self._reg_user_id,
                    "captured": self._reg_count,
                    "target": SAMPLES_TO_CAPTURE,
                    "finished": self._reg_finished,
                },
                "modelExists": os.path.exists(MODEL_PATH),
                "events": list(self._events),
            }

    # ------------------------------------------------------------ frame loop

    def _loop(self):
        while True:
            with self._lock:
                if not self._running or self._cap is None:
                    break
                cap = self._cap
                mode = self._mode

            ok, frame = cap.read()
            if not ok:
                with self._lock:
                    self._error = "Lost the camera feed."
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)  # mirror, so it reads like a mirror on screen

            try:
                if mode == MODE_REGISTER:
                    self._handle_register(frame)
                elif mode == MODE_ATTENDANCE:
                    self._handle_attendance(frame)
                else:
                    self._handle_idle(frame)
            except Exception as exc:  # keep the stream alive, surface the problem
                with self._lock:
                    self._error = str(exc)

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()

            time.sleep(0.01)

    def _detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return gray, faces

    def _handle_idle(self, frame):
        _, faces = self._detect(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREY, 2)

    def _handle_register(self, frame):
        gray, faces = self._detect(frame)

        for (x, y, w, h) in faces:
            with self._lock:
                count = self._reg_count
                user_dir = self._reg_dir
            if count >= SAMPLES_TO_CAPTURE:
                break

            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            cv2.imwrite(os.path.join(user_dir, f"{count + 1}.jpg"), face_img)

            with self._lock:
                self._reg_count += 1
                count = self._reg_count

            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
            cv2.putText(frame, f"Captured {count}/{SAMPLES_TO_CAPTURE}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
            break  # one face per frame, same as register_user.py

        with self._lock:
            done = self._reg_count >= SAMPLES_TO_CAPTURE
            name = self._reg_name
        if done:
            with self._lock:
                self._reg_finished = True
                self._mode = MODE_IDLE
            self._log_event("success",
                            f"Captured {SAMPLES_TO_CAPTURE} samples for '{name}' — now retrain.")

    def _handle_attendance(self, frame):
        gray, faces = self._detect(frame)
        with self._lock:
            recognizer = self._recognizer
        if recognizer is None:
            return

        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            user_id, confidence = recognizer.predict(face_img)

            if confidence < CONFIDENCE_THRESHOLD:
                name = db.get_user_name(user_id) or f"Unknown (id {user_id})"
                colour = GREEN
                with self._lock:
                    already_seen = user_id in self._marked_session
                    if not already_seen:
                        self._marked_session.add(user_id)
                if not already_seen:
                    if db.log_attendance(user_id, confidence):
                        self._log_event("success", f"Marked {name} present")
                    else:
                        self._log_event("info", f"{name} was already marked today")
            else:
                name = "Unknown"
                colour = RED

            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, f"{name} ({confidence:.0f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    # ------------------------------------------------------------- streaming

    def frames(self):
        """Yield multipart JPEG chunks for an <img> MJPEG stream."""
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with self._lock:
                running = self._running
                jpeg = self._latest_jpeg
            if not running:
                break
            if jpeg is not None:
                yield boundary + jpeg + b"\r\n"
            time.sleep(0.03)  # ~30 fps ceiling


camera = CameraManager()
