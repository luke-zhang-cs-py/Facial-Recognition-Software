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
import guidance
import traits as facetraits

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "trainer.yml")
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Same tuning knobs as the CLI version. LBPH "confidence" is a distance,
# so LOWER means a better match.
CONFIDENCE_THRESHOLD = 70
SAMPLES_TO_CAPTURE = 30

# A full trait read runs five networks and costs ~130 ms, so it cannot go on
# every frame without collapsing the frame rate. Once a second is plenty for
# a readout a human is looking at.
TRAIT_INTERVAL = 1.0

MODE_IDLE = "idle"
MODE_REGISTER = "register"
MODE_ATTENDANCE = "attendance"

# BGR equivalents of the CSS custom properties in static/css/style.css, so
# what is drawn onto the frame matches the panel around it rather than using
# stock full-saturation OpenCV colours.
GREEN = (138, 201, 94)    # --good    #5ec98a
RED = (94, 106, 223)      # --bad     #df6a5e
AMBER = (74, 192, 224)    # --warn    #e0c04a
BLUE = (255, 163, 77)     # --accent  #4da3ff
GREY = (165, 150, 139)    # --muted   #8b96a5
PANEL = (26, 20, 16)      # --bg-ish, for the instruction bar fill


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

        # live trait readout
        self._traits_on = True
        self._live_traits = None
        self._last_trait_at = 0.0

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
        # Train on demand rather than refusing. The old behaviour told the user
        # to go and press a button that this could press itself; the only case
        # that genuinely cannot proceed is having nobody enrolled at all.
        if not os.path.exists(MODEL_PATH):
            if not self.retrain(reason="(no model yet)"):
                raise CameraError(
                    "Nobody is enrolled yet. Register at least one person first."
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

    def retrain(self, reason=""):
        """Rebuild trainer.yml from dataset/ and reload it.

        Runs on the capture thread, which blocks the video stream for as long
        as training takes. LBPH on a few hundred 200x200 crops is well under a
        second, so that is preferable to the alternative -- a background thread
        writing trainer.yml while the recogniser is reading it.
        """
        import train_model
        try:
            faces, labels = train_model.load_training_data()
            if not faces:
                self._log_event("info", "Nothing to train on yet.")
                return False
            train_model.train()
            self.invalidate_model()
            n_users = len(set(labels))
            suffix = f" {reason}" if reason else ""
            self._log_event("success",
                            f"Model retrained{suffix}: {len(faces)} images, "
                            f"{n_users} {'person' if n_users == 1 else 'people'}.")
            return True
        except Exception as exc:
            self._log_event("error", f"Training failed: {exc}")
            with self._lock:
                self._error = f"Training failed: {exc}"
            return False

    def ensure_trained(self):
        """Train at startup if dataset/ has samples but trainer.yml is missing
        or stale. Means the app is usable straight away after a fresh clone
        with an existing dataset, instead of insisting on a manual step."""
        if not os.path.isdir(DATASET_DIR):
            return False
        newest = 0.0
        for root, _, files in os.walk(DATASET_DIR):
            for f in files:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    pass
        if newest == 0.0:
            return False
        if os.path.exists(MODEL_PATH) and os.path.getmtime(MODEL_PATH) >= newest:
            return False
        return self.retrain(reason="(dataset changed since last training)")

    # ---------------------------------------------------------------- status

    def set_traits_enabled(self, on):
        with self._lock:
            self._traits_on = bool(on)
            if not on:
                self._live_traits = None

    def status(self):
        with self._lock:
            return {
                "running": self._running,
                "mode": self._mode,
                "error": self._error,
                "traitsOn": self._traits_on,
                "liveTraits": self._live_traits,
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
        try:
            self._capture_loop()
        except Exception as exc:
            # Last resort. A dead capture thread with running=True looks
            # identical to a broken camera from the browser's side, so record
            # why and mark the camera down rather than failing silently.
            with self._lock:
                self._error = f"Capture thread stopped: {exc}"
                self._running = False
            self._log_event("error", f"Capture thread stopped: {exc}")

    def _capture_loop(self):
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
                # The readout and the overlay are cosmetic; a failure in either
                # must not take the video feed down with it. They used to sit
                # outside this guard, where one exception would kill the
                # capture thread while `running` stayed True -- the stream
                # simply stopped and nothing said why.
                self._maybe_traits(frame)
                self._draw_guidance(frame)
            except Exception as exc:  # keep the stream alive, surface the problem
                with self._lock:
                    self._error = str(exc)

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()

            time.sleep(0.01)

    def _maybe_traits(self, frame):
        """Refresh the live trait readout at most once every TRAIT_INTERVAL.

        Reads the colour frame straight off the camera rather than a stored
        crop, which is the best input these models ever get: full resolution,
        in colour, before any of register_user.py's grey down-conversion.
        """
        now = time.monotonic()
        with self._lock:
            if not self._traits_on or now - self._last_trait_at < TRAIT_INTERVAL:
                return
            self._last_trait_at = now

        try:
            # require_detection: no face in frame means no demographic guess.
            t = facetraits.analyze(frame.copy(), want_embedding=False,
                                   require_detection=True)
        except Exception as exc:
            with self._lock:
                self._live_traits = {"error": str(exc)}
            return

        if t is None:
            return

        geom = t.get("geometry") or {}
        demo = t.get("demographics") or {}
        # Keep this JSON-safe: no numpy arrays past this point.
        summary = {
            "detected": t["detected"],
            "faces": t["faces"],
            "sharpness": t["sharpness"],
            "brightness": t["brightness"],
            "contrast": t["contrast"],
            "qualityScore": t["qualityScore"],
            "facePx": t["facePx"],
            "yaw": geom.get("yaw"),
            "roll": geom.get("roll"),
            "shadowClip": t.get("shadowClip"),
            "highlightClip": t.get("highlightClip"),
            "dynamicRange": t.get("dynamicRange"),
            "flags": t["flags"],
            "usable": t["usable"],
        }

        with self._lock:
            mode = self._mode
        severity, message, ready = guidance.instruction(
            summary, frame_shape=frame.shape[:2], mode=mode)
        summary["guidance"] = {"severity": severity, "message": message,
                               "ready": ready}
        summary["checklist"] = guidance.checklist(
            summary, frame_shape=frame.shape[:2])
        if t.get("demographicsSkipped"):
            summary["demographicsSkipped"] = t["demographicsSkipped"]
        if demo:
            for key in ("age", "gender"):
                if demo.get(key):
                    summary[key] = {
                        "label": demo[key]["label"],
                        "confidence": demo[key]["confidence"],
                        "uncertain": demo[key]["uncertain"],
                        "runnerUp": demo[key]["runnerUp"],
                    }
            if demo.get("age", {}).get("estimate"):
                summary["age"]["estimate"] = demo["age"]["estimate"]

        with self._lock:
            self._live_traits = summary

    def _draw_guidance(self, frame):
        """Burn the current instruction into the frame.

        People being registered are looking at the camera, not at a sidebar,
        so the instruction has to be where their eyes already are. Drawn on a
        filled bar rather than as bare text, because text over a webcam feed
        is unreadable against whatever happens to be behind it.
        """
        with self._lock:
            live = self._live_traits
        g = (live or {}).get("guidance")
        if not g:
            return

        colour = {"block": RED, "warn": AMBER, "ok": GREEN}.get(g["severity"], GREY)
        text = g["message"]
        h, w = frame.shape[:2]

        # Shrink the text until it fits, rather than letting it run off-frame.
        scale, thick = 0.62, 2
        while scale > 0.34:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            if tw <= w - 28:
                break
            scale -= 0.04
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)

        bar_h = th + 26
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), PANEL, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (0, h - bar_h), (6, h), colour, -1)
        cv2.putText(frame, text, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thick, cv2.LINE_AA)

    def _detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return gray, faces

    def _handle_idle(self, frame):
        _, faces = self._detect(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE, 2)

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
                            f"Captured {SAMPLES_TO_CAPTURE} samples for '{name}'.")
            # Train immediately rather than leaving a "now retrain" instruction
            # the user has to act on. A model that silently lags behind the
            # dataset is the same failure as no model: the person is enrolled,
            # the system does not know them, and nothing says why.
            self.retrain(reason=f"after registering {name}")

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
