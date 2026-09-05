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
import landmarks as facelandmarks
import traits as facetraits

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "trainer.yml")
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Same tuning knobs as the CLI version. LBPH "confidence" is a distance,
# so LOWER means a better match.
CONFIDENCE_THRESHOLD = 70
SAMPLES_TO_CAPTURE = 30

# Registration walks through poses instead of taking 30 frames of somebody
# holding still. Thirty near-identical shots teach the recogniser one angle,
# which is exactly what analytics.summarize_user complains about ("every
# sample is the same angle"); a face turned ten degrees at the door then fails
# to match. Each stage is a pose, an instruction, and a quota.
#
# Yaw and roll are absolute (degrees, from YuNet's landmarks). Pitch is
# relative to the person's own neutral, measured during the first stage --
# necessary because pitchRatio depends on face proportions, so one person's
# level head reads differently from another's. Same reasoning as the relative
# blur check.
PITCH_DELTA = 0.055

CAPTURE_PLAN = [
    {"key": "front", "label": "Look straight at the camera", "count": 10},
    {"key": "left", "label": "Turn your head slightly left", "count": 5},
    {"key": "right", "label": "Turn your head slightly right", "count": 5},
    {"key": "up", "label": "Lift your chin slightly", "count": 5},
    {"key": "down", "label": "Lower your chin slightly", "count": 5},
]


def pose_matches(key, yaw, roll, pitch, baseline_pitch):
    """Is the current head pose the one this stage is asking for?"""
    if yaw is None:
        return False
    if key == "front":
        return abs(yaw) <= 12 and (roll is None or abs(roll) <= 12)
    if key == "left":
        return -38 <= yaw <= -13
    if key == "right":
        return 13 <= yaw <= 38
    if pitch is None or baseline_pitch is None:
        return False
    # Looking down foreshortens the lower face, so the nose sits further down
    # between the eyes and the mouth: pitchRatio rises. Looking up lowers it.
    if key == "up":
        return pitch <= baseline_pitch - PITCH_DELTA and abs(yaw) <= 22
    if key == "down":
        return pitch >= baseline_pitch + PITCH_DELTA and abs(yaw) <= 22
    return False

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
        self._reg_stage = 0
        self._reg_stage_count = 0
        self._reg_baseline_pitch = None
        self._reg_front_pitches = []
        self._reg_poses = []          # yaw/roll/pitch actually captured
        self._reg_report = None

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
            self._reg_stage = 0
            self._reg_stage_count = 0
            self._reg_baseline_pitch = None
            self._reg_front_pitches = []
            self._reg_poses = []
            self._reg_report = None
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
                    "target": sum(p["count"] for p in CAPTURE_PLAN),
                    "finished": self._reg_finished,
                    "stage": self._reg_stage,
                    "stageCount": self._reg_stage_count,
                    "stages": [{"key": p["key"], "label": p["label"],
                                "count": p["count"]} for p in CAPTURE_PLAN],
                },
                "report": self._reg_report,
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

        # 68-point part measurements: eyes open, mouth neutral, both halves
        # of the face equally visible. A face can pass every geometric check
        # and still be unusable because the person blinked.
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows = facetraits.detect(frame)
            if rows:
                box = facetraits.geometry(rows[0])["box"]
                pts = facelandmarks.fit(gray, box)
                pm = facelandmarks.metrics(pts)
                if pm:
                    summary["parts"] = pm
                    summary["flags"] = list(summary["flags"]) + pm["flags"]
        except Exception:
            pass

        with self._lock:
            mode = self._mode
        summary["guidance"] = guidance.instruction(
            summary, frame_shape=frame.shape[:2], mode=mode)
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

    def _build_report(self):
        """Summarise the enrollment that just finished.

        Everything here is measured from the samples on disk -- pose coverage,
        image quality, how distinctive the face is against everyone already
        enrolled, and what that implies for the recognition threshold. It is a
        report on the *enrollment*, not a reading of the person.
        """
        import analytics
        import calibration

        with self._lock:
            user_id = self._reg_user_id
            name = self._reg_name
            poses = list(self._reg_poses)
            folder = self._reg_dir

        try:
            records = []
            if folder and os.path.isdir(folder):
                for fn in sorted(os.listdir(folder)):
                    path = os.path.join(folder, fn)
                    if os.path.isfile(path):
                        rec = analytics.analyze_sample(user_id, path, use_cache=False)
                        if rec:
                            records.append(rec)
            if not records:
                return

            summary = analytics.summarize_user(user_id, name, records)

            # Pose coverage: what the staged capture was for.
            yaws = [p["yaw"] for p in poses if p.get("yaw") is not None]
            stages = {}
            for p in poses:
                stages[p["stage"]] = stages.get(p["stage"], 0) + 1

            # How separable is this person from everyone already enrolled?
            import numpy as np
            mine = [r["embedding"] for r in records if r["embedding"] is not None]
            nearest = None
            if mine:
                centroid = np.mean(np.vstack(mine), axis=0)
                n = float(np.linalg.norm(centroid)) or 1.0
                centroid = centroid / n
                best = None
                for other_id, other_name in db.get_all_users():
                    if other_id == user_id:
                        continue
                    rows = db.get_traits_for_user(other_id)
                    vecs = [np.frombuffer(r["embedding"], dtype=np.float32)
                            for r in rows if r["embedding"]]
                    if not vecs:
                        continue
                    sim = float(np.max(np.vstack(vecs) @ centroid))
                    if best is None or sim > best[1]:
                        best = (other_name, sim)
                nearest = ({"name": best[0], "similarity": round(best[1], 3)}
                           if best else None)

            gallery = len(db.get_all_users())
            thr, risk, ok = calibration.recommend_threshold(max(gallery, 2))

            report = {
                "userId": user_id,
                "name": name,
                "samples": len(records),
                "usable": summary["usable"],
                "verdict": summary["verdict"],
                "flags": summary["flags"],
                "recommendations": summary["recommendations"],
                "sharpness": summary["sharpness"],
                "quality": summary["quality"],
                "facePx": summary["facePx"],
                "poseStages": stages,
                "yawSpread": round(float(np.std(yaws)), 1) if len(yaws) > 1 else None,
                "yawRange": ([round(min(yaws), 1), round(max(yaws), 1)]
                             if yaws else None),
                "nearestOther": nearest,
                "gallerySize": gallery,
                "threshold": thr,
                "thresholdRisk": round(risk, 5),
                "thresholdReachable": ok,
                "age": summary.get("age"),
                "fairness": calibration.FAIRNESS,
            }
            with self._lock:
                self._reg_report = report
            self._log_event("success", f"Enrollment report ready for {name}.")
        except Exception as exc:
            self._log_event("error", f"Could not build report: {exc}")

    def _detect(self, frame):
        """Detect faces, preferring YuNet because it returns landmarks.

        The Haar cascade only ever produced a rectangle, which is why every
        overlay in this file used to be a box. YuNet gives five points -- both
        eyes, the nose tip and both mouth corners -- so the overlay can show
        what is actually being measured rather than a shape drawn around it.

        Returns (gray, faces) where each face is a dict with `box` and, when
        YuNet is available, `landmarks`. Falls back to Haar with landmarks
        None so the app still works without the downloaded weights.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rows = facetraits.detect(frame)
        if rows:
            faces = []
            for row in rows:
                g = facetraits.geometry(row)
                face = {"box": tuple(g["box"]), "landmarks": g["landmarks"],
                        "yaw": g["yaw"], "roll": g["roll"],
                        "pitch": g["pitchRatio"], "score": g["score"],
                        "points68": None}
                # ~5 ms on top of detection, and it is what turns "a face is
                # here" into "the eyes are open and nothing is covering it".
                try:
                    face["points68"] = facelandmarks.fit(gray, face["box"])
                except Exception:
                    pass
                faces.append(face)
            return gray, faces

        boxes = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return gray, [{"box": (int(x), int(y), int(w), int(h)), "landmarks": None,
                       "yaw": None, "roll": None, "pitch": None, "score": None}
                      for (x, y, w, h) in boxes]

    def _draw_face(self, frame, face, colour, label=None):
        """Mark the measured points on the face instead of boxing it.

        Five dots where the detector actually found features, joined by a few
        hairlines: eye to eye, the bridge down to the nose, and the mouth
        line. Corner ticks stand in for the bounding box so framing is still
        legible without a solid rectangle around someone's head.
        """
        x, y, w, h = face["box"]
        pts = face.get("landmarks")

        # Corner ticks -- framing without the full box.
        t = max(8, int(0.16 * max(w, h)))
        for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1),
                               (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * t, cy), colour, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * t), colour, 1, cv2.LINE_AA)

        pts68 = face.get("points68")
        if pts68 is not None:
            # One colour for the whole mesh; which colour is the capture state.
            facelandmarks.draw(frame, pts68, colour)
        elif pts:
            p = [(int(round(a)), int(round(b))) for a, b in pts]
            right_eye, left_eye, nose, mouth_r, mouth_l = p
            eye_mid = ((right_eye[0] + left_eye[0]) // 2,
                       (right_eye[1] + left_eye[1]) // 2)

            for a, b in ((right_eye, left_eye), (eye_mid, nose),
                         (nose, mouth_r), (nose, mouth_l), (mouth_r, mouth_l)):
                cv2.line(frame, a, b, colour, 1, cv2.LINE_AA)

            for q in p:
                cv2.circle(frame, q, 3, colour, -1, cv2.LINE_AA)
                cv2.circle(frame, q, 5, colour, 1, cv2.LINE_AA)

        if label:
            cv2.putText(frame, label, (x, max(16, y - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)

    def _handle_idle(self, frame):
        _, faces = self._detect(frame)
        for face in faces:
            self._draw_face(frame, face, BLUE)

    def _handle_register(self, frame):
        """Walk the person through CAPTURE_PLAN, one pose at a time.

        A sample is only written when the head is actually in the pose the
        current stage asked for, so the resulting set spans angles instead of
        being thirty copies of one. Frames in the wrong pose are ignored
        rather than rejected -- the instruction on screen already says what to
        do, and counting failures at somebody is not useful feedback.
        """
        gray, faces = self._detect(frame)
        if not faces:
            return

        face = faces[0]
        yaw, roll, pitch = face.get("yaw"), face.get("roll"), face.get("pitch")

        with self._lock:
            stage_idx = self._reg_stage
            stage_count = self._reg_stage_count
            user_dir = self._reg_dir
            count = self._reg_count
            baseline = self._reg_baseline_pitch

        if stage_idx >= len(CAPTURE_PLAN):
            return
        stage = CAPTURE_PLAN[stage_idx]

        matched = pose_matches(stage["key"], yaw, roll, pitch, baseline)
        colour = GREEN if matched else AMBER
        self._draw_face(frame, face, colour,
                        label=f"{stage['label']}  {stage_count}/{stage['count']}")

        if not matched:
            return

        x, y, w, h = face["box"]
        x0, y0 = max(0, x), max(0, y)
        crop = gray[y0:y + h, x0:x + w]
        if crop.size == 0:
            return
        cv2.imwrite(os.path.join(user_dir, f"{count + 1}.jpg"),
                    cv2.resize(crop, (200, 200)))

        with self._lock:
            self._reg_count += 1
            self._reg_stage_count += 1
            self._reg_poses.append({"stage": stage["key"], "yaw": yaw,
                                    "roll": roll, "pitch": pitch})
            if stage["key"] == "front" and pitch is not None:
                self._reg_front_pitches.append(pitch)
            stage_count = self._reg_stage_count
            total = self._reg_count

        if stage_count >= stage["count"]:
            with self._lock:
                # The neutral pitch measured during the front stage is what the
                # up/down stages are compared against.
                if stage["key"] == "front" and self._reg_front_pitches:
                    vals = sorted(self._reg_front_pitches)
                    self._reg_baseline_pitch = vals[len(vals) // 2]
                self._reg_stage += 1
                self._reg_stage_count = 0
                done = self._reg_stage >= len(CAPTURE_PLAN)
                name = self._reg_name
            if done:
                with self._lock:
                    self._reg_finished = True
                    self._mode = MODE_IDLE
                self._log_event("success",
                                f"Captured {total} samples for '{name}' across "
                                f"{len(CAPTURE_PLAN)} poses.")
                self.retrain(reason=f"after registering {name}")
                self._build_report()
            else:
                self._log_event("info",
                                f"Pose done — next: "
                                f"{CAPTURE_PLAN[stage_idx + 1]['label'].lower()}")

    def _handle_attendance(self, frame):
        gray, faces = self._detect(frame)
        with self._lock:
            recognizer = self._recognizer
        if recognizer is None:
            return

        for face in faces:
            x, y, w, h = face["box"]
            x0, y0 = max(0, x), max(0, y)
            crop = gray[y0:y + h, x0:x + w]
            if crop.size == 0:
                continue
            face_img = cv2.resize(crop, (200, 200))
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

            self._draw_face(frame, face, colour,
                            label=f"{name} ({confidence:.0f})")

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
