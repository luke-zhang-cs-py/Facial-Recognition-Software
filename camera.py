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
import enrollment
import landmarks as facelandmarks
import liveness as faceliveness
import traits as facetraits

import paths
import readout
import vision

BASE_DIR = paths.BASE_DIR
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# The same tuning knob as the CLI version -- and now literally the same one.
# LBPH "confidence" is a distance, so LOWER means a better match.
#
# "Same tuning knobs as the CLI version" was a comment asserting a property
# nothing enforced: this file and attendance.py each declared their own 70,
# and analytics.py reported a third literal 70 back to the user as their
# current setting. Two of the three would have kept saying 70 after the
# first was changed.
CONFIDENCE_THRESHOLD = vision.CONFIDENCE_THRESHOLD
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
# Pose gates, in degrees of yaw. These were bare literals inside
# pose_matches, repeated and unexplained; every other threshold in this
# project is a named constant with its reasoning written above it.
#
# FRONT_YAW is what still counts as looking at the lens. TURN_MIN is far
# enough that the sample carries genuinely new information rather than
# repeating the front stage; TURN_MAX is where enough of the far side of the
# face is hidden that the crop stops being useful.
FRONT_YAW = 12.0
FRONT_ROLL = 12.0
TURN_MIN = 13.0
TURN_MAX = 38.0
TILT_MAX_YAW = 22.0     # chin stages still want a roughly frontal face

PITCH_DELTA = 0.055

CAPTURE_PLAN = [
    {"key": "front", "label": "Look straight at the camera", "count": 10},
    {"key": "left", "label": "Turn your head slightly left", "count": 5},
    {"key": "right", "label": "Turn your head slightly right", "count": 5},
    {"key": "up", "label": "Lift your chin slightly", "count": 5},
    {"key": "down", "label": "Lower your chin slightly", "count": 5},
]


# Two of traits.quality_flags' reasons are about where the head is pointing,
# and registration deliberately asks for off-axis frames: MAX_YAW is 30 while
# the left and right stages accept 13 to 38 degrees. Gating on those would
# reject the frames CAPTURE_PLAN exists to collect. Pose is already the
# stage's own job, via pose_matches.
POSE_FLAGS = frozenset({"turned away", "head tilted"})


def weak_photograph(bgr_face, face):
    """Why this crop is a poor photograph, ignoring where the head points.

    Deliberately `traits.quality_flags` rather than a threshold of its own.
    Those thresholds are already argued for and already measured: nothing
    there gates on absolute brightness or contrast, because benchmarking
    found both encode skin tone (2.15x and 1.61x disparity across race
    groups), so gating on them rejects people instead of photographs. A
    second set of numbers here would be a second thing to keep honest, and
    the one that had not been bias-tested.
    """
    if bgr_face is None or getattr(bgr_face, "size", 0) == 0:
        return []
    metrics = facetraits.quality_metrics(bgr_face)
    x, y, w, h = face["box"]
    geom = {"facePx": int(max(w, h)),
            "yaw": face.get("yaw") or 0.0,
            "roll": face.get("roll") or 0.0}
    flags = facetraits.quality_flags(metrics, geom, grayscale_source=False)
    return [f for f in flags if f not in POSE_FLAGS]


def pose_matches(key, yaw, roll, pitch, baseline_pitch):
    """Is the current head pose the one this stage is asking for?"""
    if yaw is None:
        return False
    if key == "front":
        return abs(yaw) <= FRONT_YAW and (roll is None or abs(roll) <= FRONT_ROLL)
    if key == "left":
        return -TURN_MAX <= yaw <= -TURN_MIN
    if key == "right":
        return TURN_MIN <= yaw <= TURN_MAX
    if pitch is None or baseline_pitch is None:
        return False
    # Looking down foreshortens the lower face, so the nose sits further down
    # between the eyes and the mouth: pitchRatio rises. Looking up lowers it.
    if key == "up":
        return pitch <= baseline_pitch - PITCH_DELTA and abs(yaw) <= TILT_MAX_YAW
    if key == "down":
        return pitch >= baseline_pitch + PITCH_DELTA and abs(yaw) <= TILT_MAX_YAW
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


def open_default_camera():
    """The real device: index 0 over DirectShow.

    This is the only call in this module that needs hardware, which is worth
    stating plainly -- for a long time it made the whole file look
    untestable and left it the least-covered module in the project.
    Everything downstream of it (the grab loop, the mode handlers, the MJPEG
    generator) only needs an object with `isOpened`, `read` and `release`,
    and `read` only has to hand back a numpy frame. A webcam is one way to
    get that. It is not the only way.
    """
    return cv2.VideoCapture(0, cv2.CAP_DSHOW)


class CameraManager:
    def __init__(self, capture_factory=None):
        # Injecting that one call is what makes the capture loop reachable
        # from a test: hand it something whose `read()` returns synthetic
        # frames and the loop runs exactly as it does against a webcam.
        # Existing callers are untouched -- `CameraManager()` still opens
        # the real camera.
        self._open_capture = capture_factory or open_default_camera

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

        # presentation-attack detection
        self._liveness = faceliveness.LivenessVote()
        self._live_verdict = "unknown"
        self._live_score = None

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
            cap = self._open_capture()
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
        user_dir = os.path.join(paths.dataset_dir(), f"{user_id}_{name.replace(' ', '_')}")
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
        if not os.path.exists(paths.model_path()):
            if not self.retrain(reason="(no model yet)"):
                raise CameraError(
                    "Nobody is enrolled yet. Register at least one person first."
                )
        self.start()
        self._load_recognizer()
        with self._lock:
            self._marked_session.clear()
            self._liveness.reset()
            self._live_verdict = "unknown"
            self._mode = MODE_ATTENDANCE
        self._log_event("info", "Attendance mode on")

    def set_idle(self):
        with self._lock:
            self._mode = MODE_IDLE
        self._log_event("info", "Back to idle")

    def _load_recognizer(self):
        """(Re)load trainer.yml, but only when it has actually changed."""
        mtime = os.path.getmtime(paths.model_path())
        with self._lock:
            if self._recognizer is not None and self._model_mtime == mtime:
                return
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(paths.model_path())
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
        if not os.path.isdir(paths.dataset_dir()):
            return False
        newest = 0.0
        for root, _, files in os.walk(paths.dataset_dir()):
            for f in files:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    pass
        if newest == 0.0:
            return False
        if os.path.exists(paths.model_path()) and os.path.getmtime(paths.model_path()) >= newest:
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
                "liveness": {
                    "available": faceliveness.available(),
                    "verdict": self._live_verdict,
                    "score": self._live_score,
                    "samples": self._liveness.samples,
                },
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
                "modelExists": os.path.exists(paths.model_path()),
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

            # Keep an untouched copy BEFORE any overlay is drawn. The mode
            # handlers paint the landmark mesh straight onto `frame`, and the
            # trait read used to run on that same object -- so every quality
            # score, every 68-point fit and every age estimate was measured
            # through a wireframe drawn over the face. Measured cost on a real
            # frame: quality 0.434 -> 0.407, eye-open 0.377 -> 0.297. The
            # analysis has to see what the camera saw, not what we annotated.
            clean = frame.copy()

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
                self._maybe_traits(clean)
            except Exception as exc:  # keep the stream alive, surface the problem
                with self._lock:
                    self._error = str(exc)

            ok, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), vision.STREAM_JPEG_QUALITY])
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
            t = facetraits.analyze(frame, want_embedding=False,
                                   require_detection=True)
        except Exception as exc:
            with self._lock:
                self._live_traits = {"error": str(exc)}
            return

        if t is None:
            return

        with self._lock:
            mode = self._mode

        summary = readout.build(t, frame, mode)

        with self._lock:
            self._live_traits = summary

    def _build_report(self):
        """Summarise the enrollment that just finished.

        The measuring is `enrollment.build`, which reads files and database
        rows and touches nothing of the camera's. What is left here is the
        part that genuinely is the camera's: lifting the registration state
        out from under the lock, and putting the finished report back.
        """
        with self._lock:
            user_id = self._reg_user_id
            name = self._reg_name
            poses = list(self._reg_poses)
            folder = self._reg_dir

        try:
            report = enrollment.build(user_id, name, poses, folder)
        except Exception as exc:
            self._log_event("error", f"Could not build report: {exc}")
            return

        if report is None:
            return

        with self._lock:
            self._reg_report = report
        self._log_event("success", f"Enrollment report ready for {name}.")

    def _detect(self, frame):
        """Detect faces via traits.detect, so this and the guidance panel
        always agree about whether somebody is there.

        This used to run its own YuNet call with its own Haar fallback while
        traits.analyze ran a different one without a fallback, which meant the
        overlay could be tracking a face the sidebar was calling absent.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows = facetraits.detect(frame)
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

        x, y, w, h = face["box"]
        x0, y0 = max(0, x), max(0, y)
        crop = gray[y0:y + h, x0:x + w]
        # Colour, and taken before _draw_face paints over `frame`. The
        # learned quality model wants the image the camera saw, not the one
        # with a wireframe on it -- the same reason the trait read gets its
        # own clean copy in _capture_loop.
        bgr_crop = frame[y0:y + h, x0:x + w].copy()

        # The gallery used to accept any frame whose head was in the right
        # position, however badly photographed. enrollment.build then
        # measured the sharpness and the quality score and reported them --
        # after the samples were already on disk. The measuring was the only
        # part missing from the gate, and it was already being done.
        weak = weak_photograph(bgr_crop, face) if (matched and crop.size) else []
        if weak:
            # Said out loud, not silently skipped. A stalled counter with no
            # reason on screen looks like the capture has broken; "too dark"
            # is something a person can act on.
            self._draw_face(frame, face, RED, label="; ".join(weak))
            return

        colour = GREEN if matched else AMBER
        self._draw_face(frame, face, colour,
                        label=f"{stage['label']}  {stage_count}/{stage['count']}")

        if not matched:
            return

        if crop.size == 0:
            return
        cv2.imwrite(os.path.join(user_dir, f"{count + 1}.jpg"),
                    cv2.resize(crop, vision.LBPH_INPUT_SIZE))

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
        if not faces:
            return

        # self._liveness / self._live_verdict / self._live_score are one
        # shared instance per camera, not per face -- and status() only ever
        # reports a single verdict/score for the whole camera anyway. Scoring
        # every face in the frame against that one shared vote would blend
        # unrelated people's liveness samples into a single verdict, which is
        # worse than useless with more than one face in view. Rather than a
        # deeper per-face-tracking refactor, attendance and liveness are
        # decided against the single largest face in the frame -- the one
        # most likely to be the person actually presenting -- and any other
        # faces are drawn (so they are not silently invisible) but not
        # recognised, scored, or marked present.
        primary = max(faces, key=lambda f: f["box"][2] * f["box"][3])

        for face in faces:
            x, y, w, h = face["box"]
            x0, y0 = max(0, x), max(0, y)
            crop = gray[y0:y + h, x0:x + w]
            if crop.size == 0:
                continue

            if face is not primary:
                self._draw_face(frame, face, AMBER, label="Other face")
                continue

            # Person, or a picture of one? Scored before recognition, because
            # how confidently we recognise a photograph does not matter.
            live_score = faceliveness.score(frame, face["box"])
            self._liveness.push(live_score)
            verdict = self._liveness.verdict()
            with self._lock:
                self._live_verdict = verdict
                self._live_score = (round(live_score, 3)
                                    if live_score is not None else None)

            face_img = cv2.resize(crop, vision.LBPH_INPUT_SIZE)
            user_id, confidence = recognizer.predict(face_img)

            if confidence < CONFIDENCE_THRESHOLD:
                name = db.get_user_name(user_id) or f"Unknown (id {user_id})"

                if verdict == "spoof":
                    # Recognised, but the frame looks like a presentation
                    # attack. Name the person anyway -- somebody genuine in
                    # bad light needs to know why they are being refused.
                    colour = RED
                    name = f"{name}? photo"
                    with self._lock:
                        self._marked_session.discard(user_id)
                elif verdict == "unknown":
                    colour = AMBER      # still gathering frames
                else:
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
