"""
app.py
-------
Web front-end for the attendance system — the browser equivalent of
register_user.py / train_model.py / attendance.py / view_report.py.

Run it:
    pip install -r requirements.txt
    python app.py
Then open:  http://127.0.0.1:5001

The webcam is opened by *this process*, not by the browser, so the machine
running this server must be the machine with the camera. That is the same
constraint the CLI scripts had; it just means you run this locally rather
than deploying it somewhere.

Endpoints
---------
GET  /                     the dashboard page
GET  /video_feed            MJPEG stream of the annotated camera frames
GET  /api/status            camera state, mode, register progress, recent events
POST /api/camera/start      open the camera
POST /api/camera/stop       release the camera
POST /api/register          body {name} -> begin capturing samples for a new user
POST /api/train             retrain trainer.yml from everything in dataset/
POST /api/attendance/start  begin recognising + logging
POST /api/attendance/stop   back to idle
GET  /api/report            users + today's and all-time attendance rows

Deliberately binds to 127.0.0.1: this streams a live camera and reads a
database of people's attendance, neither of which should be exposed to the
network by default.
"""

import os
import threading

from flask import Flask, jsonify, render_template, request, Response

# Every sibling module (db.py in particular) resolves its paths relative to
# the working directory, exactly as the CLI scripts assumed. Pin the cwd to
# this file's folder so `python app.py` works from anywhere.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

import db                                  # noqa: E402
import train_model                         # noqa: E402
import analytics                           # noqa: E402
import recognition                         # noqa: E402
import facemodels                          # noqa: E402
from camera import camera, CameraError     # noqa: E402

app = Flask(__name__)

# /api/identify accepts an arbitrary upload. Without a ceiling, a large file
# is a one-request memory-exhaustion vector -- the whole body is read into
# memory before it is decoded. 12 MB is far more than any camera still needs.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.errorhandler(413)
def too_large(_):
    return jsonify({"ok": False,
                    "error": f"Image is larger than "
                             f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}), 413


# A full dataset scan runs five networks over every stored sample, so it is
# seconds-to-minutes of work, not a request. Run it on a worker thread and let
# the page poll for progress.
_analysis = {"running": False, "done": 0, "total": 0, "report": None, "error": None}
_analysis_lock = threading.Lock()


def _analysis_worker(use_cache):
    def progress(done, total):
        with _analysis_lock:
            _analysis["done"], _analysis["total"] = done, total

    try:
        report = analytics.scan(progress=progress, use_cache=use_cache)
        with _analysis_lock:
            _analysis["report"], _analysis["error"] = report, None
    except Exception as exc:
        with _analysis_lock:
            _analysis["error"] = str(exc)
    finally:
        with _analysis_lock:
            _analysis["running"] = False


def fail(exc, code=400):
    return jsonify({"ok": False, "error": str(exc)}), code


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(camera.frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def json_safe(obj):
    """Coerce values so one stray number cannot break the response.

    Two failure modes, both seen here:

    * A single numpy float32 in the trait payload took the entire /api/status
      endpoint down with a 500, which from the browser looked exactly like the
      camera having stopped working -- the page just stopped receiving
      anything.
    * A float that is not finite serialises as a bare NaN or -Infinity token.
      Python's json module writes those happily and reads them back, so it
      looks fine from the server side; the browser's JSON.parse rejects them,
      so the page fails with a parse error that says nothing about which
      field was wrong. sface_analysis produced -Infinity whenever somebody had
      only one usable image.

    Values are coerced at source too. This is the backstop, so a future one
    degrades a single field to null instead of taking the whole page with it.
    """
    import math

    import numpy as np
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


@app.route("/api/status")
def api_status():
    status = json_safe(camera.status())
    status["users"] = [{"id": uid, "name": name} for uid, name in db.get_all_users()]
    status["today"] = [
        {"name": name, "timestamp": ts, "confidence": conf}
        for name, ts, conf in db.get_attendance_for_today()
    ]
    return jsonify(status)


@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    try:
        camera.start()
    except CameraError as exc:
        return fail(exc)
    return jsonify({"ok": True})


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    camera.stop()
    return jsonify({"ok": True})


@app.route("/api/register", methods=["POST"])
def api_register():
    body = request.get_json(force=True, silent=True) or {}
    try:
        user_id = camera.start_register(body.get("name", ""))
    except CameraError as exc:
        return fail(exc)
    return jsonify({"ok": True, "userId": user_id})


@app.route("/api/train", methods=["POST"])
def api_train():
    faces, labels = train_model.load_training_data()
    if not faces:
        return fail("No face samples on disk yet — register someone first.")

    camera.retrain(reason="(manual)")
    return jsonify({
        "ok": True,
        "images": len(faces),
        "users": len(set(labels)),
    })


@app.route("/api/attendance/start", methods=["POST"])
def api_attendance_start():
    try:
        camera.start_attendance()
    except CameraError as exc:
        return fail(exc)
    return jsonify({"ok": True})


@app.route("/api/attendance/stop", methods=["POST"])
def api_attendance_stop():
    camera.set_idle()
    return jsonify({"ok": True})


@app.route("/api/traits", methods=["POST"])
def api_traits_toggle():
    body = request.get_json(force=True, silent=True) or {}
    camera.set_traits_enabled(body.get("enabled", True))
    return jsonify({"ok": True})


@app.route("/api/models")
def api_models():
    return jsonify({
        "models": facemodels.available(),
        "missing": facemodels.missing_summary(),
    })


@app.route("/api/analysis/start", methods=["POST"])
def api_analysis_start():
    body = request.get_json(force=True, silent=True) or {}
    with _analysis_lock:
        if _analysis["running"]:
            return fail("An analysis is already running.", 409)
        _analysis.update(running=True, done=0, total=0, error=None)

    # refresh=True ignores the sample_traits cache and re-reads every image.
    use_cache = not body.get("refresh", False)
    threading.Thread(target=_analysis_worker, args=(use_cache,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/analysis")
def api_analysis():
    with _analysis_lock:
        # json_safe, like every other endpoint that returns measured numbers.
        # This one did not have it, and it is the endpoint that carries the
        # most of them.
        return jsonify(json_safe({
            "running": _analysis["running"],
            "done": _analysis["done"],
            "total": _analysis["total"],
            "error": _analysis["error"],
            "report": _analysis["report"],
        }))


@app.route("/api/identify", methods=["POST"])
def api_identify():
    """Identify the face in an uploaded image.

    Deliberately separate from the camera path, and deliberately not gated on
    liveness. Holding a photograph up to the webcam is a presentation attack
    and is refused there, which is correct -- but it also makes it impossible
    to test recognition against still images of anybody. This endpoint takes
    the image directly and is honest about what it is: an identification test,
    not an attendance mark. Nothing here writes to the attendance table.
    """
    import numpy as np
    import cv2

    upload = request.files.get("image")
    if upload is None:
        return fail("No image uploaded.")
    data = np.frombuffer(upload.read(), np.uint8)
    if data.size == 0:
        return fail("That file was empty.")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return fail("Could not read that as an image.")

    result = recognition.identify(bgr)
    if not result.get("ok"):
        return fail(result.get("error", "Identification failed."))
    return jsonify(json_safe(result))


@app.route("/api/report")
def api_report():
    rows = db.get_all_attendance()
    return jsonify({
        "users": [{"id": uid, "name": name} for uid, name in db.get_all_users()],
        "all": [{"name": n, "timestamp": t, "confidence": c} for n, t, c in rows],
        "today": [{"name": n, "timestamp": t, "confidence": c}
                  for n, t, c in db.get_attendance_for_today()],
    })


if __name__ == "__main__":
    db.init_db()
    users = db.get_all_users()
    print(f"Database ready — {len(users)} registered user(s).")
    # Train before serving if dataset/ has samples the model has not seen, so
    # the app comes up ready to recognise instead of requiring a manual step.
    if camera.ensure_trained():
        print("Model retrained from dataset/ (it was missing or out of date).")
    elif os.path.exists(os.path.join(BASE_DIR, "trainer.yml")):
        print("Model is up to date.")
    print("Open http://127.0.0.1:5001")
    # threaded=True matters: the MJPEG stream holds a request open indefinitely,
    # so a single-threaded server would never answer anything else.
    # No reloader — it would spawn a second process fighting for the webcam.
    app.run(host="127.0.0.1", port=5001, threaded=True, use_reloader=False)
