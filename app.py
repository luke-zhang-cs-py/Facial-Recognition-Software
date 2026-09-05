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
import facemodels                          # noqa: E402
from camera import camera, CameraError     # noqa: E402

app = Flask(__name__)

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


@app.route("/api/status")
def api_status():
    status = camera.status()
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
        return jsonify({
            "running": _analysis["running"],
            "done": _analysis["done"],
            "total": _analysis["total"],
            "error": _analysis["error"],
            "report": _analysis["report"],
        })


@app.route("/api/report")
def api_report():
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT u.name, a.timestamp, a.confidence
        FROM attendance a
        JOIN users u ON u.id = a.user_id
        ORDER BY a.timestamp DESC
    """)
    rows = cur.fetchall()
    conn.close()

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
