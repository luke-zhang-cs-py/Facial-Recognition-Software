"""app.py's error branches, through the Flask test client.

The camera is the module singleton, and every method a route calls on it is
replaced per test: no webcam is opened. Models are pointed at an empty
folder where the test is about them being missing. No request leaves the
process.
"""
import io
import json
import time

import numpy as np
import pytest

import app as web
from pipeline.camera import CameraError


@pytest.fixture
def client(isolated_db):
    web.app.config["TESTING"] = True
    return web.app.test_client()


@pytest.fixture
def idle_analysis():
    """Leave the module-level analysis state as it was found."""
    saved = dict(web._analysis)
    web._analysis.update(running=False, done=0, total=0, report=None,
                         error=None)
    yield web._analysis
    web._analysis.clear()
    web._analysis.update(saved)


def raises(exc):
    def f(*a, **k):
        raise exc
    return f


# ------------------------------------------------------- JSON body shapes

@pytest.mark.parametrize("body", [[1, 2], "text", 7, None, True])
def test_non_object_bodies_are_treated_as_empty(client, body, monkeypatch):
    seen = []
    monkeypatch.setattr(web.camera, "set_traits_enabled", seen.append)
    r = client.post("/api/traits", data=json.dumps(body),
                    content_type="application/json")
    assert r.status_code == 200
    assert seen == [True], "an empty body means the default, traits on"
    r = client.post("/api/register", data=json.dumps(body),
                    content_type="application/json")
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_a_body_that_is_not_json_at_all_is_treated_as_empty(client, monkeypatch):
    seen = []
    monkeypatch.setattr(web.camera, "set_traits_enabled", seen.append)
    r = client.post("/api/traits", data="{not json", content_type="text/plain")
    assert r.status_code == 200 and seen == [True]


@pytest.mark.parametrize("enabled", ["false", "off", 0, 1, None, [], {}])
def test_traits_toggle_refuses_anything_but_a_boolean(client, monkeypatch,
                                                      enabled):
    """{"enabled": "false"} was passed to bool(), which is True, so asking
    for traits off turned them on. The page always sends a boolean."""
    seen = []
    monkeypatch.setattr(web.camera, "set_traits_enabled", seen.append)
    r = client.post("/api/traits", json={"enabled": enabled})
    assert r.status_code == 400
    assert "true or false" in r.get_json()["error"]
    assert seen == [], "a refused request changed nothing"


@pytest.mark.parametrize("enabled", [True, False])
def test_traits_toggle_passes_a_boolean_through(client, monkeypatch, enabled):
    seen = []
    monkeypatch.setattr(web.camera, "set_traits_enabled", seen.append)
    assert client.post("/api/traits",
                       json={"enabled": enabled}).status_code == 200
    assert seen == [enabled]


def test_register_reports_a_camera_error(client, monkeypatch):
    monkeypatch.setattr(web.camera, "start_register",
                        raises(CameraError("Could not open the webcam.")))
    r = client.post("/api/register", json={"name": "Ada"})
    assert r.status_code == 400
    assert r.get_json() == {"ok": False, "error": "Could not open the webcam."}


def test_register_returns_the_new_id(client, monkeypatch):
    monkeypatch.setattr(web.camera, "start_register", lambda name: 42)
    r = client.post("/api/register", json={"name": "Ada"})
    assert r.get_json() == {"ok": True, "userId": 42}


# ------------------------------------------------------------ the camera

def test_camera_start_reports_a_camera_error(client, monkeypatch):
    monkeypatch.setattr(web.camera, "start", raises(CameraError("busy")))
    r = client.post("/api/camera/start")
    assert r.status_code == 400 and r.get_json()["error"] == "busy"


def test_camera_start_ok(client, monkeypatch):
    calls = []
    monkeypatch.setattr(web.camera, "start", lambda: calls.append(1))
    assert client.post("/api/camera/start").get_json() == {"ok": True}
    assert calls == [1]


def test_attendance_start_reports_a_camera_error(client, monkeypatch):
    monkeypatch.setattr(web.camera, "start_attendance",
                        raises(CameraError("nobody enrolled")))
    r = client.post("/api/attendance/start")
    assert r.status_code == 400 and r.get_json()["error"] == "nobody enrolled"


def test_attendance_start_ok(client, monkeypatch):
    monkeypatch.setattr(web.camera, "start_attendance", lambda: None)
    assert client.post("/api/attendance/start").get_json() == {"ok": True}


def test_video_feed_streams_the_camera_frames(client, monkeypatch):
    monkeypatch.setattr(web.camera, "frames",
                        lambda: iter([b"--frame\r\n", b"jpeg"]))
    r = client.get("/video_feed")
    assert r.status_code == 200
    assert r.mimetype == "multipart/x-mixed-replace"
    assert r.data == b"--frame\r\njpeg"


# --------------------------------------------------------------- training

def test_train_reports_a_failed_retrain(client, monkeypatch):
    monkeypatch.setattr(web.train_model, "load_training_data",
                        lambda: ([np.zeros((2, 2))], [1]))
    monkeypatch.setattr(web.camera, "retrain", lambda reason: False)
    r = client.post("/api/train")
    assert r.status_code == 400 and "Training failed" in r.get_json()["error"]


def test_train_reports_what_it_trained_on(client, monkeypatch):
    faces = [np.zeros((2, 2))] * 5
    monkeypatch.setattr(web.train_model, "load_training_data",
                        lambda: (faces, [1, 1, 2, 2, 3]))
    reasons = []
    monkeypatch.setattr(web.camera, "retrain",
                        lambda reason: reasons.append(reason) or True)
    r = client.post("/api/train")
    assert r.get_json() == {"ok": True, "images": 5, "users": 3}
    assert reasons == ["(manual)"]


# ----------------------------------------------------------------- models

def test_models_endpoint_names_every_missing_model(client, monkeypatch,
                                                   tmp_path):
    monkeypatch.setattr(web.facemodels.paths, "models_dir",
                        lambda: str(tmp_path))
    body = client.get("/api/models").get_json()
    assert set(body["models"]) == set(web.facemodels.SPECS)
    assert not any(m["ready"] for m in body["models"].values())
    assert body["models"]["age"]["files"] == ["age_deploy.prototxt",
                                              "age_net.caffemodel"]
    assert body["missing"].startswith("7 model(s) not downloaded")
    assert "python -m cli.fetch_models" in body["missing"]


def test_models_endpoint_when_everything_is_present(client, monkeypatch,
                                                    tmp_path):
    for files, _ in web.facemodels.SPECS.values():
        for f in files:
            (tmp_path / f).write_bytes(b"x")
    monkeypatch.setattr(web.facemodels.paths, "models_dir",
                        lambda: str(tmp_path))
    body = client.get("/api/models").get_json()
    assert body["missing"] is None
    assert all(m["ready"] and m["files"] == [] for m in body["models"].values())


def test_models_endpoint_with_one_file_of_a_pair(client, monkeypatch, tmp_path):
    (tmp_path / "age_deploy.prototxt").write_bytes(b"x")
    monkeypatch.setattr(web.facemodels.paths, "models_dir",
                        lambda: str(tmp_path))
    age = client.get("/api/models").get_json()["models"]["age"]
    assert age["ready"] is False and age["files"] == ["age_net.caffemodel"]


# --------------------------------------------------------------- identify

def test_an_oversized_upload_is_a_413_with_a_json_reason(client):
    big = io.BytesIO(b"\0" * (web.MAX_UPLOAD_BYTES + 1))
    r = client.post("/api/identify", data={"image": (big, "big.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 413
    assert r.get_json() == {"ok": False,
                            "error": "Image is larger than 12 MB."}


def test_an_empty_upload_is_refused(client):
    r = client.post("/api/identify", data={"image": (io.BytesIO(b""), "e.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and r.get_json()["error"] == "That file was empty."


def png_upload():
    import cv2
    ok, buf = cv2.imencode(".png", np.full((20, 20, 3), 128, np.uint8))
    return {"image": (io.BytesIO(buf.tobytes()), "grey.png")}


def test_identify_passes_on_the_recogniser_error(client, monkeypatch):
    monkeypatch.setattr(web.recognition, "identify",
                        lambda bgr: {"ok": False, "error": "No face found."})
    r = client.post("/api/identify", data=png_upload(),
                    content_type="multipart/form-data")
    assert r.status_code == 400 and r.get_json()["error"] == "No face found."


def test_identify_has_a_default_error(client, monkeypatch):
    monkeypatch.setattr(web.recognition, "identify", lambda bgr: {"ok": False})
    r = client.post("/api/identify", data=png_upload(),
                    content_type="multipart/form-data")
    assert r.get_json()["error"] == "Identification failed."


def test_identify_success_is_json_safe(client, monkeypatch):
    got = []

    def identify(bgr):
        got.append(bgr.shape)
        return {"ok": True, "best": {"similarity": np.float32(0.5)},
                "margin": float("nan")}
    monkeypatch.setattr(web.recognition, "identify", identify)
    r = client.post("/api/identify", data=png_upload(),
                    content_type="multipart/form-data")
    assert r.status_code == 200 and got == [(20, 20, 3)]
    assert r.get_json() == {"ok": True, "best": {"similarity": 0.5},
                            "margin": None}


# --------------------------------------------------------------- analysis

def test_a_second_analysis_is_refused_while_one_runs(client, idle_analysis,
                                                     monkeypatch):
    started = []
    monkeypatch.setattr(web.threading, "Thread",
                        lambda **kw: started.append(kw) or _NoThread())
    idle_analysis["running"] = True
    r = client.post("/api/analysis/start", json={})
    assert r.status_code == 409
    assert r.get_json()["error"] == "An analysis is already running."
    assert started == []


class _NoThread:
    def start(self):
        pass


@pytest.mark.parametrize("body, use_cache", [
    ({}, True), ({"refresh": False}, True), ({"refresh": True}, False),
    ([1, 2], True)])
def test_analysis_start_runs_the_scan_on_a_worker(client, idle_analysis,
                                                  monkeypatch, body, use_cache):
    calls = []

    def scan(progress=None, use_cache=True):
        calls.append(use_cache)
        progress(1, 2)
        progress(2, 2)
        return {"totalSamples": 2, "score": float("inf")}
    monkeypatch.setattr(web.analytics, "scan", scan)
    r = client.post("/api/analysis/start", json=body)
    assert r.get_json() == {"ok": True}
    deadline = time.time() + 5
    while web._analysis["running"] and time.time() < deadline:
        time.sleep(0.01)
    status = client.get("/api/analysis").get_json()
    assert calls == [use_cache]
    assert status["running"] is False and status["error"] is None
    assert (status["done"], status["total"]) == (2, 2)
    assert status["report"] == {"totalSamples": 2, "score": None}


@pytest.mark.parametrize("refresh", ["yes", "false", 1, None])
def test_analysis_start_refuses_a_refresh_that_is_not_a_boolean(
        client, idle_analysis, monkeypatch, refresh):
    """"false" is truthy, so it re-read every image -- minutes of work --
    when the caller asked for the cache."""
    started = []
    monkeypatch.setattr(web.threading, "Thread",
                        lambda **kw: started.append(kw) or _NoThread())
    r = client.post("/api/analysis/start", json={"refresh": refresh})
    assert r.status_code == 400 and "true or false" in r.get_json()["error"]
    assert started == [] and idle_analysis["running"] is False


def test_a_failed_scan_is_reported_and_releases_the_lock(idle_analysis,
                                                         monkeypatch):
    monkeypatch.setattr(web.analytics, "scan",
                        raises(RuntimeError("disk on fire")))
    idle_analysis.update(running=True, report={"old": 1})
    web._analysis_worker(True)
    assert idle_analysis["running"] is False
    assert idle_analysis["error"] == "disk on fire"
    assert idle_analysis["report"] == {"old": 1}


# ------------------------------------------------- same machine only

@pytest.mark.parametrize("host, expected", [
    ("127.0.0.1:5001", "127.0.0.1"), ("LOCALHOST", "localhost"),
    ("[::1]:5001", "[::1]"), ("[::1]", "[::1]"), (" evil.example ", "evil.example"),
    ("", ""), (None, ""), ("[::1", "[::1]"),
])
def test_hostname_strips_the_port(host, expected):
    assert web._hostname(host) == expected


@pytest.mark.parametrize("origin", [
    "https://127.0.0.1.evil.example", "http://localhost.evil.example:5001",
    "http://evil.example/127.0.0.1", "null", "file://",
])
def test_lookalike_origins_cannot_post(client, origin, monkeypatch):
    seen = []
    monkeypatch.setattr(web.camera, "set_traits_enabled", seen.append)
    r = client.post("/api/traits", json={"enabled": False},
                    headers={"Origin": origin})
    assert r.status_code == 403
    assert r.get_json()["error"] == "Cross-site requests are refused."
    assert seen == []


def test_ipv6_loopback_origin_can_post(client, monkeypatch):
    monkeypatch.setattr(web.camera, "set_traits_enabled", lambda on: None)
    r = client.post("/api/traits", json={"enabled": True},
                    headers={"Origin": "http://[::1]:5001"})
    assert r.status_code == 200


def test_cross_site_fetch_metadata_wins_over_a_loopback_origin(client):
    r = client.post("/api/camera/stop",
                    headers={"Origin": "http://127.0.0.1:5001",
                             "Sec-Fetch-Site": "Cross-Site"})
    assert r.status_code == 403


def test_a_cross_site_get_is_not_state_changing(client):
    """Reads are protected by the Host check; the Origin check is for
    writes, so a cross-site GET (an <img> tag) is not refused by it."""
    r = client.get("/api/report", headers={"Origin": "https://evil.example",
                                           "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 200


@pytest.mark.parametrize("method", ["put", "patch", "delete"])
def test_every_state_changing_method_is_guarded(client, method):
    r = getattr(client, method)("/api/traits",
                                headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_a_rebound_host_cannot_post_either(client, monkeypatch):
    calls = []
    monkeypatch.setattr(web.camera, "start", lambda: calls.append(1))
    r = client.post("/api/camera/start", headers={"Host": "evil.example"})
    assert r.status_code == 403
    assert "only answers to 127.0.0.1" in r.get_json()["error"]
    assert calls == []
