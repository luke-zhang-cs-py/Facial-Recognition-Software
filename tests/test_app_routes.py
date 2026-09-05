"""Flask routes via the test client. No camera needed for most of them, and
/api/status returning 500 was a real outage that no test would have caught."""
import json

import pytest


@pytest.fixture
def client(isolated_root, tmp_path):
    """Isolate BOTH the database and the dataset directory.

    Isolating only the database is not enough, and finding that out was the
    point: a fresh database with the real dataset/ still on disk is the
    split-brain that let a dataset folder reference a user id with no row.

    This used to patch db.DB_PATH, train_model.DATASET_DIR and
    analytics.DATASET_DIR one at a time -- three attributes, because each
    module had read its path at import and paths.use() could not reach them.
    Fixing that at the source turned three patches into one redirect, which
    is what paths.use() was written to be.
    """
    import db
    db.init_db()
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    return flask_app.app.test_client()


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<html" in r.data.lower()


def test_status_is_json_and_never_500s(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.get_json()
    for key in ("running", "mode", "users", "today", "liveness"):
        assert key in body


def test_json_safe_coerces_numpy():
    """The exact bug: one numpy float32 took down the whole response."""
    import numpy as np
    import app as flask_app
    payload = {"a": np.float32(1.5), "b": [np.int64(2)],
               "c": {"d": np.array([1.0, 2.0])}}
    json.dumps(flask_app.json_safe(payload))


def test_models_endpoint(client):
    r = client.get("/api/models")
    assert r.status_code == 200 and "models" in r.get_json()


def test_report_endpoint_on_empty_db(client):
    r = client.get("/api/report")
    assert r.status_code == 200
    body = r.get_json()
    assert body["users"] == [] and body["all"] == []


def test_analysis_endpoint_starts_idle(client):
    r = client.get("/api/analysis")
    assert r.status_code == 200 and r.get_json()["running"] is False


def test_register_rejects_a_blank_name(client):
    r = client.post("/api/register", json={"name": "   "})
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_train_with_no_samples_fails_cleanly(client):
    r = client.post("/api/train", json={})
    assert r.status_code == 400


def test_identify_requires_an_image(client):
    assert client.post("/api/identify", data={}).status_code == 400


def test_identify_rejects_a_non_image(client):
    from io import BytesIO
    r = client.post("/api/identify",
                    data={"image": (BytesIO(b"not a picture"), "x.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
    assert "image" in r.get_json()["error"].lower()


def test_traits_toggle(client):
    assert client.post("/api/traits", json={"enabled": False}).status_code == 200
    assert client.get("/api/status").get_json()["traitsOn"] is False
    client.post("/api/traits", json={"enabled": True})


def test_attendance_stop_is_always_safe(client):
    assert client.post("/api/attendance/stop", json={}).status_code == 200


def test_camera_stop_when_never_started(client):
    assert client.post("/api/camera/stop", json={}).status_code == 200
