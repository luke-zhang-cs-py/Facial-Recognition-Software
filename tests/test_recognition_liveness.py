"""The two modules that decide whether somebody gets marked present."""
import numpy as np
import pytest


def test_identify_reports_unknown_when_nobody_is_enrolled(isolated_db, blank_frame):
    import recognition
    r = recognition.identify(blank_frame, gal={})
    assert r["ok"] is False


def test_identify_declines_an_ambiguous_match():
    """Clearing the threshold is not enough. Two near-identical scores mean
    the pair is being separated by noise, and naming a winner presents a coin
    toss as an identification."""
    import recognition
    v = np.zeros(128, np.float32); v[0] = 1.0
    a = np.zeros(128, np.float32); a[0] = 0.99; a[1] = 0.14
    b = np.zeros(128, np.float32); b[0] = 0.985; b[1] = 0.17
    gal = {1: {"name": "A", "centroid": a / np.linalg.norm(a), "samples": 10},
           2: {"name": "B", "centroid": b / np.linalg.norm(b), "samples": 10}}

    import unittest.mock as mock
    with mock.patch.object(recognition, "embed_image", return_value=(v, {"box": [0, 0, 10, 10]})):
        r = recognition.identify(np.zeros((10, 10, 3), np.uint8), gal)
    assert r["margin"] < recognition.MIN_MARGIN
    assert r["match"] is None and r["ambiguous"]
    assert r["verdict"] == "unknown"


def test_identify_accepts_a_clear_match():
    import recognition
    import unittest.mock as mock
    v = np.zeros(128, np.float32); v[0] = 1.0
    a = np.zeros(128, np.float32); a[0] = 1.0
    b = np.zeros(128, np.float32); b[1] = 1.0
    gal = {1: {"name": "A", "centroid": a, "samples": 10},
           2: {"name": "B", "centroid": b, "samples": 10}}
    with mock.patch.object(recognition, "embed_image", return_value=(v, {"box": [0, 0, 10, 10]})):
        r = recognition.identify(np.zeros((10, 10, 3), np.uint8), gal)
    assert r["match"]["name"] == "A"
    assert r["verdict"] == "identified"


def test_identify_threshold_follows_gallery_size():
    import recognition
    import unittest.mock as mock
    v = np.zeros(128, np.float32); v[0] = 1.0
    def gal_of(n):
        out = {}
        for i in range(n):
            c = np.zeros(128, np.float32); c[i % 128] = 1.0
            out[i] = {"name": f"P{i}", "centroid": c, "samples": 5}
        return out
    with mock.patch.object(recognition, "embed_image", return_value=(v, {})):
        small = recognition.identify(np.zeros((4, 4, 3), np.uint8), gal_of(5))
        large = recognition.identify(np.zeros((4, 4, 3), np.uint8), gal_of(120))
    assert large["threshold"] >= small["threshold"]


def test_liveness_vote_needs_evidence_before_deciding():
    import liveness
    v = liveness.LivenessVote(window=7, required=4)
    assert v.verdict() == "unknown"
    v.push(0.9); v.push(0.9)
    assert v.verdict() == "unknown", "must not decide on two frames"
    for _ in range(3):
        v.push(0.9)
    assert v.verdict() == "live"


def test_liveness_vote_calls_a_spoof():
    import liveness
    v = liveness.LivenessVote(window=7, required=4)
    for _ in range(7):
        v.push(0.01)
    assert v.verdict() == "spoof"


def test_liveness_vote_window_slides():
    import liveness
    v = liveness.LivenessVote(window=4, required=3)
    for _ in range(10):
        v.push(0.99)
    assert v.samples == 4
    assert v.verdict() == "live"


def test_liveness_vote_ignores_none_scores():
    import liveness
    v = liveness.LivenessVote()
    v.push(None); v.push(None)
    assert v.samples == 0 and v.verdict() == "unknown"


def test_liveness_reset_clears_history():
    import liveness
    v = liveness.LivenessVote()
    for _ in range(5):
        v.push(0.9)
    v.reset()
    assert v.samples == 0 and v.verdict() == "unknown"


def test_liveness_scores_a_real_frame_as_live(face_image):
    import liveness
    import traits
    if not liveness.available():
        pytest.skip("liveness model not downloaded")
    rows = traits.detect(face_image)
    if not rows:
        pytest.skip("no face in sample")
    s = liveness.score(face_image, traits.geometry(rows[0])["box"])
    assert s is not None and s >= liveness.LIVE_THRESHOLD
