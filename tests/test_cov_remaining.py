"""The remaining branches of calibration, guidance, facemodels, readout and
recognition.

Networks are mocked wherever one would load (facemodels' cv2 constructors,
traits.detect, traits.embed, landmarks.fit), so each test means the same
thing whether or not the weights happen to be on this machine.
"""
import numpy as np
import pytest

from analysis import calibration
from core import facemodels
from pipeline import guidance, readout, recognition

from test_cov_landmarks import neutral


# ------------------------------------------------------------ calibration

def test_risk_table_has_every_measured_threshold():
    table = calibration.risk_table(100)
    assert [r["threshold"] for r in table] == [t for t, _ in calibration.SFACE_FMR]
    assert table[0]["fmrPerPair"] == calibration.SFACE_FMR[0][1]
    risks = [r["galleryRisk"] for r in table]
    assert risks == sorted(risks, reverse=True)
    assert calibration.risk_table(1)[0]["galleryRisk"] == 0.0


def test_describe_says_when_no_threshold_is_enough():
    reachable = calibration.describe(10)
    assert "keeps the chance of any false match" in reachable
    assert calibration.CORPUS in reachable
    huge = calibration.describe(100_000)
    assert "not enough at this scale" in huge
    assert str(calibration.SFACE_FMR[-1][0]) in huge


@pytest.mark.parametrize("half_width, expected", [
    (0, 0.056), (1, 0.056), (5, 0.332), (7, (0.398 + 0.486) / 2),
    (11, (0.543 + 0.584) / 2), (20, 0.800), (40, 0.800),
])
def test_age_band_coverage_clamps_matches_and_interpolates(half_width,
                                                           expected):
    assert calibration.age_band_coverage(half_width) == pytest.approx(expected)


def test_age_band_coverage_without_a_table(monkeypatch):
    monkeypatch.setattr(calibration, "AGE_COVERAGE", {})
    assert calibration.age_band_coverage(3) is None


@pytest.mark.parametrize("n, expected", [
    (0, 0.420), (1, 0.420), (4, (0.790 + 0.891) / 2), (12, 0.916),
    (10, (0.908 + 0.916) / 2), (30, 0.924), (99, 0.924),
])
def test_accuracy_for_samples(n, expected):
    assert calibration.accuracy_for_samples(n) == pytest.approx(expected)


def test_accuracy_for_samples_without_a_table(monkeypatch):
    monkeypatch.setattr(calibration, "SAMPLE_ACCURACY", {})
    assert calibration.accuracy_for_samples(5) is None


def test_describe_samples_below_and_at_saturation():
    below = calibration.describe_samples(5)
    assert below == "5 samples measures ~89% identification; 16 would reach ~92%."
    at = calibration.describe_samples(calibration.SAMPLE_SATURATION)
    assert "where more stop helping (16+)" in at


# ---------------------------------------------------------------- guidance

GOOD = {"detected": True, "faces": 1, "facePx": 200, "yaw": 0.0, "roll": 0.0,
        "sharpness": 300.0, "qualityScore": 0.8}


def test_a_camera_error_is_reported_before_anything_else():
    g = guidance.instruction({**GOOD, "error": "device busy"})
    assert g == {"severity": "block", "message": "Camera error",
                 "detail": "device busy", "ready": False}


def test_a_face_filling_the_frame_is_told_to_move_back():
    g = guidance.instruction({**GOOD, "facePx": 520}, frame_shape=(480, 640))
    assert g["message"] == "Move back" and g["severity"] == "warn"
    assert guidance.instruction({**GOOD, "facePx": 520})["ready"], \
        "without a frame size there is no fraction to judge"


def test_one_eye_closed_only_warns():
    g = guidance.instruction({**GOOD, "parts": {"flags": ["one eye closed"]}})
    assert g["message"] == "Open both eyes" and g["severity"] == "warn"


def test_register_mode_says_hold_still_when_ready():
    g = guidance.instruction(GOOD, mode="register")
    assert g == {"severity": "ok", "message": "Hold still",
                 "detail": "Capturing samples.", "ready": True}


def test_yaw_names_the_way_to_turn():
    left = guidance.instruction({**GOOD, "yaw": 40.0})["detail"]
    right = guidance.instruction({**GOOD, "yaw": -40.0})["detail"]
    assert "to the left" in left and "to the right" in right


def test_face_fraction_without_a_size_or_a_frame_is_zero():
    assert guidance._face_fraction({"facePx": 0}, 1000) == 0.0
    assert guidance._face_fraction({"facePx": 100}, None) == 0.0
    assert guidance._face_fraction({"facePx": 10}, 400) == 0.25
    r = guidance.Reading({"detected": True, "facePx": None}, (10, 10))
    assert r.face_fraction() == 0.0 and r.face_px == 0


def test_checklist_of_nothing_is_empty():
    assert guidance.checklist({}) == [] and guidance.checklist(None) == []


def test_checklist_flags_a_mismatched_pair_of_eyes():
    items = {i["label"]: i["ok"] for i in guidance.checklist(
        {**GOOD, "parts": {"eyeMismatch": 0.5, "flags": []}})}
    assert items["Eyes unobstructed"] is False
    assert all(ok for label, ok in items.items() if label != "Eyes unobstructed")


# --------------------------------------------------------------- facemodels

@pytest.fixture
def weights(monkeypatch, tmp_path):
    monkeypatch.setattr(facemodels.paths, "models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(facemodels, "_cache", {})
    return tmp_path


def touch_all(folder):
    for files, _ in facemodels.SPECS.values():
        for f in files:
            (folder / f).write_bytes(b"x")


def test_get_of_a_missing_model_is_none(weights):
    assert facemodels.get("sface") is None
    assert facemodels._cache == {}
    assert facemodels.lock_for("sface") is facemodels._lock


def test_every_model_is_built_from_its_own_files_once(weights, monkeypatch):
    touch_all(weights)
    built = []
    cv2 = facemodels.cv2

    class Ctor:
        def __init__(self, kind):
            self.kind = kind

        def __call__(self, *args, **kwargs):
            built.append((self.kind, args[:2], kwargs))
            return ("model", self.kind)
    monkeypatch.setattr(cv2.FaceDetectorYN, "create", Ctor("yunet"))
    monkeypatch.setattr(cv2.FaceRecognizerSF, "create", Ctor("sface"))
    monkeypatch.setattr(cv2.dnn, "readNet", Ctor("readNet"))
    monkeypatch.setattr(cv2.dnn, "readNetFromCaffe", Ctor("caffe"))
    for name in ("yunet", "sface", "ediffiqa", "age", "gender"):
        assert facemodels.get(name) is facemodels.get(name)
    kinds = [b[0] for b in built]
    assert kinds == ["yunet", "sface", "readNet", "caffe", "caffe"], \
        "each model is constructed exactly once"
    p = facemodels.path_for
    assert built[0][1][0] == p("face_detection_yunet_2023mar.onnx")
    assert built[0][2]["score_threshold"] == 0.6
    assert built[3][1] == (p("age_deploy.prototxt"), p("age_net.caffemodel"))
    assert built[4][1] == (p("gender_deploy.prototxt"),
                           p("gender_net.caffemodel"))
    assert facemodels.lock_for("age") is not facemodels._lock


def test_an_unknown_model_name_is_a_key_error():
    with pytest.raises(KeyError):
        facemodels._build("landmarks")


def test_missing_summary_is_none_when_everything_is_present(weights):
    touch_all(weights)
    assert facemodels.missing_summary() is None
    assert all(m["ready"] for m in facemodels.available().values())


# ------------------------------------------------------------------ readout

def summary():
    return {"flags": ["low quality"]}


def test_part_metrics_are_added_with_their_flags(monkeypatch):
    row = np.zeros(15, np.float32)
    row[:4] = [10, 10, 100, 100]
    fitted = []
    monkeypatch.setattr(readout.facetraits, "detect", lambda frame: [row])
    pts = neutral()
    pts[readout.facelandmarks.NOSE_BRIDGE[0], 0] += 30   # one side differs

    def fit(gray, box):
        fitted.append((gray.ndim, box))
        return pts
    monkeypatch.setattr(readout.facelandmarks, "fit", fit)
    out = readout.add_part_metrics(summary(), np.zeros((150, 150, 3), np.uint8))
    assert fitted == [(2, [10, 10, 100, 100])]
    assert out["parts"]["flags"] == ["face partly obscured"]
    assert out["flags"] == ["low quality", "face partly obscured"]


def test_part_metrics_without_a_fit_leave_the_summary_alone(monkeypatch):
    row = np.zeros(15, np.float32)
    row[:4] = [10, 10, 100, 100]
    monkeypatch.setattr(readout.facetraits, "detect", lambda frame: [row])
    monkeypatch.setattr(readout.facelandmarks, "fit", lambda gray, box: None)
    out = readout.add_part_metrics(summary(), np.zeros((150, 150, 3), np.uint8))
    assert out == {"flags": ["low quality"]}


def test_part_metrics_with_no_face_detected(monkeypatch):
    monkeypatch.setattr(readout.facetraits, "detect", lambda frame: [])
    monkeypatch.setattr(readout.facelandmarks, "fit",
                        lambda g, b: pytest.fail("fitted with no face"))
    out = readout.add_part_metrics(summary(), np.zeros((50, 50, 3), np.uint8))
    assert out == {"flags": ["low quality"]}


# -------------------------------------------------------------- recognition

def test_embed_image_uses_the_most_prominent_face(monkeypatch):
    rows = [np.r_[[5, 6, 70, 80], np.zeros(10), [0.9]].astype(np.float32),
            np.r_[[1, 1, 10, 10], np.zeros(10), [0.4]].astype(np.float32)]
    embedded = []
    monkeypatch.setattr(recognition.traits, "detect", lambda bgr: rows)
    monkeypatch.setattr(recognition.traits, "embed",
                        lambda bgr, row: embedded.append(row) or np.ones(3))
    vec, geom = recognition.embed_image(np.zeros((100, 100, 3), np.uint8))
    assert embedded[0] is rows[0]
    assert vec.tolist() == [1, 1, 1] and geom["box"] == [5, 6, 70, 80]


def test_embed_image_with_nobody_in_it(monkeypatch):
    monkeypatch.setattr(recognition.traits, "detect", lambda bgr: [])
    assert recognition.embed_image(np.zeros((10, 10, 3), np.uint8)) == (None, None)


def test_identify_when_the_face_cannot_be_embedded(monkeypatch):
    """A face found but no SFace weights: no vector, so no identification."""
    monkeypatch.setattr(recognition, "embed_image", lambda bgr: (None, {"box": 1}))
    gal = {1: {"name": "A", "centroid": np.ones(3) / np.sqrt(3), "samples": 3}}
    r = recognition.identify(np.zeros((10, 10, 3), np.uint8), gal)
    assert r == {"ok": False, "error": "No face found in that image."}


def test_a_single_person_gallery_has_no_runner_up():
    v = np.zeros(4, np.float32)
    v[0] = 1
    r = recognition.match_vector(v, {9: {"name": "Solo", "centroid": v,
                                         "samples": 5}})
    assert r["runnerUp"] is None and r["margin"] is None
    assert r["verdict"] == "identified" and r["ambiguous"] is False


def test_refresh_gallery_embeds_only_new_samples_of_enrolled_people(
        isolated_db, monkeypatch, tmp_path):
    import os
    from core import db, paths
    from test_cov_analytics import squares
    ada = db.add_user("Ada")
    squares(paths.user_folder(ada, "Ada"), [90, 91, 92])
    squares(os.path.join(paths.dataset_dir(), "77_Orphan"), [50])
    calls = []

    def analyze_sample(user_id, path, use_cache=True):
        calls.append(os.path.basename(path))
        return (None if path.endswith("2.png")
                else {"path": path, "embedding": np.ones(4, np.float32)})
    from analysis import analytics
    monkeypatch.setattr(analytics, "analyze_sample", analyze_sample)
    first = paths.user_folder(ada, "Ada") + os.sep + "0.png"
    monkeypatch.setattr(db, "get_cached_traits",
                        lambda path, mtime: ({"embedding": b"cached" * 4}
                                             if path == first else None))
    assert recognition.refresh_gallery() == 1
    assert calls == ["1.png", "2.png"], \
        "cached and orphaned samples are not analysed; an unreadable one " \
        "is not counted"


# ------------------------------------------- refresh_gallery's count (fourth pass)

def fake_traits(embedding):
    """What traits.analyze returns for a detected face; `embedding` is None
    without the SFace weights."""
    return {"sharpness": 50.0, "brightness": 100.0, "contrast": 40.0,
            "qualityScore": 0.8, "facePx": 30, "geometry": {"yaw": 0.0,
                                                            "roll": 0.0},
            "detected": True, "flags": [], "embedding": embedding,
            "demographics": {}}


@pytest.fixture
def two_samples(isolated_db):
    from core import db, paths
    from test_cov_analytics import squares
    ada = db.add_user("Ada")
    squares(paths.user_folder(ada, "Ada"), [90, 91])
    return ada


def test_refresh_gallery_does_not_count_a_sample_with_no_embedding(
        two_samples, monkeypatch):
    """Regression: every analysed sample was counted as embedded, so with no
    SFace weights seed_demo printed "24 samples embedded" over a gallery that
    could recognise nobody."""
    from analysis import analytics
    vector = np.ones(128, np.float32) / np.sqrt(128)
    answers = [fake_traits(None), fake_traits(vector)]
    monkeypatch.setattr(analytics.traits, "analyze",
                        lambda img: answers.pop(0))
    monkeypatch.setattr(facemodels, "have", lambda name: False)
    assert recognition.refresh_gallery() == 1


def test_refresh_gallery_embeds_samples_cached_before_the_weights_arrived(
        two_samples, monkeypatch):
    """Regression: a sample analysed without the weights was cached with no
    embedding and then skipped for good, so fetching the weights afterwards
    never gave that person an SFace centroid."""
    from analysis import analytics
    monkeypatch.setattr(analytics.traits, "analyze",
                        lambda img: fake_traits(None))
    monkeypatch.setattr(facemodels, "have", lambda name: False)
    assert recognition.refresh_gallery() == 0
    assert recognition.gallery() == {}

    vector = np.ones(128, np.float32) / np.sqrt(128)
    monkeypatch.setattr(analytics.traits, "analyze",
                        lambda img: fake_traits(vector))
    monkeypatch.setattr(facemodels, "have", lambda name: True)
    assert recognition.refresh_gallery() == 2
    assert recognition.gallery()[two_samples]["samples"] == 2
    assert recognition.refresh_gallery() == 0, "an embedded sample is not redone"


def test_refresh_gallery_without_the_weights_does_not_redo_cached_samples(
        two_samples, monkeypatch):
    """Re-analysing an embedding-less row is only worth it when an embedding
    can come out of it; otherwise it is the same read, every call."""
    from analysis import analytics
    calls = []

    def analyze(img):
        calls.append(1)
        return fake_traits(None)
    monkeypatch.setattr(analytics.traits, "analyze", analyze)
    monkeypatch.setattr(facemodels, "have", lambda name: False)
    recognition.refresh_gallery()
    recognition.refresh_gallery()
    assert len(calls) == 2, "two samples, each analysed once"
