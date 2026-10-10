"""The pure parts of analytics.py, one at a time, and the I/O around them.

summarize_user and lbph_analysis were split into extraction, statistics and
presentation (the golden test proves the split changed nothing). These test
each piece on plain data. The I/O -- the sample cache, the dataset walk,
scan() -- runs against folders of flat grey squares built in tmp_path, with
traits.analyze replaced: nothing here detects, or pretends to detect, a face.
"""
import json
import math
import os

import cv2
import numpy as np
import pytest

from analysis import analytics
from core import db, paths


def rec(**over):
    r = {"path": "a/1.jpg", "userId": 1, "sharpness": 300.0,
         "brightness": 110.0, "contrast": 45.0, "quality": 0.5,
         "facePx": 200, "yaw": 3.0, "roll": 1.0, "detected": True,
         "flags": [], "embedding": None, "age": "25-32", "ageConf": 0.6,
         "gender": "male", "genderConf": 0.8}
    r.update(over)
    return r


def strict_loads(text):
    def refuse(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(text, parse_constant=refuse)


# ------------------------------------------------------------- extraction

def test_user_columns_counts_flags_and_reads_every_column():
    records = [rec(flags=["blurry"], yaw=None),
               rec(flags=["blurry", "underexposed"], yaw=4.0),
               rec(yaw=-2.0, age=None, ageConf=None)]
    cols = analytics.user_columns(records)
    assert cols["flags"] == {"blurry": 2, "underexposed": 1}
    assert cols["usable"] == 1
    assert cols["yaws"] == [4.0, -2.0], "None yaws are not readings"
    assert set(cols["values"]) == set(analytics.STAT_KEYS)
    assert cols["values"]["sharpness"] == [300.0, 300.0, 300.0]
    assert cols["age"] == (["25-32", "25-32", None], [0.6, 0.6, None])
    assert cols["gender"][0] == ["male"] * 3


def test_user_columns_does_not_mark_blur_itself():
    """Relative blur is summarize_user's job, done before extraction;
    extraction only reads."""
    records = [rec(sharpness=1000.0) for _ in range(5)] + [rec(sharpness=10.0)]
    cols = analytics.user_columns(records)
    assert cols["flags"] == {}
    assert all(r["flags"] == [] for r in records)


def test_paths_by_user_keeps_arrival_order():
    records = [rec(userId=2, path="b1"), rec(userId=1, path="a1"),
               rec(userId=2, path="b0")]
    assert dict(analytics.paths_by_user(records)) == {2: ["b1", "b0"],
                                                      1: ["a1"]}


# ------------------------------------------------------------- statistics

@pytest.mark.parametrize("yaws, spread, span", [
    ([], None, None),
    ([5.0], None, (5.0, 5.0)),
    ([-10.0, 10.0], 10.0, (-10.0, 10.0)),
    ([1.04, 1.06, 1.05], 0.0, (1.0, 1.1)),
])
def test_yaw_statistics(yaws, spread, span):
    assert analytics.yaw_statistics(yaws) == (spread, span)


def test_user_statistics_from_columns():
    cols = analytics.user_columns([rec(yaw=0.0, quality=None),
                                   rec(yaw=10.0, quality=None)])
    st = analytics.user_statistics(cols)
    assert st["yawSpread"] == 5.0 and st["yawRange"] == (0.0, 10.0)
    assert st["stats"]["sharpness"] == {"mean": 300.0, "min": 300.0,
                                        "max": 300.0, "std": 0.0}
    assert st["stats"]["quality"] is None
    assert st["age"]["label"] == "25-32" and st["age"]["agreement"] == 1.0
    assert st["gender"]["meanConfidence"] == 0.8


def test_modal_ignores_missing_confidences_for_the_mean():
    m = analytics._modal(["a", "a", "b"], [None, None, 0.9])
    assert m["label"] == "a" and m["meanConfidence"] is None


def test_lbph_sweep_treats_confidence_as_a_distance():
    """Accepted when BELOW the threshold. The inverted reading is the
    plausible-looking curve pointing the wrong way."""
    results = [(1, 1, 35.0), (2, 2, 55.0), (1, 2, 45.0), (2, 2, 200.0)]
    sweep = {row["threshold"]: row for row in analytics._lbph_sweep(results)}
    assert sweep[30] == {"threshold": 30, "accept": 0.0, "falseMatch": 0.0}
    assert sweep[40] == {"threshold": 40, "accept": 25.0, "falseMatch": 0.0}
    assert sweep[50] == {"threshold": 50, "accept": 25.0, "falseMatch": 25.0}
    assert sweep[60] == {"threshold": 60, "accept": 50.0, "falseMatch": 25.0}
    assert sweep[130]["accept"] == 50.0, "a distance of 200 is never accepted"


def test_lbph_statistics_on_known_predictions():
    results = [(1, 1, 20.0), (1, 1, 40.0), (2, 1, 100.0), (2, 2, 60.0)]
    st = analytics.lbph_statistics(results)
    assert st["samples"] == 4 and st["accuracy"] == 75.0
    assert st["confidence"] == {"mean": 55.0, "min": 20.0, "max": 100.0}
    # 50 accepts the two at 20 and 40 with nothing wrong; 70 accepts three
    # right ones, still clean, so it is the most permissive clean setting.
    assert st["best"]["threshold"] == 70
    assert st["best"]["accept"] == 75.0 and st["best"]["falseMatch"] == 0.0


def test_stratify_spreads_each_person_round_robin():
    by_user = {1: ["c", "a", "b"], 2: ["z"]}
    got = analytics._stratify(by_user, 2)
    assert got == [(0, 1, "a"), (1, 1, "b"), (0, 1, "c"), (0, 2, "z")]


# ----------------------------------------------------------- presentation

def test_present_user_shapes_the_report_and_derives_the_verdict():
    cols = analytics.user_columns([rec()])
    st = analytics.user_statistics(cols)
    good = analytics.present_user(3, "Ada", 1, cols, st, [], [])
    assert good["verdict"] == "good" and good["userId"] == 3
    assert good["sharpness"] == st["stats"]["sharpness"]
    work = analytics.present_user(3, "Ada", 1, cols, st, [], ["recapture"])
    assert work["verdict"] == "needs work"
    assert work["recommendations"] == ["recapture"]


def test_present_lbph_reads_the_live_threshold(monkeypatch):
    st = analytics.lbph_statistics([(1, 1, 20.0), (2, 2, 30.0)])
    monkeypatch.setattr(analytics.vision, "CONFIDENCE_THRESHOLD", 55)
    out = analytics.present_lbph(st, 4, 2)
    assert out["available"] is True
    assert out["protocol"] == "4-fold cross-validation of LBPH"
    assert out["currentThreshold"] == 55
    assert out["recommendedThreshold"] == st["best"]["threshold"]


# ---------------------------------------------------------- recommendations

def test_recommendations_for_no_records_is_empty():
    assert analytics._recommendations([], {}, 0, None) == []


def test_each_recommendation_fires_on_its_own():
    n = analytics.EXPECTED_SAMPLES
    base = [rec() for _ in range(n)]
    assert analytics._recommendations(base, {}, n, 10.0) == []

    flagged = analytics._recommendations(base, {}, n // 2, 10.0)
    assert flagged == ["More than a quarter of samples are flagged — recapture."]

    soft = analytics._recommendations(base, {"soft focus": n // 2}, n, 10.0)
    assert len(soft) == 1 and soft[0].startswith(f"{n // 2} samples")

    dark = analytics._recommendations(base, {"underexposed": n // 2}, n, 10.0)
    assert dark == ["Add light in front of the face, not behind it."]
    flat = analytics._recommendations(base, {"no tonal range": 1}, n, 10.0)
    assert flat == dark, "any sample with no tonal range is worth the advice"

    one_angle = analytics._recommendations(base, {}, n, 1.0)
    assert len(one_angle) == 1 and "angle" in one_angle[0]

    thin = analytics._recommendations(base[:3], {}, 3, 10.0)
    assert thin == [f"Only 3 samples; {analytics.INTENDED_SAMPLES} is the "
                    f"intended count."]


# ------------------------------------------------------- non-finite values

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_nan_reading_becomes_null_through_json_safe():
    """The statistics do not filter NaN; json_safe is the backstop that turns
    it into null instead of a token the browser's JSON.parse rejects."""
    import app as web
    stats = analytics._stats([float("nan"), 1.0])
    assert math.isnan(stats["mean"])
    span = analytics.yaw_statistics([float("inf"), 2.0])
    lb = analytics.lbph_statistics([(1, 1, float("nan")), (2, 2, 30.0)])
    payload = web.json_safe({"stats": stats, "yaw": span, "lbph": lb})
    out = strict_loads(json.dumps(payload))
    # numpy propagates NaN through mean, min, max and std alike.
    assert out["stats"] == {"mean": None, "min": None, "max": None,
                            "std": None}
    assert out["yaw"] == [None, [2.0, None]], "inf spread and inf max null"
    assert out["lbph"]["confidence"]["mean"] is None


def test_leave_one_out_with_one_person_has_no_impostors():
    sims = np.array([[-np.inf, 0.9], [0.9, -np.inf]])
    genuine, impostor, correct, unmatchable = analytics.leave_one_out(
        sims, np.array([1, 1]))
    assert genuine == [0.9, 0.9] and impostor == []
    assert correct == 0 and unmatchable == 0


def test_sface_reports_when_nobody_has_a_second_image():
    a = np.zeros(128, np.float32)
    a[0] = 1
    b = np.zeros(128, np.float32)
    b[1] = 1
    out = analytics.sface_analysis([rec(userId=1, embedding=a),
                                    rec(userId=2, embedding=b)])
    assert out["available"] is False
    assert "only one usable image" in out["reason"]


# --------------------------------------------------- LBPH extraction (fake)

class MeanLBPH:
    """Nearest class mean over intensity: the stand-in for LBPH."""
    trained = []

    def train(self, images, labels):
        MeanLBPH.trained.append(sorted(set(np.asarray(labels).tolist())))
        groups = {}
        for img, lab in zip(images, np.asarray(labels).tolist()):
            groups.setdefault(lab, []).append(float(img.mean()))
        self.means = {k: float(np.mean(v)) for k, v in groups.items()}

    def predict(self, img):
        m = float(img.mean())
        lab = min(self.means, key=lambda k: abs(self.means[k] - m))
        return lab, abs(self.means[lab] - m)


def squares(folder, values):
    os.makedirs(folder, exist_ok=True)
    out = []
    for i, v in enumerate(values):
        path = os.path.join(folder, f"{i}.png")
        cv2.imwrite(path, np.full((30, 30), v, np.uint8))
        out.append(path)
    return out


def test_lbph_predictions_hold_every_sample_out_once(tmp_path, monkeypatch):
    MeanLBPH.trained = []
    monkeypatch.setattr(cv2.face, "LBPHFaceRecognizer_create", MeanLBPH)
    by_user = {1: squares(str(tmp_path / "a"), [20, 22, 24]),
               2: squares(str(tmp_path / "b"), [200, 202, 204])}
    results, folds = analytics.lbph_predictions(by_user, folds=3)
    assert folds == [0, 1, 2]
    assert sorted(t for t, _, _ in results) == [1, 1, 1, 2, 2, 2]
    assert all(t == p for t, p, _ in results)
    assert MeanLBPH.trained == [[1, 2]] * 3


def test_load_square_resizes_to_the_lbph_geometry(tmp_path):
    path = squares(str(tmp_path), [77])[0]
    img = analytics._load_square(path)
    assert img.shape == analytics.LBPH_INPUT_SIZE[::-1]
    assert analytics._load_square(str(tmp_path / "missing.png")) is None


def test_train_lbph_refuses_a_single_identity(tmp_path):
    p = squares(str(tmp_path), [10, 20])
    assert analytics._train_lbph([(1, p[0]), (1, p[1])]) is None


def test_real_lbph_trains_on_two_identities(tmp_path):
    """The real recogniser, which needs no weights: trained from pixels."""
    a = squares(str(tmp_path / "a"), [10, 12])
    b = squares(str(tmp_path / "b"), [240, 238])
    model = analytics._train_lbph([(1, a[0]), (1, a[1]), (2, b[0]), (2, b[1])])
    assert model is not None
    scored = analytics._score_fold(model, [(1, a[0]), (2, str(tmp_path / "x"))])
    assert len(scored) == 1 and scored[0][0] == 1


# ------------------------------------------------- the cache and the scan

def fake_analyze(img):
    """What traits.analyze returns, keyed off the square's intensity."""
    v = float(img.mean())
    vec = np.zeros(128, np.float32)
    vec[0 if v < 128 else 1] = 1.0
    vec[2] = v / 1000.0
    vec /= np.linalg.norm(vec)
    return {"sharpness": 100.0 + v, "brightness": v, "contrast": 3.0,
            "qualityScore": 0.6, "facePx": 30, "detected": v > 50,
            "flags": [] if v > 50 else ["low quality"],
            "geometry": {"yaw": v / 10.0, "roll": 0.5},
            "embedding": vec,
            "demographics": {"age": {"label": "25-32", "confidence": 0.7},
                             "gender": {"label": "female", "confidence": 0.9}}}


@pytest.fixture
def analysed(monkeypatch):
    calls = []

    def analyze(img):
        calls.append(img.shape)
        return fake_analyze(img)
    monkeypatch.setattr(analytics.traits, "analyze", analyze)
    return calls


def test_analyze_sample_caches_and_round_trips(isolated_db, tmp_path, analysed):
    uid = db.add_user("Ada")
    path = squares(str(tmp_path), [90])[0]
    first = analytics.analyze_sample(uid, path)
    assert len(analysed) == 1
    assert first["yaw"] == 9.0 and first["age"] == "25-32"

    again = analytics.analyze_sample(uid, path)
    assert len(analysed) == 1, "an unchanged file is read from the cache"
    assert again["embedding"].dtype == np.float32
    assert np.allclose(again["embedding"], first["embedding"])
    assert again["detected"] is True and again["flags"] == []
    assert again["genderConf"] == pytest.approx(0.9)

    analytics.analyze_sample(uid, path, use_cache=False)
    assert len(analysed) == 2, "--refresh re-reads the image"


def test_analyze_sample_skips_what_it_cannot_read(isolated_db, tmp_path,
                                                  monkeypatch, analysed):
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image")
    assert analytics.analyze_sample(1, str(junk)) is None
    assert analysed == [], "nothing to analyse in an undecodable file"

    monkeypatch.setattr(analytics.traits, "analyze", lambda img: None)
    path = squares(str(tmp_path), [90])[0]
    assert analytics.analyze_sample(1, path) is None


def test_analyze_sample_without_an_embedding_or_demographics(
        isolated_db, tmp_path, monkeypatch):
    def bare(img):
        out = fake_analyze(img)
        out.update(embedding=None, demographics=None, geometry=None)
        return out
    monkeypatch.setattr(analytics.traits, "analyze", bare)
    uid = db.add_user("Ada")
    path = squares(str(tmp_path), [90])[0]
    got = analytics.analyze_sample(uid, path)
    assert got["embedding"] is None and got["age"] is None and got["yaw"] is None
    cached = analytics.analyze_sample(uid, path)
    assert cached["embedding"] is None and cached["gender"] is None


def build_dataset(people):
    """{folder_name: [intensity, ...]} written under the isolated dataset/."""
    for folder, values in people.items():
        squares(os.path.join(paths.dataset_dir(), folder), values)


def test_iter_sample_paths_skips_stray_files_and_subfolders(isolated_root):
    build_dataset({"3_Ada": [90, 91]})
    os.makedirs(os.path.join(paths.dataset_dir(), "3_Ada", "nested"))
    with open(os.path.join(paths.dataset_dir(), "README.txt"), "w") as fh:
        fh.write("not a person")
    found = list(analytics.iter_sample_paths())
    assert [(u, f, os.path.basename(p)) for u, f, p in found] == [
        (3, "3_Ada", "0.png"), (3, "3_Ada", "1.png")]


def test_scan_reports_people_orphans_and_progress(isolated_db, analysed,
                                                  monkeypatch):
    monkeypatch.setattr(cv2.face, "LBPHFaceRecognizer_create", MeanLBPH)
    ada = db.add_user("Ada")
    bob = db.add_user("Bob")
    build_dataset({f"{ada}_Ada": [60, 62, 64, 66],
                   f"{bob}_Bob": [200, 202, 204, 206],
                   "9_Ghost": [90, 92]})
    junk = os.path.join(paths.dataset_dir(), f"{ada}_Ada", "9.png")
    with open(junk, "wb") as fh:
        fh.write(b"broken")
    ticks = []
    report = analytics.scan(progress=lambda d, t: ticks.append((d, t)))

    assert ticks[-1] == (11, 11) and len(ticks) == 11
    assert report["totalSamples"] == 10, "the broken file is not a sample"
    assert report["totalUsers"] == 3
    assert [u["name"] for u in report["users"]] == ["Ada", "Bob", "9_Ghost"]
    assert report["orphanFolders"] == [{"userId": 9, "folder": "9_Ghost",
                                        "samples": 2}]
    assert report["sface"]["available"] and report["lbph"]["available"]
    assert report["lbph"]["users"] == 3
    assert report["notes"][0] == analytics.traits.AGE_CAVEAT


def test_scan_runs_without_a_progress_callback(isolated_db, analysed):
    uid = db.add_user("Ada")
    build_dataset({f"{uid}_Ada": [90]})
    report = analytics.scan(use_cache=False)
    assert report["totalSamples"] == 1 and len(analysed) == 1
    assert report["sface"]["available"] is False
