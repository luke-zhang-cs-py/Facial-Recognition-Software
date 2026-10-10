"""pipeline/enrollment.py: the post-registration report, end to end.

Samples are flat grey squares in tmp_path and traits.analyze is replaced
(test_cov_analytics.fake_analyze), so the report is computed over known
numbers with no detector and no weights. Embeddings for the people already
enrolled are synthetic unit vectors written to the database cache.
"""
import os

import numpy as np
import pytest

from analysis import analytics
from core import db, paths
from pipeline import enrollment

from test_cov_analytics import fake_analyze, squares


@pytest.fixture
def analysed(monkeypatch):
    monkeypatch.setattr(analytics.traits, "analyze", fake_analyze)


def unit(i, j=None, w=0.0):
    v = np.zeros(128, np.float32)
    v[i] = 1.0
    if j is not None:
        v[j] = w
    return v / np.linalg.norm(v)


def cache_embedding(user_id, path, vec):
    db.save_traits({
        "path": path, "user_id": user_id, "mtime": 0.0, "sharpness": 1.0,
        "brightness": 1.0, "contrast": 1.0, "quality": 0.5, "face_px": 10,
        "yaw": 0.0, "roll": 0.0, "detected": 1, "flags": "",
        "embedding": None if vec is None else vec.astype(np.float32).tobytes(),
        "age_label": None, "age_conf": None, "gender_label": None,
        "gender_conf": None})


# ------------------------------------------------------------ the samples

def test_no_folder_means_no_samples(isolated_db, analysed, tmp_path):
    assert enrollment.samples_from_folder(1, None) == []
    assert enrollment.samples_from_folder(1, "") == []
    assert enrollment.samples_from_folder(1, str(tmp_path / "gone")) == []


def test_samples_skip_subfolders_and_unreadable_files(isolated_db, analysed,
                                                      tmp_path):
    uid = db.add_user("Ada")
    folder = str(tmp_path / "1_Ada")
    squares(folder, [90, 95])
    os.makedirs(os.path.join(folder, "thumbs"))
    with open(os.path.join(folder, "notes.txt"), "w") as fh:
        fh.write("not an image")
    records = enrollment.samples_from_folder(uid, folder)
    assert [os.path.basename(r["path"]) for r in records] == ["0.png", "1.png"]


def test_samples_are_never_read_from_the_cache(isolated_db, monkeypatch,
                                               tmp_path):
    """A cached analysis from an earlier enrollment of the same user would
    describe the images that were just replaced."""
    calls = []
    monkeypatch.setattr(analytics.traits, "analyze",
                        lambda img: calls.append(1) or fake_analyze(img))
    uid = db.add_user("Ada")
    folder = str(tmp_path / "f")
    squares(folder, [90])
    enrollment.samples_from_folder(uid, folder)
    enrollment.samples_from_folder(uid, folder)
    assert len(calls) == 2


# ------------------------------------------------------ pose and centroid

def test_pose_coverage_counts_stages_and_ignores_missing_yaws():
    poses = [{"stage": "front", "yaw": -2.0}, {"stage": "front", "yaw": 2.0},
             {"stage": "left", "yaw": None}, {"stage": "right"}]
    stages, spread, span = enrollment.pose_coverage(poses)
    assert stages == {"front": 2, "left": 1, "right": 1}
    assert spread == 2.0 and span == [-2.0, 2.0]


@pytest.mark.parametrize("poses, spread, span", [
    ([], None, None),
    ([{"stage": "front", "yaw": 4.0}], None, [4.0, 4.0]),
    ([{"stage": "front", "yaw": None}], None, None),
])
def test_pose_coverage_is_honest_about_too_little_data(poses, spread, span):
    _, got_spread, got_span = enrollment.pose_coverage(poses)
    assert (got_spread, got_span) == (spread, span)


def test_centroid_of_records_without_embeddings_is_none():
    assert enrollment.centroid_of([{"embedding": None}]) is None
    c = enrollment.centroid_of([{"embedding": unit(0)}, {"embedding": None},
                                {"embedding": unit(1)}])
    assert np.isclose(np.linalg.norm(c), 1.0) and c[0] == pytest.approx(c[1])


# --------------------------------------------------------- nearest other

def test_nearest_other_without_an_embedding_is_none(isolated_db):
    assert enrollment.nearest_other(1, [{"embedding": None}]) is None


def test_nearest_other_with_nobody_else_comparable(isolated_db, tmp_path):
    me = db.add_user("Ada")
    other = db.add_user("Bob")
    cache_embedding(other, str(tmp_path / "b.png"), None)
    cache_embedding(me, str(tmp_path / "a.png"), unit(0))
    assert enrollment.nearest_other(me, [{"embedding": unit(0)}]) is None


def test_nearest_other_finds_the_closest_person_and_skips_yourself(
        isolated_db, tmp_path):
    me = db.add_user("Ada")
    far = db.add_user("Bob")
    near = db.add_user("Cy")
    later = db.add_user("Dee")           # after the best, and further away
    cache_embedding(later, str(tmp_path / "later.png"), unit(0, 2, 3.0))
    cache_embedding(me, str(tmp_path / "me.png"), unit(0))
    cache_embedding(far, str(tmp_path / "far.png"), unit(5))
    cache_embedding(near, str(tmp_path / "near1.png"), unit(0, 1, 1.0))
    cache_embedding(near, str(tmp_path / "near2.png"), unit(3))
    got = enrollment.nearest_other(me, [{"embedding": unit(0)}])
    assert got == {"name": "Cy", "similarity": round(1 / np.sqrt(2), 3)}


# --------------------------------------------------------------- build

def test_build_of_an_empty_enrollment_is_none(isolated_db, analysed, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    assert enrollment.build(1, "Ada", [], str(folder)) is None


def test_build_reports_the_enrollment(isolated_db, analysed):
    uid = db.add_user("Ada")
    folder = paths.user_folder(uid, "Ada")
    squares(folder, [60, 70, 80, 90])
    poses = [{"stage": "front", "yaw": -5.0}, {"stage": "left", "yaw": 15.0}]
    report = enrollment.build(uid, "Ada", poses, folder)

    assert report["userId"] == uid and report["samples"] == 4
    assert report["usable"] == 4 and report["flags"] == {}
    assert report["poseStages"] == {"front": 1, "left": 1}
    assert report["yawSpread"] == 10.0 and report["yawRange"] == [-5.0, 15.0]
    assert report["nearestOther"] is None, "nobody else is enrolled"
    # A gallery of one is calibrated as at least a pair.
    assert report["gallerySize"] == 1
    t2 = enrollment.calibration.recommend_threshold(2)
    assert report["threshold"] == t2[0] and report["thresholdReachable"] is t2[2]
    assert "Only 4 samples" in " ".join(report["recommendations"])
    assert report["sampleSaturation"] == enrollment.calibration.SAMPLE_SATURATION
    assert report["sampleAccuracy"] == round(
        enrollment.calibration.accuracy_for_samples(4), 3)
    assert report["age"]["label"] == "25-32"
