"""analytics.py is 217 statements of pure computation with no hardware
dependency, and was the least covered module in the project."""
import os

import numpy as np
import pytest

import analytics
import paths


def rec(**over):
    r = {"path": "a.jpg", "userId": 1, "sharpness": 300.0, "brightness": 110.0,
         "contrast": 45.0, "quality": 0.5, "facePx": 200, "yaw": 3.0,
         "roll": 1.0, "detected": True, "flags": [], "embedding": None,
         "age": "25-32", "ageConf": 0.6, "gender": "male", "genderConf": 0.8}
    r.update(over)
    return r


def unit(*vals):
    v = np.zeros(128, np.float32)
    for i, x in enumerate(vals):
        v[i] = x
    n = np.linalg.norm(v)
    return v / n if n else v


# ------------------------------------------------------------------ _stats

def test_stats_of_empty_is_none():
    assert analytics._stats([]) is None
    assert analytics._stats([None, None]) is None


def test_stats_ignores_none_values():
    s = analytics._stats([1.0, None, 3.0])
    assert s["mean"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0


# ------------------------------------------------------------------ _modal

def test_modal_reports_agreement_not_just_the_winner():
    m = analytics._modal(["a", "a", "b"], [0.9, 0.8, 0.7])
    assert m["label"] == "a"
    assert m["agreement"] == pytest.approx(2 / 3, abs=0.01)
    assert m["distinctLabels"] == 2


def test_modal_of_nothing_is_none():
    assert analytics._modal([], []) is None
    assert analytics._modal([None, None], [0.1, 0.2]) is None


def test_modal_perfect_agreement():
    m = analytics._modal(["x"] * 5, [0.5] * 5)
    assert m["agreement"] == 1.0 and m["distinctLabels"] == 1


# ------------------------------------------------- relative blur detection

def test_relative_blur_flags_the_outlier_not_an_absolute_level():
    """The whole point: Laplacian variance has no absolute meaning, so blur is
    judged against the person's own median."""
    records = [rec(sharpness=1000.0) for _ in range(5)] + [rec(sharpness=100.0)]
    analytics._mark_relative_blur(records)
    assert "soft focus" in records[-1]["flags"]
    assert all("soft focus" not in r["flags"] for r in records[:5])


def test_relative_blur_scales_with_the_cohort():
    """A person whose sharpest images measure 200 must not have all of them
    condemned just because somebody else's measure 1000."""
    records = [rec(sharpness=200.0) for _ in range(6)]
    analytics._mark_relative_blur(records)
    assert all(not r["flags"] for r in records)


def test_relative_blur_needs_enough_samples_to_have_a_median():
    records = [rec(sharpness=1000.0), rec(sharpness=10.0)]
    analytics._mark_relative_blur(records)
    assert all("soft focus" not in r["flags"] for r in records)


def test_relative_blur_survives_a_zero_median():
    records = [rec(sharpness=0.0) for _ in range(5)]
    analytics._mark_relative_blur(records)


# ---------------------------------------------------------- summarize_user

def test_summary_of_a_good_enrollment():
    records = [rec(yaw=float(y)) for y in range(-10, 12, 1)]
    s = analytics.summarize_user(1, "Alice", records)
    assert s["samples"] == len(records) and s["usable"] == len(records)
    assert s["verdict"] == "good" and s["recommendations"] == []


def test_summary_flags_a_thin_enrollment():
    s = analytics.summarize_user(1, "Bob", [rec() for _ in range(4)])
    assert s["verdict"] == "needs work"
    assert any("samples" in r for r in s["recommendations"])


def test_summary_flags_one_angle_repeated():
    records = [rec(yaw=2.0) for _ in range(25)]
    s = analytics.summarize_user(1, "Carol", records)
    assert s["yawSpread"] == 0.0
    assert any("angle" in r.lower() for r in s["recommendations"])


def test_summary_names_the_worst_files_to_recapture():
    records = [rec() for _ in range(20)]
    records[3]["flags"] = ["blurry", "too dark"]
    records[3]["path"] = os.path.join("d", "bad.jpg")
    s = analytics.summarize_user(1, "Dave", records)
    assert s["worstSamples"] and s["worstSamples"][0]["file"] == "bad.jpg"


def test_summary_warns_when_samples_do_not_redetect():
    records = [rec(detected=False) for _ in range(20)]
    s = analytics.summarize_user(1, "Erin", records)
    assert any("re-detect" in r for r in s["recommendations"])


# ------------------------------------------------------------ sface_analysis

def test_sface_needs_two_people():
    out = analytics.sface_analysis([rec(embedding=unit(1.0))])
    assert out["available"] is False and "2" in out["reason"]


def test_sface_separates_two_distinct_people():
    a = [rec(userId=1, embedding=unit(1.0, 0.02 * i)) for i in range(4)]
    b = [rec(userId=2, embedding=unit(0.02 * i, 1.0)) for i in range(4)]
    out = analytics.sface_analysis(a + b)
    assert out["available"]
    assert out["accuracy"] == 100.0
    assert out["genuine"]["mean"] > out["impostor"]["mean"]
    assert out["margin"] > 0


def test_sface_falls_back_to_calibration_for_a_tiny_gallery():
    a = [rec(userId=1, embedding=unit(1.0, 0.01 * i)) for i in range(4)]
    b = [rec(userId=2, embedding=unit(0.01 * i, 1.0)) for i in range(4)]
    out = analytics.sface_analysis(a + b)
    assert out["localSweepTrusted"] is False
    assert out["warning"] and "2 people" in out["warning"]
    assert out["calibration"]["gallerySize"] == 2


def test_sface_never_matches_an_image_against_itself():
    """Leave-one-out: the diagonal must be excluded or every score is 1.0."""
    a = [rec(userId=1, embedding=unit(1.0)) for _ in range(3)]
    b = [rec(userId=2, embedding=unit(0.0, 1.0)) for _ in range(3)]
    out = analytics.sface_analysis(a + b)
    assert out["impostor"]["max"] < 0.5


# ------------------------------------------------------------- lbph_analysis

def test_lbph_needs_two_people():
    out = analytics.lbph_analysis([rec(userId=1)])
    assert out["available"] is False


# ------------------------------------------------------- iter_sample_paths

def test_iter_sample_paths_on_a_missing_directory(isolated_root):
    """Redirected through paths.use(), which now actually reaches analytics --
    it used to hold its own import-time copy of the dataset directory."""
    assert list(analytics.iter_sample_paths()) == []


def test_iter_sample_paths_skips_unparseable_folders(isolated_root):
    ds = paths.dataset_dir()
    os.makedirs(os.path.join(ds, "7_Alice"))
    open(os.path.join(ds, "7_Alice", "1.jpg"), "wb").write(b"x")
    os.makedirs(os.path.join(ds, "notanid"))
    open(os.path.join(ds, "notanid", "1.jpg"), "wb").write(b"x")
    found = list(analytics.iter_sample_paths())
    assert len(found) == 1 and found[0][0] == 7


def test_scan_of_an_empty_dataset(isolated_root, isolated_db):
    os.makedirs(paths.dataset_dir(), exist_ok=True)
    report = analytics.scan()
    assert report["totalSamples"] == 0
    assert report["users"] == [] and report["orphanFolders"] == []


def test_the_intended_sample_count_matches_the_capture_plan():
    """analytics tells people to aim for INTENDED_SAMPLES; camera decides how
    many a full enrollment actually captures. The number lived in both files,
    as a constant in one and a bare literal inside a sentence in the other.

    Not an import at runtime -- analytics is used by scripts that never open
    a camera -- so this is where the two are kept honest.
    """
    import analytics
    from camera import CAPTURE_PLAN
    assert analytics.INTENDED_SAMPLES == sum(p["count"] for p in CAPTURE_PLAN)
    assert analytics.EXPECTED_SAMPLES < analytics.INTENDED_SAMPLES,         "the complain-at threshold has to be below the target"

