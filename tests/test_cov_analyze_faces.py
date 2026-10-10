"""cli/analyze_faces.py end to end: main() over a dataset built in tmp_path.

The dataset is flat grey squares. traits.analyze is replaced, so nothing is
detected or pretends to be -- the subject here is the console report: which
sections each flag combination prints, what an empty dataset and a single
image say, and that missing weights are named before anything else.
LBPH is the deterministic stand-in from test_cov_analytics, so the sweep
numbers do not move with the OpenCV build.
"""
import os
import sys

import cv2
import numpy as np
import pytest

from analysis import analytics
from cli import analyze_faces
from core import db, paths

from test_cov_analytics import MeanLBPH, fake_analyze, squares

REAL_MISSING_SUMMARY = analyze_faces.facemodels.missing_summary

QUALITY = "ENROLLMENT QUALITY"
THRESHOLDS = "RECOGNITION ANALYTICS"


@pytest.fixture
def world(isolated_db, monkeypatch, tmp_path):
    """An isolated root, fake analysis, fake LBPH, and every model present."""
    calls = []

    def analyze(img):
        calls.append(1)
        return fake_analyze(img)
    monkeypatch.setattr(analytics.traits, "analyze", analyze)
    monkeypatch.setattr(cv2.face, "LBPHFaceRecognizer_create", MeanLBPH)
    monkeypatch.setattr(analyze_faces.facemodels, "missing_summary",
                        lambda: None)
    return calls


def people(spec):
    for name, values in spec.items():
        uid = db.add_user(name)
        squares(os.path.join(paths.dataset_dir(), f"{uid}_{name}"), values)


def run(monkeypatch, capsys, *flags):
    monkeypatch.setattr(sys, "argv", ["analyze_faces", *flags])
    code = analyze_faces.main()
    return code, capsys.readouterr().out


TWO = {"Ada": [60, 62, 64, 66, 68], "Bob": [200, 202, 204, 206, 208]}


@pytest.mark.parametrize("flags, quality, thresholds", [
    ((), True, True),
    (("--quality",), True, False),
    (("--thresholds",), False, True),
    (("--quality", "--thresholds"), True, True),
    (("--refresh",), True, True),
    (("--refresh", "--quality"), True, False),
    (("--refresh", "--thresholds"), False, True),
    (("--refresh", "--quality", "--thresholds"), True, True),
])
def test_every_flag_combination_prints_the_right_sections(
        world, monkeypatch, capsys, flags, quality, thresholds):
    people(TWO)
    code, out = run(monkeypatch, capsys, *flags)
    assert code == 0
    assert (QUALITY in out) is quality
    assert (THRESHOLDS in out) is thresholds
    assert ("[1] Ada" in out) is quality
    assert ("LBPH" in out) is thresholds
    # The notes close every report, whichever sections ran.
    assert analytics.traits.AGE_CAVEAT in out


def test_refresh_ignores_the_cache(world, monkeypatch, capsys):
    people(TWO)
    run(monkeypatch, capsys)
    assert len(world) == 10
    run(monkeypatch, capsys)
    assert len(world) == 10, "the second run read every sample from the cache"
    run(monkeypatch, capsys, "--refresh")
    assert len(world) == 20, "--refresh re-analysed every image"


def test_an_empty_dataset_says_so_and_fails(world, monkeypatch, capsys):
    os.makedirs(paths.dataset_dir())
    code, out = run(monkeypatch, capsys)
    assert code == 1
    assert "No samples found under dataset/" in out
    assert "register_user" in out
    assert QUALITY not in out


def test_one_person_with_one_image(world, monkeypatch, capsys):
    people({"Solo": [90]})
    code, out = run(monkeypatch, capsys)
    assert code == 0
    assert "usable      1/1" in out
    assert "Only 1 samples" in out
    # One person: nothing to separate, so both sweeps say why, not a score.
    assert out.count("unavailable:") == 2
    assert "Needs at least 2 registered people (found 1)." in out
    assert "measure separability (found 1)" in out


def test_missing_models_are_named_first(world, monkeypatch, capsys, tmp_path):
    """Pointed at an empty models folder, the real missing_summary runs."""
    empty = tmp_path / "no_models"
    empty.mkdir()
    monkeypatch.setattr(paths, "models_dir", lambda: str(empty))
    monkeypatch.setattr(analyze_faces.facemodels, "missing_summary",
                        REAL_MISSING_SUMMARY)
    people(TWO)
    code, out = run(monkeypatch, capsys, "--quality")
    assert code == 0
    first = out.strip().splitlines()[0]
    assert first.startswith("NOTE: 7 model(s) not downloaded: yunet, sface")
    assert first.endswith("Run: python -m cli.fetch_models")


def test_the_quality_block_names_files_flags_and_demographics(
        world, monkeypatch, capsys):
    # Intensities under 50 come back "low quality" and undetected.
    people({"Ada": [20, 22, 24, 90, 92, 94]})
    code, out = run(monkeypatch, capsys, "--quality")
    assert "NEEDS WORK" in out
    assert "flags       low quality x3" in out
    assert "recapture these first:" in out and "0.png" in out
    assert "age         25-32  (agreement 100% across 6 samples" in out
    assert "gender      female  (agreement 100%)" in out
    assert "pose        yaw spread" in out


# ------------------------------------------------- the printers, directly

def user(**over):
    u = {"userId": 4, "name": "Ada", "verdict": "good", "usable": 0,
         "samples": 0, "sharpness": None, "brightness": None,
         "contrast": None, "quality": None, "facePx": None,
         "yawSpread": None, "yawRange": None, "flags": {}, "age": None,
         "gender": None, "worstSamples": [], "recommendations": []}
    u.update(over)
    return u


def test_a_user_without_pose_says_unavailable_not_zero(capsys):
    analyze_faces.print_user(user())
    out = capsys.readouterr().out
    assert "usable      0/0" in out and " 0%" in out
    assert "pose        unavailable" in out
    for absent in ("sharpness", "flags ", "age ", "gender ", "recapture", "->"):
        assert absent not in out


def test_print_quality_with_no_users(capsys):
    analyze_faces.print_quality({"users": []})
    assert "No samples found under dataset/" in capsys.readouterr().out


def test_bar_is_clamped_and_survives_an_empty_span():
    assert analyze_faces.bar(-5) == "." * analyze_faces.BAR_WIDTH
    assert analyze_faces.bar(500) == "#" * analyze_faces.BAR_WIDTH
    assert analyze_faces.bar(50, width=10) == "#####....."
    assert analyze_faces.bar(3, width=4, lo=3, hi=3) == "...."


def sweep_block(**over):
    b = {"available": True, "protocol": "p", "accuracy": 90.0, "samples": 9,
         "sweep": [{"threshold": 50, "accept": 80.0, "falseMatch": 0.0},
                   {"threshold": 70, "accept": 90.0, "falseMatch": 5.0}],
         "recommendedThreshold": 50, "recommendedAccept": 80.0,
         "recommendedFalseMatch": 0.0, "currentThreshold": 70}
    b.update(over)
    return b


def sface_block(**over):
    from analysis import calibration
    b = sweep_block(
        sweep=[{"threshold": 0.5, "accept": 80.0, "falseMatch": 0.0}],
        recommendedThreshold=0.5, currentThreshold=None,
        genuine={"mean": 0.7}, impostor={"mean": 0.1}, margin=0.6,
        referenceThreshold=0.363,
        weakestPairs=[{"a": 1, "b": 2, "maxSimilarity": 0.41}],
        localSweepThreshold=0.3, localSweepTrusted=False,
        warning="Only 2 people enrolled.",
        calibration={"corpus": "C", "summary": "S", "reachable": True,
                     "riskAtLocalChoice": 0.0123, "gallerySize": 2,
                     "disparity": calibration.FALSE_MATCH_DISPARITY})
    b.update(over)
    return b


def test_print_thresholds_marks_recommended_and_current(capsys):
    analyze_faces.print_thresholds({"lbph": sweep_block(),
                                    "sface": sface_block()})
    out = capsys.readouterr().out
    assert "<- recommended" in out and "<- current default" in out
    assert "NOTE: attendance.py has CONFIDENCE_THRESHOLD = 70; " \
           "your data supports 50." in out
    assert "most confusable pair: user 1 vs user 2" in out
    assert "GALLERY-SIZE CALIBRATION (C)" in out
    assert "! Only 2 people enrolled." in out
    assert "would have said 0.3, which carries a 1.23%" in out
    assert "second factor" not in out
    assert "(unreachable)" in out, "10,000 enrolled is past what SFace reaches"
    assert "Indian faces false-match at 76.0%" in out


def test_print_thresholds_quiet_paths(capsys):
    """A recommendation equal to the default needs no NOTE; a trusted local
    sweep needs no contrast line; an unreachable target says add a factor."""
    lbph = sweep_block(recommendedThreshold=70, recommendedAccept=90.0,
                       recommendedFalseMatch=5.0)
    sface = sface_block(weakestPairs=[], localSweepTrusted=True, warning=None)
    sface["calibration"]["reachable"] = False
    analyze_faces.print_thresholds({"lbph": lbph, "sface": sface})
    out = capsys.readouterr().out
    assert "NOTE:" not in out
    assert "<- recommended, current default" in out
    assert "most confusable" not in out
    assert "For contrast" not in out and "! " not in out
    assert "add a second factor" in out


def test_print_thresholds_without_calibration_or_availability(capsys):
    analyze_faces.print_thresholds({
        "lbph": {"available": False, "reason": "too few"},
        "sface": sface_block(calibration=None)})
    out = capsys.readouterr().out
    assert "unavailable: too few" in out
    assert "NOTE:" not in out
    assert "GALLERY-SIZE" not in out and "genuine mean 0.7" in out


def test_no_local_sweep_threshold_prints_no_contrast(capsys):
    analyze_faces.print_thresholds({
        "lbph": {"available": False, "reason": "r"},
        "sface": sface_block(localSweepThreshold=None)})
    assert "For contrast" not in capsys.readouterr().out


def test_stats_rows_print_only_measured_properties(capsys):
    analyze_faces.print_stats(user(sharpness={"mean": 1, "min": 0, "max": 2},
                                   facePx={"mean": 9, "min": 8, "max": 10}))
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2
    assert out[0].split()[0] == "sharpness" and out[1].startswith("    face px")


def test_the_stub_squares_are_not_faces():
    """Guard on the fixture itself: a flat square has no structure for any
    detector to find, so these tests cannot be passing on a real detection."""
    img = np.full((30, 30), 90, np.uint8)
    assert float(cv2.Laplacian(img, cv2.CV_64F).var()) == 0.0
