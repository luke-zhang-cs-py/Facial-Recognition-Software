"""The measurement inside `sface_analysis`, and the console report.

`leave_one_out` is the project's actual accuracy figure, and it was buried in
the middle of a 110-line function reachable only by assembling a full record
list with real 128-dimensional embeddings. It takes a similarity matrix and
a label array, so it can be given three vectors and checked by hand.

The `print_*` functions in `analyze_faces.py` were one 45-line function.
Testing printing is usually low value; here it is not, because two of those
branches exist specifically to avoid presenting an absence as a measurement
("pose unavailable" rather than a yaw spread of 0, and an agreement figure
beside every demographic label), and a refactor that lost them would look
completely fine.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis import analytics                        # noqa: E402
from cli import analyze_faces                    # noqa: E402


def similarity_matrix(vectors):
    """Cosine similarities with the diagonal knocked out, as the real code
    builds them: leave-one-out means never matching against yourself."""
    mat = np.vstack(vectors).astype(np.float32)
    sims = mat @ mat.T
    np.fill_diagonal(sims, -np.inf)
    return sims


# ------------------------------------------------------------ leave_one_out

def test_a_sample_closer_to_its_own_person_than_to_anyone_else_is_correct():
    #          person 1        person 1        person 2
    vectors = [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]]
    labels = np.array([1, 1, 2])

    genuine, impostor, correct, unmatchable = analytics.leave_one_out(
        similarity_matrix(vectors), labels)

    # Person 2 has nobody else to be genuinely matched against.
    assert unmatchable == 1
    assert len(genuine) == 2
    assert len(impostor) == 3, "every sample has an impostor available"
    assert correct == 2, "both of person 1's samples should be correct"


def test_a_person_with_one_image_is_unmatchable_not_wrong():
    """The bug this counting exists because of.

    A sample whose owner has no *other* usable image cannot be matched to
    anything under leave-one-out. It used to be recorded as a similarity of
    -inf and averaged in with the rest, which dragged the whole genuine
    distribution to -inf -- and -inf is not valid JSON, so the report
    reached the browser as a parse error rather than as a number.
    """
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    labels = np.array([1, 2])

    genuine, impostor, correct, unmatchable = analytics.leave_one_out(
        similarity_matrix(vectors), labels)

    assert unmatchable == 2
    assert genuine == [], "an unmatchable sample produced a genuine score"
    assert correct == 0
    assert all(np.isfinite(score) for score in impostor), (
        "a -inf leaked into the impostor distribution")


def test_everything_returned_is_json_serialisable():
    """Which is the property the -inf broke, stated directly."""
    import json

    vectors = [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]]
    genuine, impostor, correct, unmatchable = analytics.leave_one_out(
        similarity_matrix(vectors), np.array([1, 1, 2]))
    json.dumps({"genuine": genuine, "impostor": impostor,
                "correct": correct, "unmatchable": unmatchable})


def test_a_sample_nearer_another_person_counts_as_incorrect():
    vectors = [[1.0, 0.0], [0.0, 1.0],          # person 1, far apart
               [0.99, 0.14], [0.14, 0.99]]      # person 2, next to each
    labels = np.array([1, 1, 2, 2])

    _genuine, _impostor, correct, unmatchable = analytics.leave_one_out(
        similarity_matrix(vectors), labels)

    assert unmatchable == 0
    assert correct < 4, "every sample was called correct"


# ---------------------------------------------------------- threshold sweep

def test_the_sweep_covers_every_candidate_threshold():
    genuine = np.array([0.8, 0.7, 0.6])
    impostor = np.array([0.3, 0.2])
    sweep = analytics.threshold_sweep(genuine, impostor)

    assert len(sweep) == len(analytics.SFACE_SWEEP)
    assert [row["threshold"] for row in sweep] == list(analytics.SFACE_SWEEP)


def test_accept_falls_and_false_matches_fall_as_the_threshold_rises():
    """Both rates are monotone in the threshold. If either is not, the sweep
    is computing something other than what its column headings say."""
    genuine = np.array([0.9, 0.7, 0.5, 0.3])
    impostor = np.array([0.6, 0.4, 0.2])
    sweep = analytics.threshold_sweep(genuine, impostor)

    accepts = [row["accept"] for row in sweep]
    false_matches = [row["falseMatch"] for row in sweep]
    assert accepts == sorted(accepts, reverse=True)
    assert false_matches == sorted(false_matches, reverse=True)


# --------------------------------------------------------- confusable pairs

def test_the_most_confusable_pair_comes_first():
    #  user 1        user 2        user 3 (close to user 1)
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.99, 0.14]]
    labels = np.array([1, 2, 3])
    pairs = analytics.confusable_pairs(similarity_matrix(vectors), labels,
                                       [1, 2, 3])

    assert len(pairs) == 3, "three people is three pairs"
    assert {pairs[0]["a"], pairs[0]["b"]} == {1, 3}
    assert pairs == sorted(pairs, key=lambda p: -p["maxSimilarity"])


def test_one_person_has_no_pairs():
    """Which is why sface_analysis refuses below MIN_USERS_FOR_SEPARABILITY
    rather than reporting a score."""
    vectors = [[1.0, 0.0]]
    assert analytics.confusable_pairs(
        similarity_matrix(vectors), np.array([1]), [1]) == []


def test_separability_needs_two_people_and_says_so_when_it_has_one():
    report = analytics.sface_analysis([
        {"userId": 1, "embedding": np.array([1.0, 0.0], dtype=np.float32)},
    ])
    assert report["available"] is False
    assert str(analytics.MIN_USERS_FOR_SEPARABILITY) in report["reason"]


def test_separability_reports_the_one_image_each_case_distinctly():
    """Two people with one image apiece is a different failure from one
    person, and the reason text has to say which."""
    report = analytics.sface_analysis([
        {"userId": 1, "embedding": np.array([1.0, 0.0], dtype=np.float32)},
        {"userId": 2, "embedding": np.array([0.0, 1.0], dtype=np.float32)},
    ])
    assert report["available"] is False
    assert "only one usable image" in report["reason"]


# ------------------------------------------------------------- the printing

def sample_user(**overrides):
    user = {
        "userId": 1, "name": "Ada Lovelace", "verdict": "good",
        "usable": 9, "samples": 10,
        "sharpness": {"mean": 40.0, "min": 12.0, "max": 88.0},
        "brightness": {"mean": 120.0, "min": 90.0, "max": 150.0},
        "contrast": {"mean": 35.0, "min": 20.0, "max": 51.0},
        "quality": {"mean": 0.7, "min": 0.4, "max": 0.9},
        "facePx": {"mean": 180.0, "min": 120.0, "max": 240.0},
        "yawSpread": 12.5, "yawRange": [-20.0, 18.0],
        "flags": {"soft": 2},
        "age": None, "gender": None,
        "worstSamples": [{"file": "7.jpg", "reasons": ["soft", "dark"]}],
        "recommendations": ["recapture 7.jpg"],
    }
    user.update(overrides)
    return user


def test_a_users_block_reports_every_measured_property(capsys):
    analyze_faces.print_user(sample_user())
    printed = capsys.readouterr().out

    assert "Ada Lovelace" in printed
    assert "GOOD" in printed
    for _key, label in analyze_faces.STAT_ROWS:
        assert label in printed, label
    assert "yaw spread 12.5" in printed
    assert "7.jpg" in printed
    assert "recapture 7.jpg" in printed


def test_absent_pose_data_says_unavailable_rather_than_zero(capsys):
    """"unavailable" and "0 degrees" mean opposite things: no landmarks at
    all, against every sample taken from the same angle."""
    analyze_faces.print_pose(sample_user(yawSpread=None, yawRange=None))
    printed = capsys.readouterr().out
    assert "unavailable" in printed
    assert "0" not in printed.replace("no landmarks", "")


def test_a_demographic_label_never_appears_without_its_agreement(capsys):
    """The agreement figure is the point: a label thirty samples disagreed
    about is a different thing from one they all produced."""
    analyze_faces.print_demographics(sample_user(
        age={"label": "25-32", "agreement": 0.6, "samples": 30,
             "distinctLabels": 4},
        gender={"label": "Woman", "agreement": 0.9}))
    printed = capsys.readouterr().out

    assert "25-32" in printed and "60%" in printed
    assert "Woman" in printed and "90%" in printed
    assert "30 samples" in printed and "4 distinct" in printed


def test_a_user_with_no_readings_prints_no_statistic_rows(capsys):
    """Every stat is None when nothing analysed -- printing "mean None"
    would be worse than printing nothing."""
    blank = {key: None for key, _label in analyze_faces.STAT_ROWS}
    analyze_faces.print_stats(blank)
    assert capsys.readouterr().out == ""


def test_an_empty_dataset_says_to_register_somebody(capsys):
    analyze_faces.print_quality({"users": []})
    printed = capsys.readouterr().out
    assert "register someone first" in printed


def test_the_quality_report_prints_a_block_per_user(capsys):
    analyze_faces.print_quality({"users": [sample_user(),
                                           sample_user(userId=2,
                                                       name="Grace Hopper")]})
    printed = capsys.readouterr().out
    assert "Ada Lovelace" in printed and "Grace Hopper" in printed
    assert printed.count("usable      ") == 2


def test_the_usable_bar_scales_and_never_overflows():
    assert analyze_faces.bar(0.0).count("#") == 0
    assert analyze_faces.bar(100.0).count("#") == analyze_faces.BAR_WIDTH
    # Out of range in either direction is clamped, not wrapped.
    assert analyze_faces.bar(-10.0).count("#") == 0
    assert analyze_faces.bar(250.0).count("#") == analyze_faces.BAR_WIDTH
    assert len(analyze_faces.bar(50.0)) == analyze_faces.BAR_WIDTH


def test_a_zero_sample_user_does_not_divide_by_zero(capsys):
    """`samples` is 0 for a folder that exists with nothing analysable in
    it, and the percentage is computed from it."""
    analyze_faces.print_user(sample_user(usable=0, samples=0))
    assert "0%" in capsys.readouterr().out


def test_the_rule_is_one_width_everywhere(capsys):
    analyze_faces.rule()
    first = capsys.readouterr().out.strip()
    analyze_faces.rule("-")
    second = capsys.readouterr().out.strip()
    assert len(first) == len(second) == analyze_faces.RULE_WIDTH
