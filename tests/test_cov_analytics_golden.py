"""Golden output for analytics.summarize_user and analytics.lbph_analysis.

Both functions were split into extraction, statistics and presentation steps
(notes/CODE_AUDIT.md called them "genuine bloaters mixing extraction,
statistics and presentation"). The golden file was captured from the code as
it was BEFORE that split, and this test asserts the split changed nothing:
same keys, same numbers, same order of recommendations.

No model is involved. LBPH is replaced by a deterministic stand-in (nearest
mean intensity), because a real LBPH distance moves between OpenCV builds and
a golden file that changes with the wheel version is not a golden file. The
"samples" are flat grey squares written into tmp_path at test time; nothing
here is, or resembles, a face.

To regenerate after a DELIBERATE change to either function's output:

    python tests/test_cov_analytics_golden.py

and review the diff of tests/golden/analytics_golden.json like any other
change. Do not regenerate to make a failure go away.
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis import analytics  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden", "analytics_golden.json")


# ------------------------------------------------------------------ fixtures

def rec(**over):
    r = {"path": "x/1.jpg", "userId": 1, "sharpness": 300.0,
         "brightness": 110.0, "contrast": 45.0, "quality": 0.5,
         "facePx": 200, "yaw": 3.0, "roll": 1.0, "detected": True,
         "flags": [], "embedding": None, "age": "25-32", "ageConf": 0.6,
         "gender": "male", "genderConf": 0.8}
    r.update(over)
    return r


def summary_cases():
    """Record lists that between them reach every branch of summarize_user."""
    cases = {}

    cases["good"] = [rec(path=f"g/{i}.jpg", yaw=float(i - 12),
                         sharpness=280.0 + i, quality=0.4 + i / 100)
                     for i in range(25)]

    flagged = [rec(path=f"f/{i}.jpg", yaw=1.0 + (i % 2) * 0.5,
                   sharpness=500.0, age=("25-32" if i % 3 else "38-43"),
                   ageConf=(None if i == 4 else 0.5 + i / 50),
                   gender=("female" if i % 4 else "male"), genderConf=0.7)
               for i in range(12)]
    flagged[0]["sharpness"] = 50.0                      # soft focus
    flagged[1]["sharpness"] = 60.0                      # soft focus
    flagged[2]["flags"] = ["blurry"]
    flagged[2]["sharpness"] = 3.0
    flagged[3]["flags"] = ["underexposed", "no tonal range"]
    flagged[5]["flags"] = ["underexposed"]
    flagged[6]["flags"] = ["underexposed", "turned away", "head tilted"]
    flagged[7]["detected"] = False
    flagged[8]["detected"] = False
    flagged[9]["detected"] = False
    flagged[10]["detected"] = False
    cases["flagged"] = flagged

    cases["no_pose"] = [rec(path=f"n/{i}.jpg", yaw=None, roll=None,
                            age=None, ageConf=None, gender=None,
                            genderConf=None, quality=None)
                        for i in range(22)]

    cases["one_sample"] = [rec(path="o/1.jpg", yaw=7.25)]

    cases["unmeasured"] = [rec(path=f"u/{i}.jpg", sharpness=None,
                               brightness=None, contrast=None, quality=None,
                               facePx=None, yaw=None)
                           for i in range(3)]

    cases["zero_median"] = [rec(path=f"z/{i}.jpg", sharpness=0.0,
                                yaw=float(i * 4)) for i in range(20)]

    cases["empty"] = []
    return cases


class FakeLBPH:
    """Nearest class mean over pixel intensity. Deterministic everywhere."""

    def train(self, images, labels):
        means = {}
        for img, lab in zip(images, np.asarray(labels).tolist()):
            means.setdefault(int(lab), []).append(float(img.mean()))
        self.means = {k: float(np.mean(v)) for k, v in means.items()}

    def predict(self, img):
        m = float(img.mean())
        lab = min(sorted(self.means), key=lambda k: abs(self.means[k] - m))
        return lab, abs(self.means[lab] - m) * 2.5 + 25.0


def write_square(path, value):
    """A flat grey square: no face, no texture, just a known intensity."""
    cv2.imwrite(path, np.full((40, 40), int(value), np.uint8))


def lbph_cases(root):
    """{case: (records, folds)} with sample files written under `root`."""
    cases = {}

    def person(case, user_id, values):
        folder = os.path.join(root, case, f"{user_id}_p")
        os.makedirs(folder, exist_ok=True)
        out = []
        for i, v in enumerate(values):
            path = os.path.join(folder, f"{i:02d}.png")
            if v is None:
                with open(path, "wb") as fh:
                    fh.write(b"not an image")
            else:
                write_square(path, v)
            out.append(rec(userId=user_id, path=path))
        return out

    # Three people whose intensities overlap a little, so some predictions
    # are wrong and the sweep has false matches to count.
    cases["three_people"] = (
        person("three_people", 1, [40, 44, 48, 52, 90, 41])
        + person("three_people", 2, [80, 84, 88, 92, 60, 81])
        + person("three_people", 3, [150, 154, 158, 162, 166, 151]), 5)

    cases["one_person"] = (person("one_person", 1, [10, 20, 30]), 5)

    # One image each: a single fold, so no training split exists.
    cases["one_image_each"] = (
        person("one_image_each", 1, [10]) + person("one_image_each", 2, [200]),
        5)

    # Unreadable files are skipped on both the training and the test side.
    cases["unreadable"] = (
        person("unreadable", 1, [30, None, 34, 36])
        + person("unreadable", 2, [None, 200, 204, 208]), 3)

    # Person 2 has one image, so the fold holding it trains on person 1 only.
    cases["lopsided"] = (
        person("lopsided", 1, [30, 32, 34]) + person("lopsided", 2, [200]), 3)

    # Every image unreadable: folds exist, nothing can be scored.
    cases["all_unreadable"] = (
        person("all_unreadable", 1, [None, None])
        + person("all_unreadable", 2, [None, None]), 2)
    return cases


def compute(root):
    """Every golden output, as plain JSON-ready data."""
    out = {"summarize_user": {}, "lbph_analysis": {}}
    for case, records in summary_cases().items():
        out["summarize_user"][case] = analytics.summarize_user(
            7, f"Person {case}", records)
    for case, (records, folds) in lbph_cases(root).items():
        out["lbph_analysis"][case] = analytics.lbph_analysis(records, folds)
    # Round-trip so tuples compare as the lists JSON stores them as.
    return json.loads(json.dumps(out, sort_keys=True))


@pytest.fixture
def fake_lbph(monkeypatch):
    monkeypatch.setattr(cv2.face, "LBPHFaceRecognizer_create", FakeLBPH)


# --------------------------------------------------------------------- tests

def test_the_split_did_not_change_a_single_output(tmp_path, fake_lbph):
    with open(GOLDEN, encoding="utf-8") as fh:
        golden = json.load(fh)
    got = compute(str(tmp_path))
    assert set(got["summarize_user"]) == set(golden["summarize_user"])
    assert set(got["lbph_analysis"]) == set(golden["lbph_analysis"])
    for section in golden:
        for case in golden[section]:
            assert got[section][case] == golden[section][case], (section, case)


def test_the_golden_file_reaches_the_branches_it_claims_to(tmp_path, fake_lbph):
    """A golden file that only holds happy paths proves little about a
    refactor. These are the shapes the split had to preserve."""
    with open(GOLDEN, encoding="utf-8") as fh:
        golden = json.load(fh)
    s, lb = golden["summarize_user"], golden["lbph_analysis"]
    assert s["good"]["verdict"] == "good" and s["good"]["recommendations"] == []
    assert len(s["flagged"]["recommendations"]) >= 4
    assert "soft focus" in s["flagged"]["flags"]
    assert s["no_pose"]["yawSpread"] is None and s["no_pose"]["yawRange"] is None
    assert s["no_pose"]["age"] is None and s["no_pose"]["quality"] is None
    assert s["one_sample"]["yawSpread"] is None
    assert s["one_sample"]["yawRange"] == [7.2, 7.2] or \
        s["one_sample"]["yawRange"] == [7.3, 7.3]
    assert s["unmeasured"]["sharpness"] is None
    assert s["empty"]["samples"] == 0 and s["empty"]["recommendations"] == []
    assert lb["three_people"]["available"] is True
    assert 0 < lb["three_people"]["accuracy"] < 100
    assert lb["one_person"]["available"] is False
    assert lb["one_image_each"]["reason"] == "Not enough samples to cross-validate."
    assert lb["all_unreadable"]["available"] is False
    assert lb["unreadable"]["available"] is True
    assert lb["lopsided"]["available"] is True


if __name__ == "__main__":
    import tempfile
    cv2.face.LBPHFaceRecognizer_create = FakeLBPH
    with tempfile.TemporaryDirectory() as tmp:
        data = compute(tmp)
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with open(GOLDEN, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {GOLDEN}")
