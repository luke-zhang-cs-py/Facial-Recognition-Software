"""The rules the first mutmut run showed nothing was checking.

.github/workflows/mutation.yml mutates the four modules setup.cfg names --
the attendance decision, the liveness vote, the database and the threshold
maths -- and runs the suite against each change. The first run killed 614
of 738 mutants. Of the 124 that survived, most were real gaps: a boundary
(`<` for `<=`), an interpolation nobody evaluated between two measured
points, a branch of add_user no test reached, an argument that was
silently dropped. Each test here pins the behaviour one or more of those
mutants changed. The survivors that cannot change behaviour (SQL keyword
case, OpenCV's own defaults, ...) are listed with reasons in
notes/CODE_AUDIT.md, fourth pass, instead of being pinned here.

No faces, no weights, no network: the detector, the recogniser and the
liveness net are stand-ins, and every database is a temporary file.
"""
import importlib
import math
import os
import re

import cv2
import numpy as np
import pytest

from analysis import calibration
from core import db, paths
from pipeline import decision, liveness, recognition, traits

W, H = 640, 480
BOX = (200, 150, 200, 200)


def noise_frame():
    rng = np.random.RandomState(7)
    return cv2.GaussianBlur(rng.randint(0, 255, (H, W, 3)).astype(np.uint8), (3, 3), 0)


def yunet_row(x, y, w, h):
    cx = x + w / 2
    return np.array([x, y, w, h, cx - 50, y + 60, cx + 50, y + 60, cx, y + 110,
                     cx - 35, y + 160, cx + 35, y + 160, 0.99], np.float32)


class Recogniser:
    """An LBPH stand-in: always this id at this distance."""

    def __init__(self, user_id, distance=10.0):
        self.user_id, self.distance = user_id, distance

    def predict(self, face):
        return self.user_id, self.distance


def one_frame_vote():
    return liveness.LivenessVote(window=1, required=1)


@pytest.fixture
def scored(monkeypatch):
    monkeypatch.setattr(liveness, "score", lambda bgr, box: 0.9)


# ================================================== analysis/calibration.py

def test_the_false_match_rate_is_interpolated_between_measured_thresholds():
    """Every earlier test read fmr_at at a measured threshold or past either
    end, where no interpolation happens -- so `*` for `/` in the fraction,
    `x1 + x0` for `x1 - x0`, and an end clamp moved one entry inward all
    passed. The first, a middle and the last interval are each checked."""
    pts = calibration.SFACE_FMR
    for (x0, y0), (x1, y1) in [(pts[0], pts[1]), (pts[8], pts[9]), (pts[-2], pts[-1])]:
        for share in (0.25, 0.5):
            t = x0 + share * (x1 - x0)
            assert calibration.fmr_at(t) == pytest.approx(y0 + share * (y1 - y0), rel=1e-9), t


def test_a_gallery_of_two_has_one_other_person_to_be_mistaken_for():
    """`< 2` is the rule: one person alone cannot be confused with anybody,
    two can. `<= 2` reported zero risk for every pair."""
    f = calibration.fmr_at(0.4)
    assert calibration.gallery_risk(0.4, 2) == pytest.approx(f, rel=1e-12)
    assert calibration.gallery_risk(0.4, 1) == 0.0
    assert calibration.gallery_risk(0.4, 0) == 0.0


def test_gallery_risk_counts_everybody_else_and_nobody_more():
    """N - 1 others: P = 1 - (1 - FMR) ** (N - 1). An exponent of N + 1 or
    N - 2 is off by a percent or so at these sizes, which no rounded figure
    on the page would show."""
    f = calibration.fmr_at(0.5)
    for n in (3, 1000, 10_000):
        assert calibration.gallery_risk(0.5, n) == pytest.approx(1 - (1 - f) ** (n - 1), rel=1e-12)


def test_describe_uses_the_risk_target_it_is_given():
    """describe() printed the target it was given but recommended the
    threshold for the default 1% whatever it was asked."""
    loose, _, ok = calibration.recommend_threshold(1000, 0.5)
    strict, _, _ = calibration.recommend_threshold(1000, 0.01)
    assert ok and loose != strict
    text = calibration.describe(1000, max_risk=0.5)
    assert f"threshold {loose} " in text and "target 50%" in text


def test_age_coverage_interpolates_next_to_both_ends_of_the_table():
    cov = calibration.AGE_COVERAGE
    assert calibration.age_band_coverage(1.5) == pytest.approx((cov[1] + cov[2]) / 2)
    assert calibration.age_band_coverage(2) == cov[2]
    assert calibration.age_band_coverage(15) == cov[15]
    assert calibration.age_band_coverage(17.5) == pytest.approx((cov[15] + cov[20]) / 2)


def test_sample_accuracy_at_the_second_measured_count():
    acc = calibration.SAMPLE_ACCURACY
    assert calibration.accuracy_for_samples(2) == acc[2]
    assert calibration.accuracy_for_samples(1.5) == pytest.approx((acc[1] + acc[2]) / 2)


def test_sample_accuracy_reads_the_top_of_the_table_from_the_top_entry(monkeypatch):
    """The measured table is flat from 16 samples up (that is the saturation
    it documents), so reading the top from the second-last entry, or
    clamping from the second-last key, gives the same numbers on it. A table
    still rising at the top tells the right key from its neighbour; the
    function is meant to interpolate whatever table is measured next."""
    monkeypatch.setattr(calibration, "SAMPLE_ACCURACY", {1: 0.1, 10: 0.5, 20: 0.7, 30: 0.9})
    assert calibration.accuracy_for_samples(30) == 0.9
    assert calibration.accuracy_for_samples(45) == 0.9
    assert calibration.accuracy_for_samples(25) == pytest.approx(0.8)
    assert calibration.accuracy_for_samples(20) == 0.7


# =============================================================== core/db.py

def test_a_connection_waits_ten_seconds_for_a_lock_not_sqlites_five(isolated_db):
    """The camera thread and the web requests write to one file. sqlite3's
    default busy timeout is 5 s; the 10 s the connection asks for is what a
    write blocked by the other one waits before "database is locked"."""
    conn = db.get_connection()
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10_000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_add_user_raises_an_existing_sequence_past_every_claimed_folder(isolated_db):
    """add_user's UPDATE branch: users already exist (so sqlite_sequence has
    a row) and a folder claims an id above it. No test reached this branch,
    so a broken UPDATE, an UPDATE that matched no row, and an INSERT of a
    second sequence row all passed. The last two hand out id 2 (SQLite reads
    the first sequence row), which the folder 2_* already claims: the old
    folder's face filed under the new person's name, the exact defect
    add_user exists to prevent."""
    assert db.add_user("Ada") == 1
    for name in ("2_Old_Entry", "7_Other_Entry"):
        os.makedirs(os.path.join(paths.dataset_dir(), name))

    ben = db.add_user("Ben")
    assert ben not in paths.folder_ids()
    assert ben == 8
    assert db.add_user("Cy") == 9


def test_a_mark_with_no_method_is_recorded_as_lbph(isolated_db):
    """The default says how to read the confidence: an LBPH distance."""
    ada = db.add_user("Ada")
    assert db.log_attendance(ada, 40.0) is True
    assert [row[3] for row in db.get_all_attendance_with_method()] == [decision.LBPH]


# ====================================================== pipeline/decision.py

def test_faces_and_boxes_run_the_detector_on_the_frame_they_are_given(monkeypatch):
    frame = noise_frame()
    row = yunet_row(100, 80, 120, 140)
    seen = []
    monkeypatch.setattr(traits, "detect", lambda bgr: seen.append(bgr) or [row])

    assert decision.boxes(frame) == [(100, 80, 120, 140)]
    [(box, got_row)] = decision.faces(frame)
    assert box == (100, 80, 120, 140) and got_row is row
    assert len(seen) == 2 and all(s is frame for s in seen)


def test_the_primary_face_is_the_largest_by_area():
    """By w * h. Not the first tuple in sort order (which is the rightmost
    box), and not the tallest."""
    wide = (0, 0, 200, 50)        # 10,000 px, short
    tall = (300, 0, 60, 80)       # 4,800 px, taller, further right
    assert decision.primary([tall, wide]) == wide
    assert decision.primary([wide, tall]) == wide
    assert decision.primary([]) is None


def test_crop_keeps_the_top_row_of_a_box_that_starts_above_the_frame():
    gray = np.zeros((100, 100), np.uint8)
    gray[0, :] = 255
    face = decision.crop(gray, (0, -10, 100, 11))     # all that is in frame is row 0
    assert face is not None and face.min() == 255


def test_crop_clips_to_the_height_of_a_portrait_frame():
    gray = np.zeros((200, 100), np.uint8)
    gray[120:170, 0:50] = 200
    face = decision.crop(gray, (0, 120, 50, 50))
    assert face is not None and face.min() == 200


@pytest.mark.parametrize("box", [
    (100, 10, 20, 20),     # starts exactly at the right edge: no columns
    (150, 10, 20, 20),     # entirely right of the frame
    (10, 100, 20, 20),     # starts exactly at the bottom edge: no rows
    (10, 150, 20, 20),     # entirely below
])
def test_a_box_with_no_pixels_on_either_axis_crops_to_none(box):
    """Either axis empty is no face. `and` for `or`, or `<` for `<=`, sent an
    empty slice on to cv2.resize, which raises."""
    assert decision.crop(np.zeros((100, 100), np.uint8), box) is None


def test_sface_gallery_embeds_new_samples_unless_told_not_to(isolated_db, monkeypatch):
    refreshed = []
    monkeypatch.setattr(decision.facemodels, "have", lambda name: True)
    monkeypatch.setattr(recognition, "refresh_gallery", lambda: refreshed.append(1) or 0)
    monkeypatch.setattr(recognition, "gallery", lambda: {1: "embedding"})

    assert decision.sface_gallery() == {1: "embedding"}
    assert refreshed == [1], "the default must embed samples not embedded yet"
    assert decision.sface_gallery(refresh=False) == {1: "embedding"}
    assert refreshed == [1]

    monkeypatch.setattr(decision.facemodels, "have", lambda name: False)
    assert decision.sface_gallery() is None


def test_the_lbph_reasons_say_what_is_missing_and_how_to_fix_it(monkeypatch):
    """Shown to whoever is running attendance. Without the weights it names
    the recogniser in use, the cost, and a command that exists; with the
    weights it does not send them to download what they already have."""
    monkeypatch.setattr(decision.facemodels, "have", lambda name: False)
    msg = decision.lbph_reason()
    assert msg.startswith("SFace weights missing: recognising with LBPH")
    assert "far less accurate." in msg
    command = re.search(r"Run: python -m (\S+)$", msg)
    assert command, msg
    importlib.import_module(command.group(1))          # the advice is runnable

    monkeypatch.setattr(decision.facemodels, "have", lambda name: True)
    msg = decision.lbph_reason()
    assert msg.startswith("Nobody enrolled has an SFace embedding yet: recognising with LBPH")
    assert msg.endswith("far less accurate.")
    assert "fetch_models" not in msg


def test_unmirror_row_maps_every_coordinate_exactly():
    """Every value distinct, unlike the symmetric test face: a non-square
    box, a tilted head (eyes at different heights) and a crooked mouth, so a
    swap that drops one y, a box mirrored by its height, a landmark x off by
    one, or a y mirrored as if it were an x each show."""
    row = [100, 50, 120, 160,          # box x, y, w, h
           130, 90, 190, 96,           # right eye, left eye (as seen on the mirror)
           160, 120,                   # nose
           140, 170, 185, 176,         # right and left mouth corner
           0.9]                        # score
    got = decision.unmirror_row(row, W)
    expected = [W - 100 - 120, 50, 120, 160,
                W - 1 - 190, 96, W - 1 - 130, 90,
                W - 1 - 160, 120,
                W - 1 - 185, 176, W - 1 - 140, 170,
                0.9]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)
    assert got.dtype == np.float32, "SFace's alignCrop takes a float32 row"


def test_unmirror_row_leaves_the_detectors_row_alone():
    """It returns a new row. The one it was given belongs to the caller --
    the camera keeps it in its face record -- and still describes the
    mirrored preview."""
    row = yunet_row(*BOX)
    before = row.copy()
    decision.unmirror_row(row, W)
    np.testing.assert_array_equal(row, before)


def test_decide_with_nobody_in_frame_accepts_nobody_and_has_no_verdict(isolated_db, scored):
    got = decision.decide(noise_frame(), None, None, one_frame_vote(), recognizer=Recogniser(1))
    assert got == decision.Decision(decision.NO_FACE, None, None, False, None, None, "unknown")
    assert got.accepted is False


def test_decide_on_a_box_with_no_pixels_accepts_nobody(isolated_db, scored):
    got = decision.decide(np.zeros((40, 40, 3), np.uint8), BOX, None, one_frame_vote(),
                          recognizer=Recogniser(1))
    assert got == decision.Decision(decision.UNREADABLE, None, None, False, None, None, None)
    assert got.accepted is False


def test_decide_scores_liveness_on_the_face_and_reports_the_score(isolated_db, monkeypatch):
    ada = db.add_user("Ada")
    boxes = []
    monkeypatch.setattr(liveness, "score", lambda bgr, box: boxes.append(box) or 0.9)
    got = decision.decide(noise_frame(), BOX, None, one_frame_vote(), recognizer=Recogniser(ada))
    assert boxes == [BOX]
    assert got.live_score == 0.9 and got.outcome == decision.LOGGED


def test_decide_embeds_an_unmirrored_frame_as_it_is(isolated_db, scored, monkeypatch):
    ada = db.add_user("Ada")
    seen = []
    monkeypatch.setattr(decision, "identify",
                        lambda bgr, row, gal: seen.append((bgr, row)) or (ada, 0.9, True))
    frame, row = noise_frame(), yunet_row(*BOX)
    got = decision.decide(frame, BOX, row, one_frame_vote(), gallery={ada: None})
    assert (got.outcome, got.method, got.confidence) == (decision.LOGGED, decision.SFACE, 0.9)
    assert seen[0][0] is frame and seen[0][1] is row


def test_an_unrecognised_frame_keeps_the_vote_whoever_it_resembled(isolated_db, scored):
    """A refused match is nobody, even when its nearest label is somebody
    else. Following that label restarted the vote, so one poor frame threw
    away the evidence of the person actually standing there."""
    ada, ben = db.add_user("Ada"), db.add_user("Ben")
    vote = liveness.LivenessVote()
    for _ in range(liveness.VOTE_REQUIRED - 1):
        decision.decide(noise_frame(), BOX, None, vote, recognizer=Recogniser(ada))
    assert vote.samples == liveness.VOTE_REQUIRED - 1

    far = Recogniser(ben, distance=decision.CONFIDENCE_THRESHOLD + 5)
    got = decision.decide(noise_frame(), BOX, None, vote, recognizer=far)
    assert got.outcome == decision.UNRECOGNISED
    assert vote.samples == liveness.VOTE_REQUIRED, "the vote was started over"
    assert vote.verdict() == "live"


# ====================================================== pipeline/liveness.py

class FakeNet:
    def __init__(self, logits=(0.0, 2.0, 0.0)):
        self.logits = np.asarray(logits, np.float32)
        self.blobs = []

    def setInput(self, blob):
        self.blobs.append(blob)

    def forward(self):
        return self.logits


def test_the_net_gets_an_unscaled_unshifted_bgr_blob(monkeypatch):
    """MiniFASNet was trained on raw 0-255 BGR: no mean subtracted, no
    channel swap. CI has no weights, so score() past the crop never ran with
    a net there, and nothing looked at the blob."""
    net = FakeNet()
    monkeypatch.setattr(liveness, "_get", lambda: net)
    frame = np.empty((H, W, 3), np.uint8)
    frame[...] = (10, 100, 200)                       # B, G, R
    p = liveness.score(frame, (270, 190, 100, 100))

    [blob] = net.blobs
    assert blob.shape == (1, 3) + liveness.INPUT
    assert (blob[0, 0] == 10).all() and (blob[0, 1] == 100).all() and (blob[0, 2] == 200).all()
    assert p == pytest.approx(math.exp(2) / (2 + math.exp(2)))


def test_the_liveness_crop_is_clipped_by_the_frame_height(monkeypatch):
    """A portrait frame (taller than wide) with the face near the bottom.
    Clipping the bottom edge by the width put it above the top edge, and
    the face got no score at all."""
    net = FakeNet()
    monkeypatch.setattr(liveness, "_get", lambda: net)
    frame = np.zeros((400, 200, 3), np.uint8)
    frame[300:, :] = 255
    box = (75, 300, 50, 50)                          # centre (100, 325), half 67.5

    assert liveness.score(frame, box) is not None
    expected = cv2.dnn.blobFromImage(frame[257:392, 32:167], 1.0, liveness.INPUT, (0, 0, 0),
                                     swapRB=False)
    np.testing.assert_array_equal(net.blobs[0], expected)
