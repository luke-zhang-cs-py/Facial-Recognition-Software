"""The decision attendance.py actually makes, and the report it prints.

`cli.attendance` used to have `identify` and `mark_present`, which were this
program's own copy of the decision. Since decision.decide() became the one
version both front ends call they were dead code, and these tests were
testing them rather than the path a camera frame takes. They now go through
`decision.decide` (the decision), `attendance.report` (what is said about it)
and `attendance.run_attendance` (the loop, driven by a fake camera).

The direction of the comparison is the thing worth pinning. LBPH's
"confidence" is a *distance*: lower is a closer match. Reading it as a
confidence and accepting `> threshold` would admit every stranger and reject
everybody enrolled, and both halves of that would look like a tuning problem
rather than an inverted comparison.
"""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cli import attendance                       # noqa: E402
from core import vision                           # noqa: E402
from pipeline import decision, liveness           # noqa: E402

BOX = (10, 20, 60, 60)


class FakeRecognizer:
    """Returns a fixed (user_id, distance), and records what it was given."""

    def __init__(self, user_id, distance):
        self.user_id = user_id
        self.distance = distance
        self.shapes = []

    def predict(self, face):
        self.shapes.append(face.shape)
        return self.user_id, self.distance


@pytest.fixture
def gray():
    """A frame, named for what the old tests passed; decide() takes BGR."""
    return np.full((200, 200, 3), 128, np.uint8)


@pytest.fixture(autouse=True)
def live(monkeypatch):
    """Every frame scores live, so these tests are about who, not whether."""
    monkeypatch.setattr(liveness, "score", lambda bgr, box: 0.9)


RealVote = liveness.LivenessVote


def one_frame_vote():
    return RealVote(window=1, required=1)


def decide(frame, recognizer, first_mark=None):
    return decision.decide(frame, BOX, None, one_frame_vote(),
                           recognizer=recognizer, first_mark=first_mark)


def run_cli(monkeypatch, recognizer, frames):
    """Run the real capture loop over `frames` with a fake camera.

    Returns the (label, colour) pairs it drew. The detector, the camera and
    the window are stand-ins; the decision, the session rule and the report
    are the real ones.
    """
    labels = []
    queue = list(frames)

    class Cap:
        def read(self):
            return (True, queue.pop(0)) if queue else (False, None)

        def release(self):
            pass
    monkeypatch.setattr(decision, "faces", lambda bgr: [(BOX, None)])
    monkeypatch.setattr(decision, "sface_gallery", lambda: None)
    monkeypatch.setattr(attendance, "_load_recognizer", lambda: recognizer)
    monkeypatch.setattr(attendance, "_open_camera", Cap)
    monkeypatch.setattr(liveness, "LivenessVote", one_frame_vote)
    monkeypatch.setattr(attendance, "annotate",
                        lambda frm, box, label, colour: labels.append((label, colour)))
    for name, fn in (("imshow", lambda *a: None), ("waitKey", lambda *a: -1),
                     ("destroyAllWindows", lambda: None)):
        monkeypatch.setattr(cv2, name, fn)
    attendance.run_attendance()
    return labels


# ------------------------------------------------------------- the decision

def test_a_close_match_is_accepted_and_named(isolated_db, gray):
    user_id = isolated_db.add_user("Ada Lovelace")
    recognizer = FakeRecognizer(user_id, vision.CONFIDENCE_THRESHOLD - 1)

    got = decide(gray, recognizer)

    assert got.accepted is True
    assert got.user_id == user_id
    assert decision.name_of(got.user_id) == "Ada Lovelace"
    assert got.confidence == vision.CONFIDENCE_THRESHOLD - 1
    assert got.outcome == decision.LOGGED


def test_a_distant_match_is_refused_and_unnamed(isolated_db, gray, monkeypatch):
    """The name is withheld, not just flagged. A refused match that still
    carries somebody's name onto the frame is the thing to avoid -- so this
    looks at what the capture loop draws, not only at the decision."""
    user_id = isolated_db.add_user("Ada Lovelace")
    recognizer = FakeRecognizer(user_id, vision.CONFIDENCE_THRESHOLD + 1)

    got = decide(gray, recognizer)
    assert got.accepted is False
    assert got.outcome == decision.UNRECOGNISED
    assert isolated_db.get_attendance_for_today() == [], "a refused face was logged"

    labels = run_cli(monkeypatch, recognizer, [gray])
    assert labels and all(label.startswith("Unknown") for label, _ in labels)
    assert all("Ada" not in label for label, _ in labels)
    assert all(colour == attendance.UNKNOWN_COLOUR for _, colour in labels)


def test_the_threshold_is_a_distance_and_the_comparison_is_strict(isolated_db,
                                                                  gray):
    """Exactly at the threshold is a refusal: the rule is `< threshold`.

    Pinned because inverting it is silent. Both directions produce a working
    program that is wrong in opposite ways, and the off-by-one at the
    boundary is the only place the two definitions disagree by a single
    value rather than by everything.
    """
    user_id = isolated_db.add_user("Ada Lovelace")

    at_cut = FakeRecognizer(user_id, vision.CONFIDENCE_THRESHOLD)
    assert decide(gray, at_cut).accepted is False

    below = FakeRecognizer(user_id, vision.CONFIDENCE_THRESHOLD - 0.001)
    assert decide(gray, below).accepted is True


def test_an_accepted_id_with_no_database_row_says_so_rather_than_crashing(
        isolated_db, gray, monkeypatch, capsys):
    """LBPH predicts a *label*, and the label is a user id that may have been
    deleted. name_of's `or f"Unknown (id {user_id})"` keeps that from putting
    `None` on the frame, and the report says why nothing was logged."""
    recognizer = FakeRecognizer(9999, vision.CONFIDENCE_THRESHOLD - 5)

    got = decide(gray, recognizer)
    assert got.accepted is True
    assert got.outcome == decision.NO_USER
    assert "9999" in decision.name_of(got.user_id)

    labels = run_cli(monkeypatch, recognizer, [gray])
    assert any("9999" in label for label, _ in labels)
    assert "has no user" in capsys.readouterr().out


def test_the_face_is_resized_to_the_shared_lbph_geometry(isolated_db, gray):
    """The gallery was trained at this size, so the query has to match it --
    and the size comes from vision.py rather than from a literal here."""
    user_id = isolated_db.add_user("Ada Lovelace")
    recognizer = FakeRecognizer(user_id, 10)
    decide(gray, recognizer)
    assert recognizer.shapes == [vision.LBPH_INPUT_SIZE]


# ------------------------------------------------------------------ logging

def test_an_accepted_face_is_logged_once_per_session(isolated_db, gray,
                                                     monkeypatch, capsys):
    """Without the session set, every frame re-queries the database for
    somebody standing still and the console fills with a line per frame."""
    user_id = isolated_db.add_user("Ada Lovelace")
    recognizer = FakeRecognizer(user_id, 40.0)

    run_cli(monkeypatch, recognizer, [gray, gray, gray])
    printed = capsys.readouterr().out

    assert printed.count("Logged attendance") == 1, printed
    assert "already marked" not in printed, "the second frame hit the database"
    assert len(isolated_db.get_attendance_for_today()) == 1


def test_a_second_frame_in_one_session_is_seen_not_logged(isolated_db, gray):
    user_id = isolated_db.add_user("Ada Lovelace")
    marked = set()

    def claim(uid):
        if uid in marked:
            return False
        marked.add(uid)
        return True
    first = decide(gray, FakeRecognizer(user_id, 40.0), claim)
    second = decide(gray, FakeRecognizer(user_id, 41.0), claim)

    assert first.outcome == decision.LOGGED and marked == {user_id}
    assert second.outcome == decision.SEEN


def test_somebody_already_marked_today_is_told_so(isolated_db, gray,
                                                  monkeypatch, capsys):
    """A second session on the same day: the database refuses the duplicate,
    and the message distinguishes that from a fresh log."""
    user_id = isolated_db.add_user("Ada Lovelace")
    isolated_db.log_attendance(user_id, 40.0)

    run_cli(monkeypatch, FakeRecognizer(user_id, 42.0), [gray])
    printed = capsys.readouterr().out
    assert "already marked present today" in printed
    assert "Logged attendance" not in printed


def test_report_says_what_record_did(capsys):
    attendance.report(decision.LOGGED, 3, "Ada", 41.26)
    assert capsys.readouterr().out == "Logged attendance: Ada at distance 41.3\n"
    attendance.report(decision.LOGGED, 3, "Ada", 0.8123, decision.SFACE)
    assert "similarity 0.812" in capsys.readouterr().out
    attendance.report(decision.ALREADY, 3, "Ada", 41.0)
    assert "Ada already marked present today." in capsys.readouterr().out
    attendance.report(decision.NO_USER, 3, "Ada", 41.0)
    assert "Recognised label 3, which has no user" in capsys.readouterr().out


def test_todays_report_lists_what_was_logged(isolated_db, capsys):
    user_id = isolated_db.add_user("Ada Lovelace")
    isolated_db.log_attendance(user_id, 37.5)

    attendance._report_today()
    printed = capsys.readouterr().out
    assert "Ada Lovelace" in printed
    assert "37.5" in printed


# ----------------------------------------------------------------- drawing

def test_annotating_a_frame_marks_the_box_without_changing_its_shape():
    frame = np.zeros((120, 160, 3), np.uint8)
    attendance.annotate(frame, BOX, "Ada (40)", attendance.MATCH_COLOUR)

    assert frame.shape == (120, 160, 3)
    assert frame.any(), "nothing was drawn"


def test_the_two_verdict_colours_are_distinct():
    """Green for accepted, red for unknown -- and BGR, not RGB, which is the
    reason a "red" box written as (255, 0, 0) comes out blue in OpenCV."""
    assert attendance.MATCH_COLOUR != attendance.UNKNOWN_COLOUR
    assert attendance.MATCH_COLOUR[1] == 255, "green should be the G channel"
    assert attendance.UNKNOWN_COLOUR[2] == 255, "red should be the R channel"


# --------------------------------------------------------------- setup paths

def test_no_trained_model_says_what_to_run_instead_of_failing(isolated_root,
                                                              capsys):
    """The state of every fresh clone: trainer.yml is gitignored.

    The advice has to name two modules, and -- since they now live in
    packages and are run with `-m` -- it has to name them the way they are
    actually invoked. Asserting the bare filenames was enough while the
    modules sat in the root; it would now pass on a message telling somebody
    to run `python register_user.py`, which fails with "no such file".

    So each named module is also imported. A message that names something
    unimportable is the failure this is really about, and a string check
    alone cannot see it.
    """
    import importlib

    assert attendance._load_recognizer() is None
    printed = capsys.readouterr().out

    wanted = ("cli.register_user", "pipeline.train_model")
    for module in wanted:
        assert module in printed, (
            "the advice does not say to run %s:\n%s" % (module, printed))
        importlib.import_module(module)      # so the advice is runnable


def test_run_attendance_gives_up_cleanly_with_no_model(isolated_root, capsys):
    """The whole entry point, with no camera involved -- it returns before
    ever asking for one."""
    assert attendance.run_attendance() is None
    assert "No trained model found" in capsys.readouterr().out


# ===================================================== the capture side
# register_user.py was at 0% -- not one statement of it had ever executed
# under test, because all of it sat inside a webcam loop. The parts that
# write the gallery are extracted now, and they are the half that matters:
# these are the images attendance.py will later be compared against.

def test_creating_a_user_makes_both_the_row_and_the_folder(isolated_db,
                                                           capsys):
    from cli import register_user

    user_id, folder = register_user.create_user("Grace Hopper")
    assert isolated_db.get_user_name(user_id) == "Grace Hopper"
    assert os.path.isdir(folder), "the dataset folder was not created"
    assert str(user_id) in os.path.basename(folder), (
        "the folder name has to carry the id; the rest of the project "
        "parses it back out")
    assert "Grace Hopper" in capsys.readouterr().out


def test_a_saved_sample_is_the_shared_lbph_geometry(isolated_db, tmp_path):
    """The gallery side of the comparison. If this and the decision's crop
    disagree about the size, recognition gets worse with nothing to point
    at -- which is why both read it from vision.py."""
    from cli import register_user

    gray = np.full((200, 200), 128, np.uint8)
    path = register_user.save_sample(gray, BOX, str(tmp_path), 1)
    assert os.path.isfile(path)

    written = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert written.shape == vision.LBPH_INPUT_SIZE[::-1], (
        "a sample was written at a different size from the one the "
        "recogniser queries at")


def test_samples_are_numbered_from_one_per_user(isolated_db, tmp_path):
    from cli import register_user

    gray = np.full((200, 200), 128, np.uint8)

    names = [os.path.basename(register_user.save_sample(
        gray, BOX, str(tmp_path), i)) for i in (1, 2, 3)]
    assert names == ["1.jpg", "2.jpg", "3.jpg"]


def test_capturing_nothing_says_the_registration_is_incomplete(capsys):
    """The row and the folder exist, so this "succeeded" -- and training on
    it would train on nothing. It gets its own message for that reason."""
    from cli import register_user

    register_user.report(0, "Grace Hopper", 1)
    printed = capsys.readouterr().out
    assert "incomplete" in printed
    assert "train_model" not in printed, (
        "it told the user to train on an empty folder")


def test_capturing_samples_points_at_the_next_step(capsys):
    from cli import register_user

    register_user.report(30, "Grace Hopper", 7)
    printed = capsys.readouterr().out
    assert "30 samples" in printed and "Grace Hopper" in printed
    assert "train_model" in printed


def test_the_preview_overlay_shows_progress_out_of_the_target(capsys):
    from cli import register_user

    frame = np.zeros((120, 160, 3), np.uint8)
    register_user.annotate(frame, BOX, 5)
    assert frame.any(), "nothing was drawn on the preview"


def test_the_cli_requires_exactly_one_name(capsys):
    """Quoting a two-word name is the obvious thing to get wrong, and the
    message has to show the quotes."""
    from cli import register_user

    assert register_user.main([]) == 1
    assert "Full Name" in capsys.readouterr().out

    assert register_user.main(["Grace", "Hopper"]) == 1
    assert "Full Name" in capsys.readouterr().out


# ============================================ the parts around the capture loop
# `run_cli` above replaces _open_camera and _load_recognizer, so neither was
# ever run. A recorder stands in for cv2.VideoCapture and for the LBPH
# recogniser; what is checked is what they are asked and what the user hears.


class FakeDevice:
    def __init__(self, opened):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True


def test_an_open_webcam_is_handed_back(monkeypatch):
    device = FakeDevice(True)
    asked = []
    monkeypatch.setattr(cv2, "VideoCapture",
                        lambda *args: asked.append(args) or device)
    assert attendance._open_camera() is device
    assert asked == [(0,)]
    assert not device.released


def test_a_webcam_that_will_not_open_is_released_and_reported(monkeypatch,
                                                              capsys):
    device = FakeDevice(False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: device)
    assert attendance._open_camera() is None
    assert device.released, "a device that would not open was left held"
    assert "Could not open webcam" in capsys.readouterr().out


def test_the_trained_model_is_read_from_where_paths_says(isolated_root,
                                                         monkeypatch):
    from core import paths

    class Recognizer:
        read_from = None

        def read(self, path):
            Recognizer.read_from = path

    open(paths.model_path(), "wb").close()    # present; the fake never parses it
    monkeypatch.setattr(cv2.face, "LBPHFaceRecognizer_create", Recognizer)
    got = attendance._load_recognizer()
    assert isinstance(got, Recognizer)
    assert Recognizer.read_from == paths.model_path()


def test_with_a_gallery_but_no_webcam_it_says_who_is_missing_and_stops(
        isolated_db, monkeypatch, capsys):
    gallery = {1: [np.zeros(128, np.float32)]}
    monkeypatch.setattr(decision, "sface_gallery", lambda: gallery)
    monkeypatch.setattr(decision, "unenrolled", lambda gal: ["Grace", "Alan"])
    monkeypatch.setattr(liveness, "available", lambda: False)
    monkeypatch.setattr(attendance, "_open_camera", lambda: None)
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda *a: shown.append(a))

    assert attendance.run_attendance() is None

    out = capsys.readouterr().out
    assert "Recognising with SFace: 1 people in the gallery." in out
    assert "never recognised: Grace, Alan" in out
    assert "liveness model is missing" in out
    assert shown == [], "it went on to the capture loop with no camera"
    assert "Attendance system running" not in out


def test_pressing_q_ends_the_session_and_reports_today(isolated_db, gray,
                                                        monkeypatch, capsys):
    """The loop reads frames until 'q'. A face with no confidence (nothing to
    compare it against) is labelled with a dash, not a number."""
    reads = []

    class Cap:
        def read(self):
            reads.append(1)
            return True, gray.copy()

        def release(self):
            self.released = True

    cap = Cap()
    labels = []
    monkeypatch.setattr(decision, "faces", lambda bgr: [(BOX, None)])
    monkeypatch.setattr(decision, "sface_gallery", lambda: None)
    monkeypatch.setattr(attendance, "_load_recognizer",
                        lambda: FakeRecognizer(1, 0))
    monkeypatch.setattr(attendance, "_open_camera", lambda: cap)
    monkeypatch.setattr(decision, "decide",
                        lambda *a, **k: decision.Decision(
                            decision.UNRECOGNISED, None, None, False, None,
                            None, "unknown"))
    monkeypatch.setattr(attendance, "annotate",
                        lambda frm, box, label, colour: labels.append(label))
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda ms: ord("q"))

    attendance.run_attendance()

    assert len(reads) == 1, "it kept reading after q"
    assert labels == ["Unknown (-)"]
    assert cap.released
    assert "Today's attendance" in capsys.readouterr().out
