"""The decision attendance.py actually makes, and the report it prints.

`run_attendance` was one 65-line function around `while True:` on a webcam,
so the only logic in the program that decides whether to accept a face was
the one piece no test could reach. It is three functions now -- `identify`,
`mark_present`, `annotate` -- and the loop that calls them is the only part
that still needs a camera.

The direction of the comparison is the thing worth pinning. LBPH's
"confidence" is a *distance*: lower is a closer match. Reading it as a
confidence and accepting `> threshold` would admit every stranger and reject
everybody enrolled, and both halves of that would look like a tuning problem
rather than an inverted comparison.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cli import attendance                       # noqa: E402
from core import vision                           # noqa: E402

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
    return np.full((200, 200), 128, np.uint8)


# ------------------------------------------------------------- the decision

def test_a_close_match_is_accepted_and_named(isolated_db, gray):
    user_id = isolated_db.add_user("Ada Lovelace")
    recognizer = FakeRecognizer(user_id, vision.CONFIDENCE_THRESHOLD - 1)

    got_id, distance, name, accepted = attendance.identify(
        recognizer, gray, BOX)

    assert accepted is True
    assert got_id == user_id
    assert name == "Ada Lovelace"
    assert distance == vision.CONFIDENCE_THRESHOLD - 1


def test_a_distant_match_is_refused_and_unnamed(isolated_db, gray):
    """The name is withheld, not just flagged. A refused match that still
    carries somebody's name onto the frame is the thing to avoid."""
    user_id = isolated_db.add_user("Ada Lovelace")
    recognizer = FakeRecognizer(user_id, vision.CONFIDENCE_THRESHOLD + 1)

    _id, _distance, name, accepted = attendance.identify(
        recognizer, gray, BOX)

    assert accepted is False
    assert name == "Unknown"


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
    assert attendance.identify(at_cut, gray, BOX)[3] is False

    below = FakeRecognizer(user_id, vision.CONFIDENCE_THRESHOLD - 0.001)
    assert attendance.identify(below, gray, BOX)[3] is True


def test_an_accepted_id_with_no_database_row_says_so_rather_than_crashing(
        isolated_db, gray):
    """LBPH predicts a *label*, and the label is a user id that may have been
    deleted. The old code's `or f"Unknown (id {user_id})"` is what keeps that
    from putting `None` on the frame."""
    recognizer = FakeRecognizer(9999, vision.CONFIDENCE_THRESHOLD - 5)
    _id, _distance, name, accepted = attendance.identify(
        recognizer, gray, BOX)
    assert accepted is True
    assert "9999" in name


def test_the_face_is_resized_to_the_shared_lbph_geometry(isolated_db, gray):
    """The gallery was trained at this size, so the query has to match it --
    and the size comes from vision.py rather than from a literal here."""
    user_id = isolated_db.add_user("Ada Lovelace")
    recognizer = FakeRecognizer(user_id, 10)
    attendance.identify(recognizer, gray, BOX)
    assert recognizer.shapes == [vision.LBPH_INPUT_SIZE]


# ------------------------------------------------------------------ logging

def test_an_accepted_face_is_logged_once_per_session(isolated_db, capsys):
    """Without the session set, every frame re-queries the database for
    somebody standing still and the console fills with a line per frame."""
    user_id = isolated_db.add_user("Ada Lovelace")
    marked = set()

    attendance.mark_present(user_id, "Ada Lovelace", 40.0, marked)
    first = capsys.readouterr().out
    assert "Logged attendance" in first
    assert marked == {user_id}

    attendance.mark_present(user_id, "Ada Lovelace", 41.0, marked)
    assert capsys.readouterr().out == "", "it logged the same person twice"


def test_somebody_already_marked_today_is_told_so(isolated_db, capsys):
    """A second session on the same day: the database refuses the duplicate,
    and the message distinguishes that from a fresh log."""
    user_id = isolated_db.add_user("Ada Lovelace")
    isolated_db.log_attendance(user_id, 40.0)

    attendance.mark_present(user_id, "Ada Lovelace", 42.0, set())
    printed = capsys.readouterr().out
    assert "already marked present today" in printed
    assert "Logged attendance" not in printed


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


def test_a_saved_sample_is_the_shared_lbph_geometry(isolated_db, gray,
                                                    tmp_path):
    """The gallery side of the comparison. If this and attendance.identify
    disagree about the size, recognition gets worse with nothing to point
    at -- which is why both read it from vision.py."""
    import cv2
    from cli import register_user

    path = register_user.save_sample(gray, BOX, str(tmp_path), 1)
    assert os.path.isfile(path)

    written = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert written.shape == vision.LBPH_INPUT_SIZE[::-1], (
        "a sample was written at a different size from the one the "
        "recogniser queries at")


def test_samples_are_numbered_from_one_per_user(isolated_db, gray, tmp_path):
    from cli import register_user

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
