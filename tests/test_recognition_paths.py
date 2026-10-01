"""The four ways a face gets into or out of the LBPH model have to agree.

Galleries are written by the web camera, `cli.register_user` and
`cli.seed_demo`; they are read by the web camera and `cli.attendance`. LBPH
compares texture histograms pixel-grid by pixel-grid, so it is not
mirror-invariant and it is sensitive to where the box was drawn. Measured on
this checkout's LFW crops (notes/CODE_AUDIT_2026-10.md): a mirrored query
took held-out rank-1 from 9/16 to 4/16, and querying a YuNet gallery with
Haar crops cut correct accepts at the threshold from 42 to 28 of 92.

These tests hold each path to the same orientation and the same detector,
and the attendance decision to the same rules -- liveness, the primary face,
and a label with no user row -- whichever front end makes it.
"""
import os
import sqlite3

import cv2
import numpy as np
import pytest

from core import vision
from pipeline import camera, liveness, traits

W, H = 640, 480
BOX = (200, 150, 200, 200)          # LBPH-sized, so resizing is the identity


def yunet_row(x=200.0, y=150.0, w=200.0, h=200.0):
    """A frontal YuNet row: eyes level, nose centred (yaw 0, roll 0)."""
    cx = x + w / 2
    return np.array([x, y, w, h,
                     cx - 50, y + 60, cx + 50, y + 60,
                     cx, y + 110,
                     cx - 35, y + 160, cx + 35, y + 160,
                     0.99], np.float32)


def raw_frame():
    """What a webcam hands over: textured, and not left-right symmetric."""
    rng = np.random.RandomState(3)
    frame = rng.randint(0, 255, (H, W, 3)).astype(np.uint8)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame[:, : W // 3] //= 3            # a dark left third, so a flip shows
    return frame


def true_crop(frame, box):
    """The face as the camera saw it: the CLI's and seed_demo's orientation."""
    x, y, w, h = box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray[y:y + h, x:x + w], vision.LBPH_INPUT_SIZE)


def mirrored(box):
    """Where `box` lands on the mirrored preview the camera module draws on."""
    x, y, w, h = box
    return (W - x - w, y, w, h)


class FakeCapture:
    def __init__(self, frames, on_empty=None):
        self.frames = list(frames)
        self.on_empty = on_empty

    def isOpened(self):
        return True

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        if self.on_empty:
            self.on_empty()
        return False, None

    def release(self):
        pass


class Recorder:
    """An LBPH stand-in that remembers what it was asked to match."""

    def __init__(self, user_id, distance=10.0):
        self.user_id, self.distance, self.seen = user_id, distance, []

    def predict(self, face):
        self.seen.append(face.copy())
        return self.user_id, self.distance


@pytest.fixture
def mgr(monkeypatch):
    m = camera.CameraManager(capture_factory=lambda: None)
    m._traits_on = False
    monkeypatch.setattr(camera.facelandmarks, "fit", lambda gray, box: None)
    return m


def run_loop(mgr, frames):
    cap = FakeCapture(frames, on_empty=lambda: setattr(mgr, "_running", False))
    mgr._cap, mgr._running = cap, True
    mgr._capture_loop()


# ------------------------------------------------------------- orientation

def test_the_camera_matches_the_face_the_way_the_camera_saw_it(
        mgr, isolated_db, monkeypatch):
    """The preview is mirrored on purpose. The pixels LBPH matches must not be:
    the gallery from cli.register_user and seed_demo is the unmirrored face,
    so a mirrored query is compared against the wrong half of every face."""
    frame = raw_frame()
    uid = isolated_db.add_user("Ada")
    mgr._mode = camera.MODE_ATTENDANCE
    mgr._recognizer = Recorder(uid)
    monkeypatch.setattr(camera.facetraits, "detect",
                        lambda f: [yunet_row(*mirrored(BOX))])
    monkeypatch.setattr(camera.faceliveness, "score", lambda f, b: 0.9)

    run_loop(mgr, [frame])

    assert mgr._recognizer.seen, "nothing was matched"
    got = mgr._recognizer.seen[0]
    assert np.array_equal(got, true_crop(frame, BOX)), (
        "the camera matched a mirrored face; the CLI and seed_demo galleries "
        "are unmirrored")


def test_the_camera_stores_the_face_the_way_the_camera_saw_it(
        mgr, isolated_root, monkeypatch):
    frame = raw_frame()
    folder = os.path.join(isolated_root, "dataset", "1_Ada")
    os.makedirs(folder)
    mgr._mode = camera.MODE_REGISTER
    mgr._reg_name, mgr._reg_user_id, mgr._reg_dir = "Ada", 1, folder
    monkeypatch.setattr(camera.facetraits, "detect",
                        lambda f: [yunet_row(*mirrored(BOX))])
    monkeypatch.setattr(camera, "weak_photograph", lambda crop, face: [])

    run_loop(mgr, [frame])

    written = cv2.imread(os.path.join(folder, "1.jpg"), cv2.IMREAD_GRAYSCALE)
    assert written is not None, "no sample was written"
    expected = true_crop(frame, BOX).astype(float)
    straight = np.abs(written - expected).mean()
    flipped = np.abs(written - cv2.flip(expected, 1)).mean()
    assert straight < flipped / 3, (
        f"the stored sample is mirrored (diff {straight:.1f} unmirrored vs "
        f"{flipped:.1f} mirrored)")


# ---------------------------------------------------------------- detector

def _no_window(monkeypatch):
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)


def _faces_in_front(monkeypatch, *boxes):
    """Put these faces in front of whichever detector the code asks.

    traits.detect is the shared one. A cascade of the same boxes as well, so
    a test of the decision means the same thing against code that still runs
    its own Haar cascade: it sees the face too, and fails on what it decides
    rather than on finding nobody.
    """
    monkeypatch.setattr(traits, "detect",
                        lambda bgr: [yunet_row(*b) for b in boxes])

    class Cascade:
        def __init__(self, *a):
            pass

        def empty(self):
            return False

        def detectMultiScale(self, *a, **k):
            return np.array(boxes, dtype=np.int32)
    monkeypatch.setattr(cv2, "CascadeClassifier", Cascade)


def test_cli_registration_crops_with_the_detector_the_camera_uses(
        isolated_db, monkeypatch, capsys):
    """A Haar box and a YuNet box are different crops of the same head, and
    LBPH distances move with the crop. The camera and seed_demo detect with
    traits.detect; the CLI ran a Haar cascade of its own."""
    from cli import register_user
    frame = raw_frame()
    _no_window(monkeypatch)
    monkeypatch.setattr(register_user, "SAMPLES_TO_CAPTURE", 1)
    monkeypatch.setattr(register_user, "_open_camera",
                        lambda: FakeCapture([frame]))
    monkeypatch.setattr(traits, "detect", lambda bgr: [yunet_row(*BOX)])

    register_user.register_user("Ada")

    folder = os.path.join(isolated_db.paths.dataset_dir(), "1_Ada")
    written = cv2.imread(os.path.join(folder, "1.jpg"), cv2.IMREAD_GRAYSCALE)
    assert written is not None, "the CLI found no face where traits.detect did"
    assert np.abs(written - true_crop(frame, BOX).astype(float)).mean() < 3


def test_a_box_off_the_edge_of_the_frame_is_clipped_not_wrapped(isolated_db):
    """YuNet can report a box starting left of the frame. A negative slice
    start counts from the far edge, so the crop came from the other side."""
    from cli import attendance
    gray = cv2.cvtColor(raw_frame(), cv2.COLOR_BGR2GRAY)
    rec = Recorder(isolated_db.add_user("Ada"))
    attendance.identify(rec, gray, (-20, 100, 120, 120))
    want = cv2.resize(gray[100:220, 0:100], vision.LBPH_INPUT_SIZE)
    assert np.array_equal(rec.seen[0], want)


# ------------------------------------------------------- the CLI decision

@pytest.fixture
def cli_session(isolated_db, monkeypatch):
    """cli.attendance's loop over a fake webcam, one face in every frame."""
    from cli import attendance
    _no_window(monkeypatch)
    uid = isolated_db.add_user("Ada")
    recognizer = Recorder(uid)
    monkeypatch.setattr(attendance, "_load_recognizer", lambda: recognizer)
    _faces_in_front(monkeypatch, BOX)
    monkeypatch.setattr(liveness, "available", lambda: True)

    def run(frames, live_score):
        monkeypatch.setattr(liveness, "score", lambda bgr, box: live_score)
        monkeypatch.setattr(attendance, "_open_camera",
                            lambda: FakeCapture([raw_frame() for _ in range(frames)]))
        attendance.run_attendance()
        return isolated_db.get_attendance_for_today()
    return run


def test_the_cli_refuses_a_photograph_like_the_camera_does(cli_session):
    """The camera refuses a presentation attack; `python -m cli.attendance`
    marked the same photograph present, on the same model and threshold."""
    assert cli_session(liveness.VOTE_WINDOW, live_score=0.01) == []


def test_the_cli_still_marks_a_live_face(cli_session):
    logged = cli_session(liveness.VOTE_WINDOW, live_score=0.9)
    assert [name for name, _, _ in logged] == ["Ada"]


def test_the_cli_marks_nobody_without_the_liveness_model(cli_session,
                                                         monkeypatch, capsys):
    monkeypatch.setattr(liveness, "available", lambda: False)
    assert cli_session(liveness.VOTE_WINDOW, live_score=None) == []
    assert "fetch_models" in capsys.readouterr().out


def test_the_cli_decides_on_the_largest_face_only(isolated_db, monkeypatch):
    """One shared liveness vote cannot vouch for two people at once; the
    camera recognises the largest face and so must the CLI."""
    from cli import attendance
    _no_window(monkeypatch)
    recognizer = Recorder(isolated_db.add_user("Ada"))
    monkeypatch.setattr(attendance, "_load_recognizer", lambda: recognizer)
    _faces_in_front(monkeypatch, (20, 20, 90, 90), BOX)
    monkeypatch.setattr(liveness, "available", lambda: True)
    monkeypatch.setattr(liveness, "score", lambda bgr, box: 0.9)
    frame = raw_frame()
    expected = true_crop(frame, BOX)    # before the overlay is drawn on it
    monkeypatch.setattr(attendance, "_open_camera", lambda: FakeCapture([frame]))
    attendance.run_attendance()
    assert len(recognizer.seen) == 1
    assert np.array_equal(recognizer.seen[0], expected)


# ------------------------------------------------- a label with no user row

def test_the_cli_does_not_crash_on_a_label_with_no_user(isolated_db, capsys):
    """train_model trains every folder in dataset/, including ones whose user
    row is gone (this checkout has 322 folders and 9 users). LBPH can then
    predict a label the foreign key refuses, and the CLI died on it."""
    from cli import attendance
    attendance.mark_present(9999, "Unknown (id 9999)", 10.0, set())
    assert isolated_db.get_attendance_for_today() == []
    assert "9999" in capsys.readouterr().out


def test_the_camera_does_not_fail_on_a_label_with_no_user(mgr, isolated_db,
                                                          monkeypatch):
    mgr._mode = camera.MODE_ATTENDANCE
    mgr._recognizer = Recorder(9999)
    monkeypatch.setattr(camera.facetraits, "detect", lambda f: [yunet_row(*BOX)])
    monkeypatch.setattr(camera.faceliveness, "score", lambda f, b: 0.9)
    monkeypatch.setattr(mgr._liveness, "verdict", lambda: "live")
    try:
        mgr._handle_attendance(raw_frame())
    except sqlite3.IntegrityError as exc:       # pragma: no cover - the bug
        pytest.fail(f"the camera raised on an orphan label: {exc}")
    assert isolated_db.get_attendance_for_today() == []
    assert any("9999" in e["message"] for e in mgr.status()["events"])
