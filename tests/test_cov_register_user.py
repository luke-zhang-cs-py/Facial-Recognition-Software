"""cli/register_user.py's capture loop, driven by a fake camera.

The frame source is a list of flat frames and the detector is traits.detect
replaced with fixed rows, so the loop's own decisions are what is tested:
when a sample counts, when the loop stops, and what is said at the end.
cv2's window calls are stubbed; nothing opens a camera or a window.
"""
import os

import cv2
import numpy as np
import pytest

from cli import register_user
from core import db, paths
from pipeline import traits


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.reads = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        self.reads += 1
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True


def row(x, y, w, h):
    r = np.zeros(15, np.float32)
    r[:4] = [x, y, w, h]
    return r


@pytest.fixture
def screen(monkeypatch):
    """Stub the window; `keys` is the sequence waitKey returns."""
    keys = []
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda title, f: shown.append(f.copy()))
    monkeypatch.setattr(cv2, "waitKey",
                        lambda delay: keys.pop(0) if keys else -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return keys, shown


def frames(n):
    return [np.full((240, 320, 3), 100 + i, np.uint8) for i in range(n)]


def detections(monkeypatch, per_frame):
    """traits.detect answers from this list, one entry per call."""
    queue = list(per_frame)
    monkeypatch.setattr(traits, "detect",
                        lambda bgr: queue.pop(0) if queue else [])


def run(monkeypatch, cap, name="Ada"):
    monkeypatch.setattr(register_user, "_open_camera", lambda: cap)
    register_user.register_user(name)
    folder = paths.user_folder(1, name)
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else None


def test_capture_stops_at_the_sample_limit(isolated_db, monkeypatch, screen,
                                           capsys):
    monkeypatch.setattr(register_user, "SAMPLES_TO_CAPTURE", 3)
    detections(monkeypatch, [[row(50, 40, 100, 100)]] * 10)
    cap = FakeCapture(frames(10))
    written = run(monkeypatch, cap)
    assert written == ["1.jpg", "2.jpg", "3.jpg"]
    assert cap.reads == 3, "not one frame read past the limit"
    assert cap.released
    img = cv2.imread(os.path.join(paths.user_folder(1, "Ada"), "1.jpg"),
                     cv2.IMREAD_GRAYSCALE)
    assert img.shape == register_user.decision.vision.LBPH_INPUT_SIZE[::-1]
    assert "Done. Captured 3 samples for 'Ada' (user_id=1)." in \
        capsys.readouterr().out


def test_frames_without_a_usable_face_do_not_count(isolated_db, monkeypatch,
                                                   screen, capsys):
    monkeypatch.setattr(register_user, "SAMPLES_TO_CAPTURE", 2)
    detections(monkeypatch, [
        [],                                   # nobody
        [row(900, 900, 50, 50)],              # a box entirely off the frame
        [row(10, 10, 60, 60), row(100, 50, 120, 120)],   # largest one counts
        [row(20, 20, 80, 80)],
    ])
    cap = FakeCapture(frames(6))
    written = run(monkeypatch, cap)
    assert written == ["1.jpg", "2.jpg"] and cap.reads == 4
    _, shown = screen
    assert len(shown) == 4, "every frame is previewed, counted or not"
    third = shown[2]
    assert (third[50, 100:220] == register_user.CAPTURE_COLOUR).all(), \
        "the box drawn is the largest face's"


def test_a_dropped_camera_ends_the_capture(isolated_db, monkeypatch, screen,
                                           capsys):
    detections(monkeypatch, [[row(50, 40, 100, 100)]] * 2)
    cap = FakeCapture(frames(2))
    written = run(monkeypatch, cap)
    out = capsys.readouterr().out
    assert written == ["1.jpg", "2.jpg"]
    assert "Failed to grab frame from webcam." in out
    assert "Captured 2 samples" in out and cap.released


def test_q_stops_early(isolated_db, monkeypatch, screen, capsys):
    keys, _ = screen
    keys.extend([-1, ord("q")])
    detections(monkeypatch, [[row(50, 40, 100, 100)]] * 5)
    cap = FakeCapture(frames(5))
    written = run(monkeypatch, cap)
    assert written == ["1.jpg", "2.jpg"] and cap.reads == 2


def test_q_with_modifier_bits_still_stops(isolated_db, monkeypatch, screen):
    """waitKey can carry flags above the low byte; only the key is compared."""
    keys, _ = screen
    keys.append(0x100000 | ord("q"))
    detections(monkeypatch, [[]])
    cap = FakeCapture(frames(5))
    assert run(monkeypatch, cap) == [] and cap.reads == 1


def test_nobody_in_front_of_the_camera_says_registration_is_incomplete(
        isolated_db, monkeypatch, screen, capsys):
    detections(monkeypatch, [])
    run(monkeypatch, FakeCapture(frames(3)))
    out = capsys.readouterr().out
    assert "No face samples captured — registration incomplete." in out
    assert "train_model" not in out


def test_a_camera_that_will_not_open(isolated_db, monkeypatch, capsys):
    cap = FakeCapture([], opened=False)
    monkeypatch.setattr(register_user.cv2, "VideoCapture", lambda index: cap)
    assert register_user._open_camera() is None
    assert cap.released
    assert "Could not open webcam" in capsys.readouterr().out


def test_register_stops_when_the_camera_will_not_open(isolated_db,
                                                      monkeypatch, capsys):
    """The row and the folder are made first, as the module documents; no
    capture is attempted."""
    monkeypatch.setattr(register_user, "_open_camera", lambda: None)
    reads = []
    monkeypatch.setattr(traits, "detect", lambda bgr: reads.append(1) or [])
    register_user.register_user("Ada")
    assert reads == []
    assert db.get_all_users() == [(1, "Ada")]
    assert "Captured" not in capsys.readouterr().out


def test_an_open_camera_is_returned(monkeypatch):
    cap = FakeCapture([])
    monkeypatch.setattr(register_user.cv2, "VideoCapture", lambda index: cap)
    assert register_user._open_camera() is cap and not cap.released


def test_save_sample_of_a_box_outside_the_frame(tmp_path):
    gray = np.zeros((50, 50), np.uint8)
    assert register_user.save_sample(gray, (60, 60, 10, 10),
                                     str(tmp_path), 1) is None
    assert os.listdir(tmp_path) == []


# ------------------------------------------------------------------ main

@pytest.mark.parametrize("argv", [["  "], ["--", "-Ada"], ["--", "  -x "]])
def test_main_refuses_a_blank_or_flag_like_name(isolated_db, monkeypatch,
                                                capsys, argv):
    called = []
    monkeypatch.setattr(register_user, "register_user", called.append)
    assert register_user.main(argv) == 1
    assert called == []
    assert 'Usage: python -m cli.register_user "Full Name"' in \
        capsys.readouterr().out


def test_main_rejects_an_unknown_option(monkeypatch, capsys):
    monkeypatch.setattr(register_user, "register_user",
                        lambda name: pytest.fail("registered"))
    assert register_user.main(["--bogus"]) == 1


def test_main_strips_the_name_and_registers(monkeypatch):
    called = []
    monkeypatch.setattr(register_user, "register_user", called.append)
    assert register_user.main(["  Grace Hopper  "]) == 0
    assert called == ["Grace Hopper"]


def test_main_reads_sys_argv_when_given_nothing(monkeypatch):
    called = []
    monkeypatch.setattr(register_user, "register_user", called.append)
    monkeypatch.setattr(register_user.sys, "argv", ["register_user", "Ada"])
    assert register_user.main() == 0 and called == ["Ada"]
