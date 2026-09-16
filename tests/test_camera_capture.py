"""The grab loop, driven without a webcam.

`camera.py` was the least-covered module in the project at 35%, and the
reason given was that it needs hardware. That was true of exactly one call
-- `cv2.VideoCapture(0, cv2.CAP_DSHOW)` -- and not true of the 400-odd
statements downstream of it. `CameraManager` asks a device for three things:
whether it is open, a frame, and a release. A list of numpy arrays answers
all three.

So the device is injected now, and these tests run the real loop against a
synthetic source. What they do *not* test is the camera itself: whether
DirectShow opens index 0 on this machine is still something only a machine
with a camera can answer. The loop's logic and its failure branches are no
longer waiting on that.

One branch here is worth more than the coverage: a read that fails
part-way through a session. On real hardware that means unplugging the
webcam at the right moment; against a fake it is a list that runs out.
"""
import threading

import cv2
import numpy as np
import pytest

import camera


class FakeCapture:
    """A capture device backed by a list of frames.

    Mirrors the slice of the cv2.VideoCapture API that `CameraManager`
    actually uses, camelCase included, so the object under test cannot tell
    the difference.
    """

    def __init__(self, frames=(), opened=True, on_empty=None, boom=None):
        self._frames = list(frames)
        self._opened = opened
        self._on_empty = on_empty
        self._boom = boom
        self.reads = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        self.reads += 1
        if self._boom is not None:
            raise self._boom
        if self._frames:
            return True, self._frames.pop(0)
        # The loop is `while True`, so something has to end it. A real
        # session ends when somebody presses stop; here the frames running
        # out is the signal, and the callback is how the test says so.
        if self._on_empty is not None:
            self._on_empty()
        return False, None

    def release(self):
        self.released = True
        self._opened = False


@pytest.fixture
def manager():
    """A manager with no device attached yet.

    Constructing one loads the Haar cascade from disk, which is a file read
    rather than a camera, so this works anywhere OpenCV is installed.
    """
    return camera.CameraManager(capture_factory=lambda: FakeCapture())


def run_loop_once(mgr, cap):
    """Run the real capture loop over `cap` until its frames run out."""
    mgr._cap = cap
    mgr._running = True
    # Stop after the frames are gone, so the loop terminates on its own.
    cap._on_empty = lambda: setattr(mgr, "_running", False)
    mgr._capture_loop()


# --------------------------------------------------------------- lifecycle


def test_start_opens_the_injected_device(blank_frame):
    opened = []

    def factory():
        cap = FakeCapture([blank_frame.copy()])
        opened.append(cap)
        return cap

    mgr = camera.CameraManager(capture_factory=factory)
    mgr.start()
    try:
        assert mgr.status()["running"] is True
        assert len(opened) == 1, "the factory should be asked exactly once"
    finally:
        mgr.stop()
    assert opened[0].released, "stop() must release the device"


def test_start_is_idempotent(blank_frame):
    opened = []

    def factory():
        cap = FakeCapture([blank_frame.copy()] * 3)
        opened.append(cap)
        return cap

    mgr = camera.CameraManager(capture_factory=factory)
    mgr.start()
    try:
        mgr.start()
        assert len(opened) == 1, "a second start opened a second device"
    finally:
        mgr.stop()


def test_start_refuses_a_device_that_will_not_open():
    """The message names the usual cause, because "camera failed" is not
    something a user can act on."""
    cap = FakeCapture(opened=False)
    mgr = camera.CameraManager(capture_factory=lambda: cap)
    with pytest.raises(camera.CameraError) as raised:
        mgr.start()
    assert "webcam" in str(raised.value).lower()
    assert cap.released, "a device that would not open was left held open"
    assert mgr.status()["running"] is False


def test_stop_resets_the_mode_and_drops_the_last_frame(blank_frame):
    cap = FakeCapture([blank_frame.copy()] * 2)
    mgr = camera.CameraManager(capture_factory=lambda: cap)
    mgr.start()
    mgr._mode = camera.MODE_ATTENDANCE
    mgr.stop()
    status = mgr.status()
    assert status["running"] is False
    assert status["mode"] == camera.MODE_IDLE
    assert mgr._latest_jpeg is None, "a stale frame outlived the session"


def test_stop_is_safe_without_a_start():
    mgr = camera.CameraManager(capture_factory=lambda: FakeCapture())
    mgr.stop()
    assert mgr.status()["running"] is False


# -------------------------------------------------------------- the loop


def test_the_loop_publishes_a_jpeg(manager, blank_frame):
    run_loop_once(manager, FakeCapture([blank_frame.copy()]))
    jpeg = manager._latest_jpeg
    assert jpeg, "the loop produced no frame for the stream"
    assert jpeg[:2] == b"\xff\xd8", "not a JPEG"
    assert jpeg[-2:] == b"\xff\xd9", "truncated JPEG"


def test_the_loop_mirrors_the_frame(manager):
    """The feed reads like a mirror, so the image is flipped horizontally."""
    frame = np.zeros((64, 64, 3), np.uint8)
    frame[:, :32] = 255                      # bright on the left
    run_loop_once(manager, FakeCapture([frame]))

    shown = cv2.imdecode(np.frombuffer(manager._latest_jpeg, np.uint8),
                         cv2.IMREAD_COLOR)
    left = float(shown[:, :32].mean())
    right = float(shown[:, 32:].mean())
    assert right > left + 100, (
        "the bright half did not move to the other side: "
        f"left={left:.0f} right={right:.0f}")


def test_a_failed_read_is_reported_and_the_loop_survives(manager):
    """An empty FakeCapture is a camera that has stopped handing over
    frames -- unplugged, or claimed by another app mid-session."""
    run_loop_once(manager, FakeCapture([]))
    assert manager.status()["error"] == "Lost the camera feed."


@pytest.mark.parametrize("mode, handler", [
    (camera.MODE_IDLE, "_handle_idle"),
    (camera.MODE_REGISTER, "_handle_register"),
    (camera.MODE_ATTENDANCE, "_handle_attendance"),
])
def test_the_loop_dispatches_on_mode(manager, blank_frame, monkeypatch,
                                     mode, handler):
    seen = []
    monkeypatch.setattr(manager, handler,
                        lambda frame: seen.append(frame.shape))
    manager._mode = mode
    run_loop_once(manager, FakeCapture([blank_frame.copy()]))
    assert seen, f"{mode} did not reach {handler}"


def test_a_handler_failure_does_not_take_the_stream_down(manager, blank_frame):
    """A broken overlay used to kill the capture thread while `running`
    stayed True: the stream stopped and nothing said why.

    The handler ends the session itself rather than letting the frames run
    out. Stopping the usual way would have the loop read once more, find
    nothing, and overwrite the handler's message with "Lost the camera
    feed." -- which is the error this test is not about.
    """
    def explode(frame):
        manager._running = False
        raise RuntimeError("overlay blew up")

    manager._handle_idle = explode
    manager._cap = FakeCapture([blank_frame.copy()])
    manager._running = True
    manager._capture_loop()

    assert manager.status()["error"] == "overlay blew up"
    assert manager._latest_jpeg, "the frame was dropped along with the overlay"


def test_the_analysis_sees_the_unannotated_frame(manager, blank_frame):
    """The handlers paint onto `frame`; the trait read must get the copy
    taken before that, or every score is measured through a wireframe."""
    painted = []

    def draw_all_over_it(frame):
        frame[:, :] = 7                      # obliterate the image

    def capture_what_analysis_saw(frame):
        painted.append(frame.copy())

    manager._handle_idle = draw_all_over_it
    manager._maybe_traits = capture_what_analysis_saw
    run_loop_once(manager, FakeCapture([blank_frame.copy()]))

    assert painted, "the trait read never ran"
    assert painted[0].mean() != pytest.approx(7, abs=0.5), (
        "the analysis was handed the annotated frame")


def test_the_thread_records_why_it_died(manager):
    """`_loop` is the last resort: a dead thread with running=True looks
    exactly like a broken camera from the browser's side."""
    manager._cap = FakeCapture(boom=RuntimeError("device exploded"))
    manager._running = True
    manager._loop()

    status = manager.status()
    assert status["running"] is False
    assert "device exploded" in status["error"]
    assert any("device exploded" in e["message"] for e in status["events"])


# ----------------------------------------------------------- the MJPEG feed


def test_frames_yields_multipart_chunks(manager, blank_frame):
    run_loop_once(manager, FakeCapture([blank_frame.copy()]))
    manager._running = True
    try:
        chunk = next(manager.frames())
    finally:
        manager._running = False
    assert chunk.startswith(b"--frame")
    assert b"image/jpeg" in chunk
    assert chunk.rstrip().endswith(b"\xff\xd9")


def test_frames_ends_when_the_camera_stops(manager):
    manager._running = False
    assert list(manager.frames()) == []


def test_frames_waits_rather_than_yielding_nothing(manager):
    """No frame grabbed yet is not the same as end of stream."""
    manager._running = True
    manager._latest_jpeg = None
    produced = []

    def pull():
        for chunk in manager.frames():
            produced.append(chunk)
            break

    reader = threading.Thread(target=pull, daemon=True)
    reader.start()
    reader.join(timeout=0.3)
    assert produced == [], "it yielded a chunk with no frame to send"
    manager._running = False
    reader.join(timeout=1.0)


# ------------------------------------------------------- per-frame helpers


def test_detect_finds_no_face_in_a_flat_frame(manager, blank_frame):
    gray, faces = manager._detect(blank_frame)
    assert gray.shape == blank_frame.shape[:2], "the grey frame is the wrong size"
    assert faces == [], "a flat grey rectangle was read as a face"


def test_idle_mode_leaves_a_faceless_frame_alone(manager, blank_frame):
    before = blank_frame.copy()
    manager._handle_idle(blank_frame)
    assert np.array_equal(blank_frame, before), (
        "something was drawn when no face was found")


def test_attendance_waits_for_a_model(manager, blank_frame):
    """No recognizer means nothing to compare against, so the handler
    returns rather than guessing."""
    manager._recognizer = None
    manager._handle_attendance(blank_frame)
    assert manager.status()["error"] is None


def test_traits_stay_off_when_switched_off(manager, blank_frame):
    manager.set_traits_enabled(False)
    manager._last_trait_at = 0.0
    manager._maybe_traits(blank_frame)
    assert manager.status()["liveTraits"] is None
    assert manager.status()["traitsOn"] is False


def test_set_idle_returns_to_idle(manager):
    manager._mode = camera.MODE_REGISTER
    manager.set_idle()
    assert manager.status()["mode"] == camera.MODE_IDLE


def test_the_default_factory_is_the_real_camera():
    """The seam must not change what production does: with no factory
    passed, the manager still reaches for the webcam."""
    mgr = camera.CameraManager()
    assert mgr._open_capture is camera.open_default_camera


# ------------------------------------------------- the trait readout's guards
#
# "The readout and the overlay are cosmetic; a failure in either must not
# take the video feed down with it." These two pin that.


def test_a_failed_trait_read_is_reported_not_raised(manager, blank_frame,
                                                    monkeypatch):
    monkeypatch.setattr(camera.facetraits, "analyze",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("model missing")))
    manager._last_trait_at = 0.0
    manager._maybe_traits(blank_frame)

    assert manager.status()["liveTraits"] == {"error": "model missing"}


def test_no_face_means_no_readout_rather_than_an_empty_one(manager,
                                                           blank_frame,
                                                           monkeypatch):
    """`analyze` returns None when it was told detection is required and
    found none. A blank panel would imply a measurement was taken."""
    monkeypatch.setattr(camera.facetraits, "analyze", lambda *a, **k: None)
    manager._last_trait_at = 0.0
    manager._maybe_traits(blank_frame)

    assert manager.status()["liveTraits"] is None


def test_the_readout_is_throttled(manager, blank_frame, monkeypatch):
    """A full trait read runs five networks at ~130 ms; once a second is
    plenty for something a human is reading."""
    calls = []
    monkeypatch.setattr(camera.facetraits, "analyze",
                        lambda *a, **k: calls.append(1) or None)
    manager._last_trait_at = 0.0
    manager._maybe_traits(blank_frame)
    manager._maybe_traits(blank_frame)      # immediately again

    assert len(calls) == 1, "the throttle let a second read through"


# ------------------------------------------------------- the landmark mesh


def test_the_mesh_is_drawn_when_the_fit_succeeded(manager, blank_frame,
                                                  monkeypatch):
    """With 68 points available the overlay draws the mesh rather than the
    five-dot fallback, and the colour carries the capture state."""
    drawn = []
    monkeypatch.setattr(camera.facelandmarks, "draw",
                        lambda frame, pts, colour: drawn.append(colour))
    face = {"box": (200, 150, 200, 200),
            "landmarks": [[250, 210], [350, 210], [300, 260],
                          [265, 310], [335, 310]],
            "points68": np.zeros((68, 2), np.float32)}

    manager._draw_face(blank_frame, face, camera.GREEN)

    assert drawn == [camera.GREEN], "the mesh was not drawn in the state colour"


def test_a_failed_landmark_fit_is_not_fatal(manager, blank_frame, monkeypatch):
    """The 68-point fit is an extra on top of detection. Losing it should
    cost the mesh, not the face."""
    row = [200.0, 150.0, 200.0, 200.0,
           250.0, 210.0, 350.0, 210.0, 300.0, 260.0,
           265.0, 310.0, 335.0, 310.0, 0.99]
    monkeypatch.setattr(camera.facetraits, "detect", lambda frame: [row])
    monkeypatch.setattr(camera.facelandmarks, "fit",
                        lambda gray, box: (_ for _ in ()).throw(
                            RuntimeError("no landmark model")))

    _gray, faces = manager._detect(blank_frame)

    assert len(faces) == 1, "the face was lost with the mesh"
    assert faces[0]["points68"] is None
