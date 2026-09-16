"""The register and attendance state machines, driven without a webcam.

These two handlers were the largest uncovered block in the project -- 85
statements between them -- and they are not hardware code. They take a
frame and a detected face and decide what to do: write a sample, ignore a
wrong pose, mark somebody present, refuse a photograph. What kept them
untested was that a detected face normally arrives from a camera pointed at
a person.

So the face is synthesised at the detector boundary instead. `facetraits.detect`
returns raw YuNet rows, and the real `geometry()` turns a row into box,
landmarks, yaw, roll and pitch -- so a made-up row still exercises the
genuine pose maths rather than a stubbed-out version of it. Only the
detection itself is replaced.

What is still not covered here is whether YuNet finds a real face in a real
frame; `tests/test_traits_landmarks.py` covers that when the sample corpus
is present, and skips when it is not.
"""
import os

import cv2
import numpy as np
import pytest

import camera


# A frontal face, 200x200 at (200, 150). Eyes level and 100px apart, nose on
# the eye midline -> yaw 0, roll 0. Mouth 100px below the eyes with the nose
# halfway between -> pitchRatio 0.5.
def yunet_row(nose_x=300.0, nose_y=260.0, eye_y=210.0, score=0.99):
    return [
        200.0, 150.0, 200.0, 200.0,        # box
        250.0, eye_y,                      # right eye
        350.0, eye_y,                      # left eye
        nose_x, nose_y,                    # nose
        265.0, 310.0,                      # right mouth corner
        335.0, 310.0,                      # left mouth corner
        score,
    ]


FRONTAL = yunet_row()
TURNED_RIGHT = yunet_row(nose_x=340.0)     # nose 40px off centre -> yaw 30
LOOKING_DOWN = yunet_row(nose_y=285.0)     # nose lower -> pitchRatio 0.75


def photo_frame():
    """A frame that passes the sample-quality gate, fresh on every call.

    Two things this fixes about using `blank_frame` for registration. A flat
    grey rectangle is not a photograph of anything -- 0 sharpness, 0 tonal
    range -- so the gate rejects it, correctly. And a *shared* frame is
    worse than unrealistic: `_handle_register` draws its overlay onto the
    frame it is handed, so reusing one array meant the label drawn on a
    rejected frame became the texture that let the next call through. A
    camera hands over a new frame every iteration; so does this.
    """
    ramp = np.tile(np.linspace(20, 235, 640, dtype=np.uint8), (480, 1))
    frame = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    # Structure at roughly face scale, inside the box the fake rows report.
    for cx, cy, r in ((260, 220, 26), (340, 220, 26),
                      (300, 270, 18), (300, 310, 34)):
        cv2.circle(frame, (cx, cy), r, (40, 45, 60), -1)
    return cv2.GaussianBlur(frame, (5, 5), 0)


@pytest.fixture
def photo():
    """Hand out a fresh passing frame on demand."""
    return photo_frame


@pytest.fixture
def faces(monkeypatch):
    """Put chosen YuNet rows in front of the detector."""
    def use(*rows):
        monkeypatch.setattr(camera.facetraits, "detect",
                            lambda frame: list(rows))
    return use


@pytest.fixture
def mgr(monkeypatch):
    manager = camera.CameraManager(capture_factory=lambda: None)
    # The 68-point fit is a separate model and not what these tests are
    # about; the handlers only use the box and the pose.
    monkeypatch.setattr(camera.facelandmarks, "fit", lambda gray, box: None)
    return manager


@pytest.fixture
def registering(mgr, isolated_root):
    """A manager mid-registration, writing into a throwaway directory."""
    user_dir = os.path.join(isolated_root, "samples", "1")
    os.makedirs(user_dir, exist_ok=True)
    mgr._mode = camera.MODE_REGISTER
    mgr._reg_name = "Test Person"
    mgr._reg_user_id = 1
    mgr._reg_dir = user_dir
    return mgr


# ------------------------------------------------------------ registration


def test_the_right_pose_writes_a_sample(registering, faces, photo):
    faces(FRONTAL)
    registering._handle_register(photo())

    assert registering._reg_count == 1
    assert registering._reg_stage_count == 1
    written = os.listdir(registering._reg_dir)
    assert written == ["1.jpg"], written
    assert registering._reg_poses[0]["stage"] == "front"


def test_the_wrong_pose_is_ignored_rather_than_counted(registering, faces,
                                                       photo):
    """Stage 0 asks for a frontal face; a turned head is not a failure to
    report, it is a frame to skip. The instruction on screen already says
    what to do."""
    faces(TURNED_RIGHT)
    registering._handle_register(photo())

    assert registering._reg_count == 0
    assert os.listdir(registering._reg_dir) == []


def test_no_face_means_nothing_happens(registering, faces, blank_frame):
    faces()
    registering._handle_register(blank_frame)
    assert registering._reg_count == 0


def test_a_finished_plan_stops_capturing(registering, faces, blank_frame):
    """Past the last stage there is nothing left to ask for."""
    registering._reg_stage = len(camera.CAPTURE_PLAN)
    faces(FRONTAL)
    registering._handle_register(blank_frame)
    assert registering._reg_count == 0


def test_completing_a_stage_advances_and_sets_the_baseline(registering, faces,
                                                           photo):
    """The neutral pitch measured during the front stage is what the up and
    down stages are later compared against, so it has to be recorded when
    that stage closes."""
    faces(FRONTAL)
    wanted = camera.CAPTURE_PLAN[0]["count"]
    for _ in range(wanted):
        registering._handle_register(photo())

    assert registering._reg_count == wanted
    assert registering._reg_stage == 1, "the stage did not advance"
    assert registering._reg_stage_count == 0, "the per-stage tally did not reset"
    assert registering._reg_baseline_pitch == pytest.approx(0.5, abs=0.01)
    assert len(os.listdir(registering._reg_dir)) == wanted


def test_the_pitch_baseline_is_the_median_not_the_last(registering, faces,
                                                       photo):
    """One bad frame at the end of the stage must not become the neutral."""
    wanted = camera.CAPTURE_PLAN[0]["count"]
    for i in range(wanted):
        faces(FRONTAL if i < wanted - 1 else LOOKING_DOWN)
        registering._handle_register(photo())

    assert registering._reg_baseline_pitch == pytest.approx(0.5, abs=0.01), (
        "the outlier frame became the baseline")


# -------------------------------------------------------------- attendance


class FakeRecognizer:
    """An LBPH stand-in: predict() answers with a label and a distance."""

    def __init__(self, user_id=1, confidence=20.0):
        self.user_id = user_id
        self.confidence = confidence
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return self.user_id, self.confidence


@pytest.fixture
def attending(mgr, isolated_db, monkeypatch):
    user_id = isolated_db.add_user("Ada")
    mgr._mode = camera.MODE_ATTENDANCE
    mgr._recognizer = FakeRecognizer(user_id=user_id)
    # Liveness runs its own model on the frame; these tests are about what
    # the handler does with a verdict, not how the verdict is reached.
    monkeypatch.setattr(camera.faceliveness, "score", lambda frame, box: 0.9)
    mgr._user_id = user_id
    return mgr


def set_verdict(manager, monkeypatch, verdict):
    monkeypatch.setattr(manager._liveness, "verdict", lambda: verdict)


def test_a_genuine_match_is_marked_present(attending, faces, blank_frame,
                                           monkeypatch, isolated_db):
    set_verdict(attending, monkeypatch, "genuine")
    faces(FRONTAL)
    attending._handle_attendance(blank_frame)

    assert attending._recognizer.calls == 1
    assert attending._user_id in attending._marked_session
    assert any("present" in e["message"] for e in attending.status()["events"])


def test_the_same_person_is_not_marked_twice(attending, faces, blank_frame,
                                             monkeypatch):
    set_verdict(attending, monkeypatch, "genuine")
    faces(FRONTAL)
    attending._handle_attendance(blank_frame)
    before = len(attending.status()["events"])
    attending._handle_attendance(blank_frame)

    assert len(attending.status()["events"]) == before, (
        "a second frame of the same person logged another event")


def test_a_photograph_is_refused_but_still_named(attending, faces, blank_frame,
                                                 monkeypatch):
    """Somebody genuine in bad light needs to know why they are refused, so
    the label names them rather than going blank."""
    set_verdict(attending, monkeypatch, "spoof")
    faces(FRONTAL)
    attending._handle_attendance(blank_frame)

    assert attending._user_id not in attending._marked_session
    assert attending.status()["liveness"]["verdict"] == "spoof"


def test_an_undecided_verdict_waits(attending, faces, blank_frame, monkeypatch):
    """Still gathering frames is not the same as a refusal."""
    set_verdict(attending, monkeypatch, "unknown")
    faces(FRONTAL)
    attending._handle_attendance(blank_frame)

    assert attending._marked_session == set()


def test_a_poor_match_is_not_marked(attending, faces, blank_frame, monkeypatch):
    """Above the distance threshold is a stranger, whatever the liveness
    verdict says."""
    set_verdict(attending, monkeypatch, "genuine")
    attending._recognizer = FakeRecognizer(
        user_id=attending._user_id,
        confidence=camera.CONFIDENCE_THRESHOLD + 10)
    faces(FRONTAL)
    attending._handle_attendance(blank_frame)

    assert attending._marked_session == set()


def test_a_face_outside_the_frame_is_skipped(attending, faces, monkeypatch):
    """An empty crop cannot be recognised; the handler moves on rather than
    handing OpenCV a zero-size image."""
    set_verdict(attending, monkeypatch, "genuine")
    tiny = np.full((40, 40, 3), 90, np.uint8)   # box is at (200, 150)
    faces(FRONTAL)
    attending._handle_attendance(tiny)

    assert attending._recognizer.calls == 0
    assert attending._marked_session == set()


# ---------------------------------------------------------- entering a mode
#
# `start()` is covered against a synthetic device in test_camera_capture.py.
# Here it is replaced by a recorder: what these tests are about is the state
# each mode sets up, and the assertion that the camera is asked to open at
# all -- not the grab thread, which would only make them slow and racy.


@pytest.fixture
def opens(mgr, monkeypatch):
    """Record that the camera was asked to start, without starting it."""
    calls = []
    monkeypatch.setattr(mgr, "start", lambda: calls.append(True))
    return calls


def test_registering_needs_a_name(mgr, opens, isolated_db):
    for blank in ("", "   ", None):
        with pytest.raises(camera.CameraError) as raised:
            mgr.start_register(blank)
        assert "name" in str(raised.value).lower()
    assert opens == [], "it opened the camera before checking the name"
    assert mgr.status()["mode"] == camera.MODE_IDLE


def test_registering_creates_the_user_and_its_folder(mgr, opens, isolated_db,
                                                     isolated_root):
    user_id = mgr.start_register("Ada Lovelace")

    assert opens == [True], "the camera was not started"
    assert isolated_db.get_user_name(user_id) == "Ada Lovelace"
    # Spaces become underscores so the folder name stays one token, and the
    # id leads so train_model can parse the label back out of it.
    expected = os.path.join(camera.paths.dataset_dir(),
                            f"{user_id}_Ada_Lovelace")
    assert os.path.isdir(expected), os.listdir(camera.paths.dataset_dir())

    status = mgr.status()
    assert status["mode"] == camera.MODE_REGISTER
    assert status["register"]["name"] == "Ada Lovelace"
    assert status["register"]["captured"] == 0


def test_registering_again_clears_the_last_run(mgr, opens, isolated_db,
                                               isolated_root):
    """Stale counters from a previous registration would make the progress
    bar start part-full."""
    mgr.start_register("Ada")
    mgr._reg_count = 17
    mgr._reg_stage = 3
    mgr._reg_finished = True
    mgr._reg_report = {"stale": True}
    mgr._reg_poses = [{"stage": "front"}]

    mgr.start_register("Grace")

    status = mgr.status()
    assert status["register"]["captured"] == 0
    assert status["register"]["stage"] == 0
    assert status["register"]["finished"] is False
    assert status["report"] is None
    assert mgr._reg_poses == []


def test_attendance_refuses_when_nobody_is_enrolled(mgr, opens, isolated_root):
    """No model and nothing to train from is the one case that genuinely
    cannot proceed."""
    with pytest.raises(camera.CameraError) as raised:
        mgr.start_attendance()
    assert "enrolled" in str(raised.value).lower()
    assert opens == [], "it opened the camera for a mode it could not enter"


def test_attendance_trains_on_demand_rather_than_refusing(mgr, opens,
                                                          isolated_root):
    """Samples on disk but no trainer.yml used to send the user off to press
    a button this can press itself."""
    folder = os.path.join(camera.paths.dataset_dir(), "1_Ada")
    os.makedirs(folder, exist_ok=True)
    rng = np.random.RandomState(0)
    import cv2
    for i in range(2):
        cv2.imwrite(os.path.join(folder, f"{i}.jpg"),
                    rng.randint(0, 255, (200, 200), dtype=np.uint8))

    mgr.start_attendance()

    assert os.path.exists(camera.paths.model_path()), "it did not train"
    assert mgr.status()["mode"] == camera.MODE_ATTENDANCE
    assert opens == [True]
    assert mgr._recognizer is not None, "the fresh model was not loaded"


def test_attendance_starts_a_clean_session(mgr, opens, isolated_root):
    """Yesterday's marked set must not suppress today's first sighting."""
    folder = os.path.join(camera.paths.dataset_dir(), "1_Ada")
    os.makedirs(folder, exist_ok=True)
    rng = np.random.RandomState(1)
    import cv2
    cv2.imwrite(os.path.join(folder, "0.jpg"),
                rng.randint(0, 255, (200, 200), dtype=np.uint8))
    mgr.start_attendance()

    mgr._marked_session.add(99)
    mgr._live_verdict = "spoof"
    mgr.start_attendance()

    assert mgr._marked_session == set(), "a stale session survived"
    assert mgr.status()["liveness"]["verdict"] == "unknown"


# ------------------------------------------------- the whole plan, end to end
#
# Poses chosen against the gate's own constants: TURN_MIN 13 / TURN_MAX 38
# for the turns, PITCH_DELTA 0.055 either side of the 0.5 neutral for the
# chin stages, TILT_MAX_YAW 22 so those stay roughly frontal.

POSE_FOR_STAGE = {
    "front": FRONTAL,                        # yaw 0
    "left": yunet_row(nose_x=260.0),         # yaw -30
    "right": yunet_row(nose_x=340.0),        # yaw +30
    "up": yunet_row(nose_y=240.0),           # pitch 0.30, below 0.445
    "down": yunet_row(nose_y=280.0),         # pitch 0.70, above 0.555
}


def test_finishing_every_stage_closes_the_registration(mgr, faces,
                                                       isolated_db,
                                                       isolated_root,
                                                       monkeypatch,
                                                       photo):
    """The last sample of the last pose returns to idle, retrains, and
    leaves a report -- the whole point of the walkthrough."""
    monkeypatch.setattr(camera.enrollment, "build",
                        lambda uid, name, poses, folder: {"poses": len(poses)})
    user_id = isolated_db.add_user("Ada")
    folder = os.path.join(camera.paths.dataset_dir(), f"{user_id}_Ada")
    os.makedirs(folder, exist_ok=True)

    mgr._mode = camera.MODE_REGISTER
    mgr._reg_name = "Ada"
    mgr._reg_user_id = user_id
    mgr._reg_dir = folder

    for stage in camera.CAPTURE_PLAN:
        faces(POSE_FOR_STAGE[stage["key"]])
        for _ in range(stage["count"]):
            mgr._handle_register(photo())

    status = mgr.status()
    total = sum(p["count"] for p in camera.CAPTURE_PLAN)
    assert status["register"]["captured"] == total, (
        f"a stage stalled: {status['register']}")
    assert status["register"]["finished"] is True
    assert status["mode"] == camera.MODE_IDLE, "it stayed in register mode"
    assert status["report"] == {"poses": total}
    assert any(f"Captured {total} samples" in e["message"]
               for e in status["events"])


def test_a_face_off_the_edge_writes_nothing(registering, faces):
    """An empty crop cannot be saved as a sample."""
    faces(FRONTAL)
    tiny = np.full((40, 40, 3), 90, np.uint8)     # the box is at (200, 150)
    registering._handle_register(tiny)
    assert registering._reg_count == 0
    assert os.listdir(registering._reg_dir) == []


def test_already_marked_today_is_said_once(attending, faces, blank_frame,
                                           monkeypatch, isolated_db):
    """A second day's session re-marking somebody the database already has
    is not an error, and not a success either."""
    set_verdict(attending, monkeypatch, "genuine")
    monkeypatch.setattr(camera.db, "log_attendance",
                        lambda uid, conf: False)
    faces(FRONTAL)
    attending._handle_attendance(blank_frame)

    assert any("already marked today" in e["message"]
               for e in attending.status()["events"])


def test_idle_mode_outlines_a_face_it_finds(mgr, faces, blank_frame):
    before = blank_frame.copy()
    faces(FRONTAL)
    mgr._handle_idle(blank_frame)
    assert not np.array_equal(blank_frame, before), "no outline was drawn"


def test_an_unknown_stage_key_matches_nothing(mgr):
    """A defensive default: every key in CAPTURE_PLAN is handled above it,
    so this only fires if somebody adds a stage and forgets the gate."""
    assert camera.pose_matches("sideways", 0.0, 0.0, 0.5, 0.5) is False


# ------------------------------------------------- the sample-quality gate
#
# Registration used to accept any frame whose head was in the right
# position. enrollment.build then measured the sharpness and the quality
# score of what had already been written. The measurement was the only part
# the gate was missing, and it was being taken anyway.


def test_a_badly_photographed_frame_is_not_written(registering, faces,
                                                   blank_frame):
    """Flat grey is the degenerate case: no sharpness, no tonal range. It
    is not a photograph of anybody and does not belong in a gallery."""
    faces(FRONTAL)
    registering._handle_register(blank_frame)

    assert registering._reg_count == 0
    assert os.listdir(registering._reg_dir) == [], "a blank frame was enrolled"


def test_the_reason_is_put_on_screen(registering, faces, blank_frame,
                                     monkeypatch):
    """A counter that stops moving with no reason shown looks like the
    capture has broken. "too dark" is something a person can act on."""
    labels = []
    monkeypatch.setattr(registering, "_draw_face",
                        lambda frame, face, colour, label=None:
                        labels.append((colour, label)))
    faces(FRONTAL)
    registering._handle_register(blank_frame)

    assert labels, "nothing was drawn at all"
    colour, label = labels[-1]
    assert colour == camera.RED, "a rejected frame was not marked as rejected"
    assert "blurry" in label, label


def test_a_good_frame_still_gets_through(registering, faces, photo):
    """The gate has to reject bad photographs without rejecting the good
    ones -- a filter that takes nothing is not a filter."""
    faces(FRONTAL)
    registering._handle_register(photo())
    assert registering._reg_count == 1


def test_the_turn_stages_are_not_rejected_for_turning(registering, faces,
                                                      photo):
    """traits.MAX_YAW is 30 and the turn stages accept 13 to 38, so gating
    on the pose flags would reject the frames CAPTURE_PLAN exists to
    collect. Pose is the stage's job; this gate is about the photograph."""
    registering._reg_stage = 2                      # the "right" stage
    faces(yunet_row(nose_x=345.0))                  # yaw ~34, past MAX_YAW
    registering._handle_register(photo())

    assert registering._reg_count == 1, (
        "a deliberately turned head was rejected as 'turned away'")


def test_the_gate_reads_the_frame_before_the_overlay(registering, faces,
                                                     monkeypatch):
    """The quality score has to be measured on what the camera saw. Drawing
    first and measuring after cost 0.434 -> 0.407 on a real frame, which is
    the same mistake the trait read used to make."""
    seen = []
    real = camera.facetraits.quality_metrics
    monkeypatch.setattr(camera.facetraits, "quality_metrics",
                        lambda crop: seen.append(crop.copy()) or real(crop))
    faces(FRONTAL)
    frame = photo_frame()
    registering._handle_register(frame)

    assert seen, "quality was never measured"
    # The overlay draws in fixed palette colours; none should be present in
    # the crop the gate looked at.
    for colour in (camera.GREEN, camera.RED, camera.AMBER):
        assert not (seen[0] == np.array(colour, np.uint8)).all(axis=2).any(), (
            "the gate measured a frame that had already been drawn on")


def test_pose_flags_are_the_only_ones_dropped():
    """If traits grows a new reason, it should reach the gate rather than be
    silently filtered out with the pose ones."""
    assert camera.POSE_FLAGS == {"turned away", "head tilted"}


def test_an_empty_crop_is_not_judged(mgr):
    """Called directly, the gate has to survive a zero-size crop rather
    than handing it to the quality model. _handle_register already checks,
    so this is the belt to that braces."""
    empty = np.zeros((0, 0, 3), np.uint8)
    assert camera.weak_photograph(empty, {"box": (0, 0, 0, 0)}) == []
    assert camera.weak_photograph(None, {"box": (0, 0, 0, 0)}) == []
