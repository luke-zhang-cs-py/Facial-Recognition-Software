"""camera.py needs hardware to run, but its decision logic does not."""
import numpy as np
import pytest


def test_capture_plan_totals_the_advertised_sample_count():
    import camera
    assert sum(s["count"] for s in camera.CAPTURE_PLAN) == camera.SAMPLES_TO_CAPTURE


def test_capture_plan_covers_distinct_poses():
    import camera
    keys = [s["key"] for s in camera.CAPTURE_PLAN]
    assert len(keys) == len(set(keys))
    assert keys[0] == "front", "the neutral pitch baseline comes from the front stage"


@pytest.mark.parametrize("key,yaw,ok", [
    ("front", 3, True), ("front", 30, False),
    ("left", -20, True), ("left", 20, False), ("left", -5, False),
    ("right", 20, True), ("right", -20, False),
])
def test_pose_gate_yaw(key, yaw, ok):
    import camera
    assert camera.pose_matches(key, yaw, 0.0, 0.5, 0.5) is ok


def test_pose_gate_pitch_is_relative_to_the_persons_own_neutral():
    """Absolute pitch would encode face proportions, not head position."""
    import camera
    base = 0.50
    assert camera.pose_matches("up", 0, 0, base - 0.10, base)
    assert not camera.pose_matches("up", 0, 0, base, base)
    assert camera.pose_matches("down", 0, 0, base + 0.10, base)
    # a different person, different neutral, same relative movement
    other = 0.62
    assert camera.pose_matches("up", 0, 0, other - 0.10, other)


def test_pose_gate_needs_a_baseline_before_judging_pitch():
    import camera
    assert not camera.pose_matches("up", 0, 0, 0.4, None)


def test_pose_gate_rejects_missing_yaw():
    import camera
    assert not camera.pose_matches("front", None, 0, 0.5, 0.5)


def test_draw_face_marks_points_without_a_full_box():
    import camera
    mgr = camera.CameraManager.__new__(camera.CameraManager)
    frame = np.full((480, 640, 3), 60, np.uint8)
    face = {"box": (240, 150, 160, 190),
            "landmarks": [[285, 215], [355, 215], [320, 258], [292, 300], [348, 300]],
            "points68": None}
    before = frame.copy()
    mgr._draw_face(frame, face, camera.GREEN)
    diff = (np.abs(frame.astype(int) - before.astype(int)).sum(axis=2) > 0)
    x, y, w, h = face["box"]
    run = cur = 0
    for v in diff[y, x:x + w]:
        cur = cur + 1 if v else 0
        run = max(run, cur)
    assert run < w * 0.5, "corner ticks, not a solid rectangle"
    for lx, ly in face["landmarks"]:
        assert diff[ly - 1:ly + 2, lx - 1:lx + 2].any()


def test_draw_face_survives_missing_landmarks():
    import camera
    mgr = camera.CameraManager.__new__(camera.CameraManager)
    frame = np.full((100, 100, 3), 60, np.uint8)
    mgr._draw_face(frame, {"box": (10, 10, 40, 40), "landmarks": None,
                           "points68": None}, camera.GREEN, label="x")


def test_overlay_colours_match_the_stylesheet_palette():
    import camera
    import re
    css = open("static/css/style.css", encoding="utf-8").read()
    def bgr_of(var):
        m = re.search(rf"--{var}:#([0-9a-fA-F]{{6}})", css)
        h = m.group(1)
        return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
    assert camera.GREEN == bgr_of("good")
    assert camera.RED == bgr_of("bad")
    assert camera.AMBER == bgr_of("warn")
