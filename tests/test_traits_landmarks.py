"""traits.py and landmarks.py produce the numbers everything else reasons
about, and both have had bugs that only showed up at the JSON boundary or in
degenerate geometry."""
import json

import numpy as np
import pytest


def test_geometry_yaw_is_bounded_even_when_the_eyes_coincide():
    """Regression: yaw divided by inter-eye distance and ran to -689..+470
    when a head turned towards profile."""
    from pipeline import traits
    row = np.zeros(15, np.float32)
    row[:4] = [200, 150, 160, 190]
    row[4:14] = [250, 215, 250, 215, 400, 258, 210, 300, 260, 300]  # eyes on top of each other
    row[14] = 0.9
    g = traits.geometry(row)
    assert -90.0 <= g["yaw"] <= 90.0
    assert np.isfinite(g["yaw"])


def test_geometry_frontal_face_reads_near_zero_yaw():
    from pipeline import traits
    row = np.zeros(15, np.float32)
    row[:4] = [200, 150, 160, 190]
    row[4:14] = [250, 215, 320, 215, 285, 258, 262, 300, 308, 300]
    row[14] = 0.9
    assert abs(traits.geometry(row)["yaw"]) < 8


def test_detection_and_person_thresholds_are_distinct():
    """One number doing both jobs is what made a real face at 0.575 vanish."""
    from pipeline import traits
    assert traits.DETECT_SCORE < traits.PERSON_SCORE


def haar_row(x=40, y=30, w=120, h=120):
    """What traits._haar_rows emits: a box, zeros for the landmarks, score 0."""
    row = np.zeros(15, np.float32)
    row[:4] = [x, y, w, h]
    return row


def test_a_haar_row_has_no_pose_rather_than_a_frontal_one():
    """Its landmarks are zeros meaning "not measured". Read as points they
    gave yaw 0 and roll 0 -- a perfectly frontal face -- so registration's
    front stage accepted every Haar-only frame as looking at the lens."""
    from pipeline import camera, traits
    g = traits.geometry(haar_row())
    assert g["box"] == [40, 30, 120, 120]
    assert g["yaw"] is None and g["roll"] is None and g["pitchRatio"] is None
    assert g["landmarks"] is None
    assert not camera.pose_matches("front", g["yaw"], g["roll"],
                                   g["pitchRatio"], None)


def test_unknown_pose_is_not_flagged_as_a_bad_one():
    from pipeline import traits
    metrics = {"sharpness": 500.0, "qualityScore": 0.9, "shadowClip": 0.0,
               "highlightClip": 0.0, "dynamicRange": 200.0}
    geom = {"facePx": 200, "yaw": None, "roll": None}
    assert traits.quality_flags(metrics, geom, False) == []


def test_a_haar_row_is_embedded_from_its_box_not_aligned_to_zeros(monkeypatch):
    """alignCrop with five zero landmarks warps the image onto one point, and
    SFace embedded whatever came out. The box is what a Haar row knows."""
    from pipeline import traits

    class FakeSFace:
        aligned_with = None
        fed = None

        def alignCrop(self, image, row):
            FakeSFace.aligned_with = row
            return np.zeros((112, 112, 3), np.uint8)

        def feature(self, image):
            FakeSFace.fed = image
            return np.ones((1, 128), np.float32)

    monkeypatch.setattr(traits.facemodels, "get",
                        lambda name: FakeSFace() if name == "sface" else None)
    frame = np.zeros((240, 240, 3), np.uint8)
    frame[30:150, 40:160] = 255          # the box is white, the rest black

    vec = traits.embed(frame, haar_row())
    assert FakeSFace.aligned_with is None, "aligned by landmarks it does not have"
    assert FakeSFace.fed.shape[:2] == traits.SFACE_INPUT_SIZE
    assert FakeSFace.fed.min() == 255, "embedded more than the face box"
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_count_people_ignores_haar_rows():
    from pipeline import traits
    haar = np.zeros(15, np.float32)          # score 0
    strong = np.zeros(15, np.float32)
    strong[14] = 0.95
    assert traits.count_people([haar]) == 0
    assert traits.count_people([haar, strong]) == 1


def test_brightness_and_contrast_are_reported_but_never_gate():
    """They encode skin tone. Reported yes, decisive no."""
    from pipeline import traits
    metrics = {"sharpness": 500.0, "brightness": 20.0, "contrast": 5.0,
               "qualityScore": 0.9, "shadowClip": 0.0, "highlightClip": 0.0,
               "dynamicRange": 200.0}
    assert traits.quality_flags(metrics, None, False) == []


def test_exposure_flags_fire_on_clipping(blank_frame):
    from pipeline import traits
    crushed = {"sharpness": 500.0, "brightness": 5.0, "contrast": 1.0,
               "qualityScore": 0.9, "shadowClip": 0.99, "highlightClip": 0.0,
               "dynamicRange": 200.0}
    assert "underexposed" in traits.quality_flags(crushed, None, False)


def test_analyze_on_a_blank_frame_reports_no_face(blank_frame):
    from pipeline import traits
    t = traits.analyze(blank_frame, want_embedding=False,
                       want_demographics=True, require_detection=True)
    assert t["detected"] is False
    assert t["demographics"] is None, \
        "a softmax classifier will happily label an empty frame; it must be suppressed"


def test_landmark_metrics_are_json_serialisable(face_image):
    """Regression: one numpy float32 in this dict returned HTTP 500 for the
    whole status endpoint, and only when a face was actually present."""
    import cv2
    from pipeline import traits
    from pipeline import landmarks
    rows = traits.detect(face_image)
    if not rows:
        pytest.skip("no face detected in the sample image")
    pts = landmarks.fit(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY),
                        traits.geometry(rows[0])["box"])
    if pts is None:
        pytest.skip("landmark model unavailable")
    m = landmarks.metrics(pts)
    json.dumps(m)
    for k, v in m.items():
        assert not isinstance(v, np.generic), f"{k} leaked a numpy scalar"


def test_derived_points_sit_where_anatomy_says_they_should(face_image):
    import cv2
    from pipeline import traits
    from pipeline import landmarks
    rows = traits.detect(face_image)
    if not rows:
        pytest.skip("no face detected")
    pts = landmarks.fit(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY),
                        traits.geometry(rows[0])["box"])
    if pts is None:
        pytest.skip("landmark model unavailable")
    d = landmarks.derived_points(pts)
    assert d["forehead"][:, 1].mean() < pts[17:27][:, 1].mean(), "forehead above brows"
    eye_y = pts[36:48][:, 1].mean()
    mouth_y = pts[48:60][:, 1].mean()
    for key in ("cheekRightHigh", "cheekLeftHigh"):
        assert eye_y <= d[key][1] <= mouth_y, f"{key} outside cheek height"
    assert d["cheekRightHigh"][0] < pts[30][0] < d["cheekLeftHigh"][0], \
        "cheekbones must flank the nose"


def test_metrics_returns_none_for_bad_input():
    from pipeline import landmarks
    assert landmarks.metrics(None) is None
    assert landmarks.metrics(np.zeros((10, 2), np.float32)) is None
