"""The helper layer: image maths, mesh geometry, gallery assembly, and the
report formatters. All pure, all previously uncovered."""
import numpy as np
import pytest


# ------------------------------------------------------------------ traits

def test_to_bgr_promotes_grayscale_and_flags_it():
    import traits
    grey = np.full((20, 20), 128, np.uint8)
    bgr, is_grey = traits.to_bgr(grey)
    assert bgr.shape == (20, 20, 3) and is_grey


def test_to_bgr_detects_colour_format_holding_grey_data():
    """Three identical channels is still grey data, and the demographic nets
    need to know that before they are trusted."""
    import traits
    faux = np.dstack([np.full((10, 10), 90, np.uint8)] * 3)
    _, is_grey = traits.to_bgr(faux)
    assert is_grey


def test_to_bgr_of_none():
    import traits
    assert traits.to_bgr(None) == (None, False)


def test_to_gray_is_idempotent():
    import traits
    g = np.full((8, 8), 100, np.uint8)
    assert traits.to_gray(g) is g
    assert traits.to_gray(np.zeros((8, 8, 3), np.uint8)).ndim == 2


def test_exposure_metrics_detect_crushed_and_blown():
    import traits
    m = traits.exposure_metrics(np.zeros((50, 50), np.uint8))
    assert m["shadowClip"] > 0.9 and m["dynamicRange"] < 5
    m2 = traits.exposure_metrics(np.full((50, 50), 255, np.uint8))
    assert m2["highlightClip"] > 0.9


def test_exposure_metrics_pass_a_well_exposed_dark_image():
    """A dark face that uses its range is correctly exposed. This is exactly
    the distinction the old absolute brightness gate could not make."""
    import traits
    ramp = np.linspace(10, 120, 2500).reshape(50, 50).astype(np.uint8)
    m = traits.exposure_metrics(ramp)
    assert m["shadowClip"] < 0.1 and m["dynamicRange"] > 80


def test_sharpness_separates_blurred_from_sharp():
    import cv2
    import traits
    noise = np.random.default_rng(0).integers(0, 255, (80, 80), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (0, 0), 5)
    assert traits.sharpness(noise) > traits.sharpness(blurred)


def test_cosine_of_none_is_none():
    import traits
    assert traits.cosine(None, np.zeros(4)) is None
    assert traits.cosine(np.zeros(4), None) is None


def test_cosine_of_identical_unit_vectors_is_one():
    import traits
    v = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    assert traits.cosine(v, v) == pytest.approx(1.0)


def test_quality_flags_report_every_geometry_problem():
    import traits
    metrics = {"sharpness": 500.0, "brightness": 120.0, "contrast": 50.0,
               "qualityScore": 0.9, "shadowClip": 0.0, "highlightClip": 0.0,
               "dynamicRange": 200.0}
    geom = {"facePx": 10, "yaw": 80.0, "roll": 70.0}
    flags = traits.quality_flags(metrics, geom, False)
    assert {"face too small", "turned away", "head tilted"} <= set(flags)


# --------------------------------------------------------------- landmarks

def fake_68():
    """A plausible 68-point face: jaw arc, brows, nose, eyes, mouth."""
    p = np.zeros((68, 2), np.float32)
    for i in range(17):
        t = np.pi * (i / 16)
        p[i] = [100 - 60 * np.cos(t), 120 + 70 * np.sin(t)]
    for i in range(17, 22):
        p[i] = [60 + (i - 17) * 8, 90]
    for i in range(22, 27):
        p[i] = [110 + (i - 22) * 8, 90]
    for i in range(27, 31):
        p[i] = [100, 100 + (i - 27) * 8]
    for i in range(31, 36):
        p[i] = [88 + (i - 31) * 6, 132]
    for i in range(36, 42):
        a = 2 * np.pi * (i - 36) / 6
        p[i] = [72 + 9 * np.cos(a), 105 + 5 * np.sin(a)]
    for i in range(42, 48):
        a = 2 * np.pi * (i - 42) / 6
        p[i] = [128 + 9 * np.cos(a), 105 + 5 * np.sin(a)]
    for i in range(48, 60):
        a = 2 * np.pi * (i - 48) / 12
        p[i] = [100 + 18 * np.cos(a), 155 + 9 * np.sin(a)]
    for i in range(60, 68):
        a = 2 * np.pi * (i - 60) / 8
        p[i] = [100 + 11 * np.cos(a), 155 + 5 * np.sin(a)]
    return p


def test_parts_splits_all_68_points():
    import landmarks
    p = landmarks.parts(fake_68())
    assert sum(len(v) for v in p.values()) == 68
    assert set(p) == set(landmarks.PARTS)


def test_parts_of_none_is_empty():
    import landmarks
    assert landmarks.parts(None) == {}


def test_derived_points_and_outline_close_the_face():
    import landmarks
    pts = fake_68()
    d = landmarks.derived_points(pts)
    assert d["forehead"][:, 1].mean() < pts[17:27][:, 1].mean()
    o = landmarks.outline(pts)
    assert len(o) == 17 + len(d["forehead"])


def test_derived_points_of_bad_input():
    import landmarks
    assert landmarks.derived_points(None) == {}
    assert landmarks.outline(None) is None


def test_mesh_points_add_the_derived_ones():
    import landmarks
    assert len(landmarks.mesh_points(fake_68())) == 68 + 4 + 10


def test_delaunay_connects_the_cloud():
    import landmarks
    edges = landmarks._delaunay_edges(landmarks.mesh_points(fake_68()), 300, 300)
    assert len(edges) > 100, "a mesh, not a scatter"


def test_delaunay_handles_too_few_points():
    import landmarks
    single = np.array([[1.0, 1.0]], np.float32)
    assert landmarks._delaunay_edges(single, 50, 50) == []


def test_draw_uses_a_single_hue():
    import landmarks
    frame = np.zeros((300, 300, 3), np.uint8)
    landmarks.draw(frame, fake_68(), (138, 201, 94))
    painted = frame.reshape(-1, 3)
    painted = painted[painted.sum(axis=1) > 0]
    assert len(painted) > 0
    ratios = painted[:, 1] / np.maximum(painted[:, 0], 1)
    assert ratios.std() < 0.6, "channels must stay on one hue ray"


def test_draw_of_bad_input_is_a_noop():
    import landmarks
    frame = np.zeros((50, 50, 3), np.uint8)
    landmarks.draw(frame, None)
    landmarks.draw(frame, np.zeros((10, 2), np.float32))
    assert frame.sum() == 0


def test_metrics_flags_closed_eyes():
    import landmarks
    pts = fake_68()
    for i in range(36, 48):
        pts[i][1] = 105
    assert "eyes closed" in landmarks.metrics(pts)["flags"]


# -------------------------------------------------------------- recognition

def test_centroid_of_nothing_is_none():
    import recognition
    assert recognition._centroid([]) is None


def test_centroid_is_unit_length():
    import recognition
    vs = [np.array([1.0, 0, 0, 0], np.float32),
          np.array([0, 1.0, 0, 0], np.float32)]
    assert np.linalg.norm(recognition._centroid(vs)) == pytest.approx(1.0)


def test_gallery_skips_users_without_embeddings(isolated_db):
    import recognition
    isolated_db.add_user("NoEmbeddings")
    assert recognition.gallery() == {}


def test_gallery_includes_a_user_with_embeddings(isolated_db):
    import recognition
    uid = isolated_db.add_user("HasOne")
    v = np.zeros(128, np.float32)
    v[0] = 1.0
    isolated_db.save_traits(dict(
        path="p.jpg", user_id=uid, mtime=1.0, sharpness=1.0, brightness=1.0,
        contrast=1.0, quality=0.5, face_px=10, yaw=0.0, roll=0.0, detected=1,
        flags="", embedding=v.tobytes(), age_label=None, age_conf=None,
        gender_label=None, gender_conf=None))
    g = recognition.gallery()
    assert len(g) == 1 and g[uid]["samples"] == 1


def test_embed_image_on_a_blank_frame():
    import recognition
    vec, geom = recognition.embed_image(np.zeros((100, 100, 3), np.uint8))
    assert vec is None and geom is None


# ------------------------------------------------------------ analyze_faces

def test_bar_renders_proportionally():
    import analyze_faces
    assert analyze_faces.bar(0, width=10).count("#") == 0
    assert analyze_faces.bar(100, width=10).count("#") == 10
    assert analyze_faces.bar(50, width=10).count("#") == 5


def test_bar_clamps_out_of_range():
    import analyze_faces
    assert analyze_faces.bar(-50, width=8).count("#") == 0
    assert analyze_faces.bar(500, width=8).count("#") == 8


def test_print_sweep_handles_an_unavailable_block(capsys):
    import analyze_faces
    analyze_faces.print_sweep("X", {"available": False, "reason": "too few"})
    assert "too few" in capsys.readouterr().out


def test_print_quality_on_an_empty_report(capsys):
    import analyze_faces
    analyze_faces.print_quality({"users": []})
    assert "register" in capsys.readouterr().out.lower()
