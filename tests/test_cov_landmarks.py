"""pipeline/landmarks.py with a fake facemark and hand-placed point sets.

The 68-point model is not in CI. A fake facemark covers loading and fit();
the measurements run on geometric point layouts (test_pure_helpers.fake_68,
plus deliberate distortions of it), which are coordinates, not images. The
degenerate cases are the ones that divide by a distance.
"""
import json

import numpy as np
import pytest

from pipeline import landmarks
from test_pure_helpers import fake_68


class FakeFacemark:
    def __init__(self, result=None):
        self.result = result
        self.loaded = []
        self.fits = []

    def loadModel(self, path):
        self.loaded.append(path)

    def fit(self, gray, faces):
        self.fits.append(faces.copy())
        return self.result


# ---------------------------------------------------------------- loading

def test_a_missing_lbf_model_means_no_points(monkeypatch, tmp_path):
    monkeypatch.setattr(landmarks.paths, "models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(landmarks, "_model", None)
    assert landmarks.available() is False
    assert landmarks._get() is None
    assert landmarks.fit(np.zeros((50, 50), np.uint8), (0, 0, 10, 10)) is None


def test_the_lbf_model_is_loaded_once(monkeypatch, tmp_path):
    (tmp_path / "lbfmodel.yaml").write_text("stub")
    monkeypatch.setattr(landmarks.paths, "models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(landmarks, "_model", None)
    made = []

    def create():
        made.append(FakeFacemark())
        return made[-1]
    monkeypatch.setattr(landmarks.cv2.face, "createFacemarkLBF", create)
    first = landmarks._get()
    assert landmarks._get() is first and len(made) == 1
    assert first.loaded == [str(tmp_path / "lbfmodel.yaml")]


# -------------------------------------------------------------------- fit

def use(monkeypatch, facemark):
    monkeypatch.setattr(landmarks, "_get", lambda: facemark)
    return facemark


def test_fit_returns_the_first_shape_as_float32(monkeypatch):
    pts = fake_68().astype(np.float64)
    fm = use(monkeypatch, FakeFacemark((True, [pts[None, :, :]])))
    got = landmarks.fit(np.zeros((300, 300), np.uint8), (10.7, 20.2, 100.9, 90))
    assert got.dtype == np.float32 and got.shape == (68, 2)
    assert np.allclose(got, pts)
    assert fm.fits[0].dtype == np.int32
    assert fm.fits[0].tolist() == [[10, 20, 100, 90]]


@pytest.mark.parametrize("box", [(0, 0, 0, 10), (0, 0, 10, 0), (5, 5, -3, 8)])
def test_a_box_with_no_area_is_not_fitted(monkeypatch, box):
    fm = use(monkeypatch, FakeFacemark((True, [np.zeros((1, 68, 2))])))
    assert landmarks.fit(np.zeros((50, 50), np.uint8), box) is None
    assert fm.fits == []


@pytest.mark.parametrize("result", [(False, [np.zeros((1, 68, 2))]),
                                    (True, []), (False, [])])
def test_a_failed_fit_is_none(monkeypatch, result):
    use(monkeypatch, FakeFacemark(result))
    assert landmarks.fit(np.zeros((50, 50), np.uint8), (0, 0, 20, 20)) is None


# --------------------------------------------------------------- geometry

def test_an_upside_down_face_keeps_the_forehead_away_from_the_chin():
    """'Up' is perpendicular to the eye line and must point away from the
    chin; on a face rotated 180 degrees that is screen-down."""
    pts = fake_68()
    flipped = pts.copy()
    flipped[:, 1] = 300 - pts[:, 1]
    d = landmarks.derived_points(flipped)
    brow_y = flipped[17:27, 1].mean()
    chin_y = flipped[landmarks.CHIN, 1]
    assert chin_y < brow_y, "the chin is now above the brows"
    assert d["forehead"][:, 1].mean() > brow_y, "forehead beyond the brows"
    assert d["up"][1] > 0


def test_mesh_points_of_too_few_points_is_the_input_unchanged():
    few = np.zeros((10, 2), np.float32)
    assert landmarks.mesh_points(few) is few
    assert landmarks.mesh_points(None) is None


def neutral():
    """fake_68 with the mouth closed to a neutral line (its own outer lip is
    open past MAR_OPEN)."""
    p = fake_68()
    idx = landmarks.PARTS["lipOuter"]
    c = p[idx].mean(axis=0)
    p[idx, 1] = c[1] + (p[idx, 1] - c[1]) * 0.5
    return p


def tweak(**moves):
    """neutral() with named parts scaled vertically about their own centre."""
    p = neutral()
    for part, factor in moves.items():
        idx = landmarks.PARTS[part]
        c = p[idx].mean(axis=0)
        p[idx, 1] = c[1] + (p[idx, 1] - c[1]) * factor
    return p


def test_the_reference_face_raises_no_flags():
    m = landmarks.metrics(neutral())
    assert m["flags"] == []
    json.dumps(m)
    assert all(isinstance(v, float) for k, v in m.items() if k != "flags")


def test_both_eyes_closed():
    m = landmarks.metrics(tweak(eyeRight=0.05, eyeLeft=0.05))
    assert "eyes closed" in m["flags"] and "one eye closed" not in m["flags"]


def test_one_eye_closed_is_its_own_flag():
    m = landmarks.metrics(tweak(eyeLeft=0.05))
    assert "one eye closed" in m["flags"] and "eyes closed" not in m["flags"]
    # One eye shut is also an asymmetry between the eyes.
    assert "face partly obscured" in m["flags"]


def test_an_open_mouth():
    m = landmarks.metrics(tweak(lipOuter=2.5))
    assert m["flags"] == ["mouth open"]
    assert m["mouthOpen"] > landmarks.MAR_OPEN


def test_a_nose_off_centre_reads_as_partly_obscured():
    p = neutral()
    p[landmarks.NOSE_BRIDGE[0], 0] += 30
    m = landmarks.metrics(p)
    assert m["flags"] == ["face partly obscured"]
    assert m["centreOffset"] > landmarks.ASYMMETRY_LIMIT


def test_degenerate_landmarks_do_not_divide_by_zero():
    """Every point on one spot: every width and distance is zero."""
    m = landmarks.metrics(np.zeros((68, 2), np.float32))
    values = [v for k, v in m.items() if k != "flags"]
    assert all(np.isfinite(values))
    # Zero distances fall back to 1.0 rather than dividing by them.
    assert m["interocularPx"] == 1.0 and m["eyeMismatch"] == 0.0
    assert m["flags"] == ["eyes closed"]
    d = landmarks.derived_points(np.zeros((68, 2), np.float32))
    assert d["interocular"] == 1.0 and np.all(np.isfinite(d["forehead"]))


def test_collinear_landmarks_are_finite():
    p = np.zeros((68, 2), np.float32)
    p[:, 0] = np.arange(68)          # all on one horizontal line
    m = landmarks.metrics(p)
    assert all(np.isfinite([v for k, v in m.items() if k != "flags"]))


# ------------------------------------------------------------------- draw

def test_draw_skips_too_few_points():
    frame = np.zeros((100, 100, 3), np.uint8)
    landmarks.draw(frame, None)
    landmarks.draw(frame, np.zeros((5, 2), np.float32))
    assert not frame.any()


def test_draw_paints_in_the_given_colour_only():
    frame = np.zeros((300, 300, 3), np.uint8)
    landmarks.draw(frame, fake_68(), colour=(0, 200, 0))
    painted = frame.reshape(-1, 3)[frame.reshape(-1, 3).any(axis=1)]
    assert len(painted) > 500
    assert not painted[:, 0].any() and not painted[:, 2].any(), \
        "one hue, separated by weight"


def test_draw_clips_points_off_the_frame():
    """A face half out of frame: points, cheeks and mesh outside the image
    are skipped rather than wrapped or raising."""
    frame = np.zeros((120, 120, 3), np.uint8)
    pts = fake_68() + np.array([-90.0, -30.0], np.float32)
    landmarks.draw(frame, pts)
    assert frame.any()
    d = landmarks.derived_points(pts)
    assert d["cheekRightHigh"][0] < 0, "a derived point really is off frame"


def test_delaunay_needs_three_points_inside():
    pts = np.array([[5, 5], [10, 10], [500, 500], [-3, 4]], np.float32)
    assert landmarks._delaunay_edges(pts, 50, 50) == []


def test_delaunay_edges_stay_inside_the_frame():
    pts = landmarks.mesh_points(fake_68())
    for (a, b) in landmarks._delaunay_edges(pts, 160, 160):
        for x, y in (a, b):
            assert 0 <= x < 160 and 0 <= y < 160


def test_triangles_reaching_the_virtual_outer_vertices_are_dropped(monkeypatch):
    """OpenCV 4 filters these inside getTriangleList; older builds return the
    triangles that join the cloud to Subdiv2D's three virtual vertices far
    outside the rectangle, which drew long lines off the edge of the frame.
    The filter is kept for those builds, so it is checked with one."""
    class OldSubdiv:
        def __init__(self, rect):
            self.rect = rect

        def insert(self, pt):
            pass

        def getTriangleList(self):
            return np.array([[1, 1, 20, 1, 1, 20],
                             [1, 1, 20, 1, -3000, 3000]], np.float32)
    monkeypatch.setattr(landmarks.cv2, "Subdiv2D", OldSubdiv)
    pts = np.array([[1, 1], [20, 1], [1, 20]], np.float32)
    edges = landmarks._delaunay_edges(pts, 50, 50)
    assert sorted(edges) == [((1, 1), (1, 20)), ((1, 1), (20, 1)),
                             ((1, 20), (20, 1))]
