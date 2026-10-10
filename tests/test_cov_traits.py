"""pipeline/traits.py with every network mocked.

facemodels.get is replaced per test, so the result is the same with or
without weights on disk. Frames are flat or banded arrays, and where a
"detection" is needed the detector itself is the fake: these tests are
about what traits.py does with a detection, never about finding a face.
"""
import numpy as np
import pytest

from pipeline import traits


def models(monkeypatch, **nets):
    """facemodels.get returns these by name, and None for anything else."""
    monkeypatch.setattr(traits.facemodels, "get", lambda name: nets.get(name))


def yunet_row(x, y, w, h, score=0.9):
    row = np.zeros(15, np.float32)
    row[:4] = [x, y, w, h]
    # Eyes level, nose centred: a frontal pose by construction.
    cx = x + w / 2
    row[4:14] = [cx - w / 5, y + h / 3, cx + w / 5, y + h / 3,
                 cx, y + h / 2, cx - w / 6, y + 0.7 * h, cx + w / 6, y + 0.7 * h]
    row[14] = score
    return row


# ------------------------------------------------------------------ to_bgr

def test_to_bgr_accepts_every_layout():
    assert traits.to_bgr(None) == (None, False)
    grey2d, g = traits.to_bgr(np.full((4, 4), 9, np.uint8))
    assert grey2d.shape == (4, 4, 3) and g is True
    one_channel, g = traits.to_bgr(np.full((4, 4, 1), 9, np.uint8))
    assert one_channel.shape == (4, 4, 3) and g is True
    bgra = np.zeros((4, 4, 4), np.uint8)
    bgra[..., 0] = 200
    bgra[..., 3] = 255
    out, g = traits.to_bgr(bgra)
    assert out.shape == (4, 4, 3) and g is False and out[0, 0, 0] == 200
    flat_colour, g = traits.to_bgr(np.full((4, 4, 3), 50, np.uint8))
    assert g is True, "identical channels are grey data in a colour container"


# --------------------------------------------------------------- detection

class FakeCascade:
    def __init__(self, boxes, empty=False):
        self.boxes, self._empty, self.calls = boxes, empty, []

    def empty(self):
        return self._empty

    def detectMultiScale(self, gray, *args, **kwargs):
        self.calls.append(gray.ndim)
        return self.boxes


def test_haar_rows_are_boxes_with_no_landmarks_and_no_score(monkeypatch):
    cascade = FakeCascade([(10, 20, 30, 40), (1, 2, 3, 4)])
    monkeypatch.setattr(traits, "_cascade", cascade)
    rows = traits._haar_rows(np.zeros((60, 60, 3), np.uint8))
    assert cascade.calls == [2], "the cascade runs on greyscale"
    assert [r[:4].tolist() for r in rows] == [[10, 20, 30, 40], [1, 2, 3, 4]]
    assert all(not r[4:].any() for r in rows)
    assert traits.count_people(rows) == 0


def test_an_empty_cascade_finds_nothing(monkeypatch):
    monkeypatch.setattr(traits, "_cascade", FakeCascade([(1, 1, 5, 5)],
                                                        empty=True))
    assert traits._haar_rows(np.zeros((20, 20, 3), np.uint8)) == []


def test_the_cascade_is_built_once(monkeypatch):
    made = []

    def build(path):
        made.append(path)
        return FakeCascade([])
    monkeypatch.setattr(traits, "_cascade", None)
    monkeypatch.setattr(traits.cv2, "CascadeClassifier", build)
    traits._haar_rows(np.zeros((20, 20, 3), np.uint8))
    traits._haar_rows(np.zeros((20, 20, 3), np.uint8))
    assert len(made) == 1 and made[0].endswith(
        "haarcascade_frontalface_default.xml")


class FakeYuNet:
    def __init__(self, faces):
        self.faces, self.thresholds, self.sizes = faces, [], []

    def setScoreThreshold(self, t):
        self.thresholds.append(t)

    def setInputSize(self, size):
        self.sizes.append(size)

    def detect(self, img):
        return 1, self.faces


def test_detect_without_yunet_falls_back_to_haar(monkeypatch):
    models(monkeypatch)
    monkeypatch.setattr(traits, "_haar_rows", lambda bgr: ["haar"])
    assert traits.detect(np.zeros((10, 10, 3), np.uint8)) == ["haar"]


@pytest.mark.parametrize("faces", [None, np.zeros((0, 15), np.float32)])
def test_detect_falls_back_to_haar_when_yunet_finds_nobody(monkeypatch, faces):
    net = FakeYuNet(faces)
    models(monkeypatch, yunet=net)
    monkeypatch.setattr(traits, "_haar_rows", lambda bgr: ["haar"])
    assert traits.detect(np.zeros((30, 40, 3), np.uint8)) == ["haar"]
    assert net.sizes == [(40, 30)], "input size is (width, height)"
    assert net.thresholds == [traits.DETECT_SCORE]


def test_detect_sorts_by_score_and_honours_an_explicit_threshold(monkeypatch):
    weak, strong = yunet_row(0, 0, 10, 10, 0.4), yunet_row(5, 5, 10, 10, 0.95)
    net = FakeYuNet(np.vstack([weak, strong]))
    models(monkeypatch, yunet=net)
    rows = traits.detect(np.zeros((30, 40, 3), np.uint8), score=0.8)
    assert [float(r[14]) for r in rows] == pytest.approx([0.95, 0.4])
    assert net.thresholds == [0.8]


# ----------------------------------------------------------------- quality

class FakeNet:
    def __init__(self, out):
        self.out, self.blobs = np.asarray(out, np.float32), []

    def setInput(self, blob):
        self.blobs.append(blob)

    def forward(self):
        return self.out


def test_learned_quality_without_ediffiqa_is_none(monkeypatch):
    models(monkeypatch)
    assert traits.learned_quality(np.zeros((20, 20, 3), np.uint8)) is None


def test_learned_quality_normalises_the_blob_and_reads_one_number(monkeypatch):
    net = FakeNet([[0.73456]])
    models(monkeypatch, ediffiqa=net)
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, 10:] = 255
    assert traits.learned_quality(img) == 0.7346
    blob = net.blobs[0]
    assert blob.shape == (1, 3) + traits.SFACE_INPUT_SIZE
    assert blob.min() == pytest.approx(-1.0) and blob.max() == pytest.approx(1.0)


# --------------------------------------------------------------- embedding

class FakeSFace:
    def __init__(self, feature):
        self._feature = np.asarray(feature, np.float32)
        self.aligned = []

    def alignCrop(self, img, row):
        self.aligned.append(row)
        return np.zeros((112, 112, 3), np.uint8)

    def feature(self, img):
        return self._feature


def test_embed_without_sface_is_none(monkeypatch):
    models(monkeypatch)
    assert traits.embed(np.zeros((20, 20, 3), np.uint8)) is None


def test_embed_aligns_by_landmarks_when_the_row_has_them(monkeypatch):
    net = FakeSFace([[3.0, 4.0]])
    models(monkeypatch, sface=net)
    row = yunet_row(10, 10, 50, 50)
    vec = traits.embed(np.zeros((100, 100, 3), np.uint8), row)
    assert len(net.aligned) == 1 and net.aligned[0] is row
    assert vec.tolist() == pytest.approx([0.6, 0.8])


def test_embed_of_a_zero_feature_is_returned_unnormalised(monkeypatch):
    models(monkeypatch, sface=FakeSFace([[0.0, 0.0]]))
    vec = traits.embed(np.zeros((30, 30, 3), np.uint8))
    assert vec.tolist() == [0.0, 0.0] and np.all(np.isfinite(vec))


def test_box_crop_clips_and_falls_back_to_the_whole_image():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert traits._box_crop(img, None) is img
    row = np.zeros(15, np.float32)
    row[:4] = [-3, -2, 6, 5]
    assert traits._box_crop(img, row).shape == (3, 3)
    row[:4] = [50, 50, 5, 5]
    assert traits._box_crop(img, row) is img


def test_cosine_of_a_missing_vector_is_none():
    assert traits.cosine(None, np.ones(2)) is None
    assert traits.cosine(np.ones(2), None) is None


# ------------------------------------------------------------ demographics

AGE_PROBS = [0.0, 0.0, 0.0, 0.1, 0.4, 0.35, 0.15, 0.0]


def test_demographics_with_neither_net_is_none(monkeypatch):
    models(monkeypatch)
    assert traits.demographics(np.zeros((20, 20, 3), np.uint8)) is None


def test_demographics_with_age_only(monkeypatch):
    models(monkeypatch, age=FakeNet([AGE_PROBS]))
    d = traits.demographics(np.zeros((20, 20, 3), np.uint8))
    assert "gender" not in d and "warning" not in d
    age = d["age"]
    assert age["label"] == "25-32" and age["runnerUp"] == "38-43"
    assert age["uncertain"] is True, "0.4 is under half"
    assert age["caveat"] == traits.AGE_CAVEAT
    # 0.1*17.5 + 0.4*28.5 + 0.35*40.5 + 0.15*50.5 = 34.9
    assert age["estimate"]["years"] == pytest.approx(34.9)
    assert age["estimate"]["halfWidth"] == traits.calibration.AGE_BAND_YEARS


def test_demographics_with_gender_only_from_a_grey_crop(monkeypatch):
    models(monkeypatch, gender=FakeNet([[0.2, 0.8]]))
    d = traits.demographics(np.zeros((20, 20, 3), np.uint8),
                            grayscale_source=True)
    assert "age" not in d
    assert d["gender"]["label"] == "female"
    assert d["gender"]["uncertain"] is False
    assert d["grayscaleSource"] is True and "greyscale" in d["warning"]


def test_an_all_zero_age_distribution_does_not_divide_by_zero():
    est = traits._age_point_estimate({b: 0.0 for b in traits.AGE_BUCKETS})
    assert est["years"] == 0.0 and est["range"][0] == 0


# ------------------------------------------------------------------ analyze

def detector(monkeypatch, rows):
    monkeypatch.setattr(traits, "detect", lambda bgr: list(rows))


def test_analyze_of_nothing_is_none():
    assert traits.analyze(None) is None


def test_analyze_crops_a_detection_with_padding_clipped_to_the_frame(
        monkeypatch):
    seen = []
    real = traits.quality_metrics

    def spy(crop):
        seen.append(crop.shape)
        return real(crop)
    monkeypatch.setattr(traits, "quality_metrics", spy)
    models(monkeypatch, sface=FakeSFace([[1.0, 0.0]]),
           gender=FakeNet([[0.9, 0.1]]))
    detector(monkeypatch, [yunet_row(0, 0, 100, 100, 0.95),
                           yunet_row(150, 0, 40, 40, 0.8)])
    frame = np.full((200, 200, 3), 60, np.uint8)
    frame[::2] = 200                     # textured, nothing clipped
    t = traits.analyze(frame)
    # pad = 12; the box starts at the corner, so only the far sides grow.
    assert seen == [(112, 112, 3)]
    assert t["detected"] is True and t["faces"] == 2
    assert t["facePx"] == 100 and t["geometry"]["box"] == [0, 0, 100, 100]
    assert t["hasEmbedding"] is True
    assert t["demographics"]["gender"]["label"] == "male"
    assert t["usable"] is True and t["flags"] == []


def test_a_weak_detection_still_counts_as_one_face(monkeypatch):
    models(monkeypatch)
    detector(monkeypatch, [yunet_row(10, 10, 100, 100, 0.4)])
    t = traits.analyze(np.full((150, 150, 3), 128, np.uint8),
                       want_embedding=False, want_demographics=False)
    assert t["faces"] == 1 and "demographics" not in t
    assert "embedding" not in t


def test_a_box_outside_the_frame_measures_the_whole_image(monkeypatch):
    seen = []
    real = traits.quality_metrics
    monkeypatch.setattr(traits, "quality_metrics",
                        lambda crop: seen.append(crop.shape) or real(crop))
    models(monkeypatch)
    detector(monkeypatch, [yunet_row(500, 500, 40, 40)])
    t = traits.analyze(np.full((60, 80, 3), 128, np.uint8),
                       want_embedding=False, want_demographics=False)
    assert seen == [(60, 80, 3)]
    assert t["detected"] is True and t["facePx"] == 40


def test_no_detection_skips_demographics_when_detection_is_required(
        monkeypatch):
    models(monkeypatch, age=FakeNet([AGE_PROBS]))
    detector(monkeypatch, [])
    t = traits.analyze(np.full((60, 80), 128, np.uint8),
                       require_detection=True)
    assert t["detected"] is False and t["faces"] == 0
    assert t["facePx"] == 60 and t["grayscaleSource"] is True
    assert t["demographics"] is None
    assert t["demographicsSkipped"] == "no face detected"
    assert t["embedding"] is None and t["hasEmbedding"] is False


def test_no_detection_still_reads_demographics_when_not_required(monkeypatch):
    """A stored crop that will not re-detect is still a face."""
    models(monkeypatch, age=FakeNet([AGE_PROBS]))
    detector(monkeypatch, [])
    t = traits.analyze(np.full((60, 80), 128, np.uint8))
    assert t["demographics"]["age"]["label"] == "25-32"
    assert "demographicsSkipped" not in t


def test_every_photographic_flag_on_its_own():
    clean = {"sharpness": 500.0, "qualityScore": 0.9, "shadowClip": 0.0,
             "highlightClip": 0.0, "dynamicRange": 200.0}
    assert traits.quality_flags(clean, None, False) == []
    for change, flag in (({"sharpness": 1.0}, "blurry"),
                         ({"qualityScore": 0.1}, "low quality"),
                         ({"shadowClip": 0.9}, "underexposed"),
                         ({"highlightClip": 0.9}, "blown highlights"),
                         ({"dynamicRange": 3.0}, "no tonal range")):
        assert traits.quality_flags({**clean, **change}, None, False) == [flag]
    # A metric that was not measured is not a failure.
    assert traits.quality_flags({"sharpness": 500.0}, None, False) == []
