"""pipeline/liveness.py with an injected network.

CI has no MiniFASNet weights, so score() -- the crop, the blob, the softmax
-- never ran there. A fake net (setInput/forward returning fixed logits)
covers all of it: what crop reaches the network, how the box is clipped at
the frame edges, how the output is read, and every verdict the vote reaches.
The frames are flat or banded arrays; nothing here is a face.
"""
import numpy as np
import pytest

from pipeline import liveness


class FakeNet:
    def __init__(self, logits):
        self.logits = np.asarray(logits, np.float32)
        self.inputs = []

    def setInput(self, blob):
        self.inputs.append(blob)

    def forward(self):
        return self.logits


@pytest.fixture
def crops(monkeypatch):
    """Record every crop handed to blobFromImage, then build the real blob."""
    seen = []
    real = liveness.cv2.dnn.blobFromImage

    def spy(img, *args, **kwargs):
        seen.append(img.copy())
        return real(img, *args, **kwargs)
    monkeypatch.setattr(liveness.cv2.dnn, "blobFromImage", spy)
    return seen


def with_net(monkeypatch, net):
    monkeypatch.setattr(liveness, "_get", lambda: net)
    return net


def frame(h=240, w=320):
    return np.full((h, w, 3), 100, np.uint8)


# ------------------------------------------------------------- loading

def test_without_the_weights_there_is_no_net_and_no_score(monkeypatch, tmp_path):
    monkeypatch.setattr(liveness.paths, "models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(liveness, "_net", None)
    assert liveness.available() is False
    assert liveness._get() is None
    assert liveness.score(frame(), (10, 10, 50, 50)) is None


def test_the_net_is_loaded_once_and_cached(monkeypatch, tmp_path):
    (tmp_path / "minifasnet_v2.onnx").write_bytes(b"stub")
    monkeypatch.setattr(liveness.paths, "models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(liveness, "_net", None)
    loads = []

    def read_net(path):
        loads.append(path)
        return FakeNet([0, 1, 0])
    monkeypatch.setattr(liveness.cv2.dnn, "readNet", read_net)
    first = liveness._get()
    assert liveness._get() is first
    assert loads == [str(tmp_path / "minifasnet_v2.onnx")]


def test_no_frame_means_no_score(monkeypatch):
    net = with_net(monkeypatch, FakeNet([0, 5, 0]))
    assert liveness.score(None, (0, 0, 10, 10)) is None
    assert net.inputs == []


# --------------------------------------------------- crop and preprocessing

def test_the_crop_is_the_face_with_context_around_it(monkeypatch, crops):
    net = with_net(monkeypatch, FakeNet([0, 0, 0]))
    liveness.score(frame(), (140, 100, 40, 40))
    # 40 px * SCALE 2.7 = 108 px square, centred on the box centre (160, 120).
    assert crops[0].shape == (108, 108, 3)
    blob = net.inputs[0]
    assert blob.shape == (1, 3) + liveness.INPUT[::-1]
    assert float(blob.max()) == 100.0, "no scaling and no mean subtraction"


def test_a_crop_is_clipped_at_the_frame_edges(monkeypatch, crops):
    with_net(monkeypatch, FakeNet([0, 0, 0]))
    img = frame()
    img[:, :20] = 7                     # a band down the left edge
    liveness.score(img, (0, 0, 40, 40))
    crop = crops[0]
    # Centre (20, 20), half-width 54: clipped to [0, 74) on both axes rather
    # than wrapping round to the far edge with a negative index.
    assert crop.shape == (74, 74, 3)
    assert (crop[:, :20] == 7).all() and (crop[:, 20:] == 100).all()

    liveness.score(img, (300, 220, 40, 40))
    assert crops[1].shape[:2] == (240 - 186, 320 - 266)


def test_the_crop_uses_the_larger_side_of_the_box(monkeypatch, crops):
    with_net(monkeypatch, FakeNet([0, 0, 0]))
    liveness.score(frame(400, 400), (150, 150, 20, 60))
    assert crops[0].shape[:2] == (162, 162)


def test_a_box_wholly_outside_the_frame_scores_nothing(monkeypatch, crops):
    net = with_net(monkeypatch, FakeNet([0, 5, 0]))
    assert liveness.score(frame(), (1000, 1000, 30, 30)) is None
    assert crops == [] and net.inputs == []


# -------------------------------------------------------- output parsing

@pytest.mark.parametrize("logits, expected", [
    ([[0.0, 0.0, 0.0]], 1 / 3),
    ([[0.0, np.log(3.0), 0.0]], 0.6),
    ([[[[1000.0]], [[1000.0]], [[998.0]]]], np.e ** 2 / (2 * np.e ** 2 + 1)),
    ([10.0, -10.0, 10.0], 1 / (2 * np.exp(20.0) + 1)),
])
def test_the_output_is_softmaxed_and_class_one_is_live(monkeypatch, logits,
                                                       expected):
    with_net(monkeypatch, FakeNet(logits))
    got = liveness.score(frame(), (100, 80, 60, 60))
    assert isinstance(got, float)
    assert got == pytest.approx(expected, rel=1e-5)
    assert 0.0 <= got <= 1.0


def test_large_logits_do_not_overflow(monkeypatch):
    """The max is subtracted before exp; without it 1000 is inf / inf."""
    with_net(monkeypatch, FakeNet([[1000.0, 1001.0, 999.0]]))
    got = liveness.score(frame(), (100, 80, 60, 60))
    assert np.isfinite(got) and got > 0.5


# ------------------------------------------------------------- verdicts

def vote_on(scores, **kw):
    v = liveness.LivenessVote(**kw)
    for s in scores:
        v.push(s)
    return v


def test_scores_from_the_net_drive_every_verdict(monkeypatch):
    live = FakeNet([0.0, 3.0, 0.0])     # ~0.91, over LIVE_THRESHOLD
    spoof = FakeNet([3.0, 0.0, 0.0])    # ~0.05, under it
    v = liveness.LivenessVote()
    for net in (live, live, live):
        with_net(monkeypatch, net)
        v.push(liveness.score(frame(), (100, 80, 60, 60)))
    assert v.verdict() == "unknown", "three frames is not yet evidence"
    with_net(monkeypatch, live)
    v.push(liveness.score(frame(), (100, 80, 60, 60)))
    assert v.verdict() == "live"

    v.reset()
    for _ in range(liveness.VOTE_WINDOW):
        with_net(monkeypatch, spoof)
        v.push(liveness.score(frame(), (100, 80, 60, 60)))
    assert v.verdict() == "spoof"


def test_the_threshold_itself_passes():
    v = vote_on([liveness.LIVE_THRESHOLD] * 4)
    assert v.verdict() == "live"
    v = vote_on([liveness.LIVE_THRESHOLD - 1e-6] * 4)
    assert v.verdict() == "spoof"


def test_mixed_evidence_needs_the_required_count():
    v = vote_on([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    assert v.samples == 7 and v.verdict() == "spoof"
    for _ in range(3):
        v.push(0.9)                      # each new 0.9 pushes an old 0.9 out
        assert v.samples == 7 and v.verdict() == "spoof"
    v.push(0.9)                          # now a 0.1 leaves: four of seven pass
    assert v.verdict() == "live"


def test_mean_and_missing_scores():
    v = liveness.LivenessVote()
    assert v.mean is None
    v.push(None)
    assert v.samples == 0, "an unavailable score is not a vote"
    v.push(0.2)
    v.push(0.4)
    assert v.mean == pytest.approx(0.3)


def test_following_the_same_person_keeps_the_vote():
    v = vote_on([0.9] * 4)
    v.follow(5)
    assert v.samples == 0, "the first person named starts the vote"
    for _ in range(4):
        v.push(0.9)
    v.follow(5)
    v.follow(None)
    assert v.samples == 4 and v.verdict() == "live"
    v.follow(6)
    assert v.samples == 0 and v.verdict() == "unknown"
