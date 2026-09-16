"""Choosing sample frames out of a video, without a face to point a camera at.

The three tests in conftest's `face_image` fixture want a real photograph.
Nothing wrote the files they look for, so they skipped permanently rather
than conditionally -- `sampleframes` is what writes them now.

These tests drive the real pipeline over a real video file, written with
cv2.VideoWriter so the decode path is exercised rather than mocked. The
detector is replaced where a face is needed, for the same reason the camera
tests replace it: a synthesised face is not one YuNet would accept, and
pretending otherwise would be the dishonest half of this.

What is still not covered is whether a real face in real footage survives
the filter. That needs footage, which is the point of the tool.
"""
import collections
import os

import cv2
import numpy as np
import pytest

import corpus_paths
import sampleframes
import traits as facetraits


# The same synthetic YuNet row the camera tests use: a frontal face at
# (200, 150), 200px square, eyes level and 100px apart.
#
# A numpy array, not a list. `geometry` accepts either, which is why the
# camera tests could use a list -- but `embed` hands the row to
# cv2.alignCrop, which takes a Mat and rejects a list outright. The real
# detector returns arrays, so the fake should too.
ROW = np.array([200.0, 150.0, 200.0, 200.0,
                250.0, 210.0, 350.0, 210.0, 300.0, 260.0,
                265.0, 310.0, 335.0, 310.0, 0.99], dtype=np.float32)


def row(**changes):
    """A fresh copy of ROW, optionally with a field moved."""
    out = ROW.copy()
    for index, value in changes.items():
        out[int(index[1:])] = value
    return out


def photo(shift=0, crisp=False):
    """A frame that passes the quality gate.

    `shift` moves the features so two frames are not the same shot. `crisp`
    adds fine detail: the plain frame sits at sharpness 9.3 against a
    MIN_SHARPNESS of 6.0, so a blurred copy falls below the gate and gets
    rejected rather than ranked -- to compare two *usable* frames the
    comparison has to go upwards.
    """
    ramp = np.tile(np.linspace(20, 235, 640, dtype=np.uint8), (480, 1))
    frame = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    for cx, cy, r in ((260, 220, 26), (340, 220, 26),
                      (300, 270, 18), (300, 310, 34)):
        cv2.circle(frame, (cx + shift, cy), r, (40, 45, 60), -1)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    if crisp:
        for yy in range(160, 340, 6):
            cv2.line(frame, (210, yy), (390, yy), (210, 215, 220), 1)
    return frame


@pytest.fixture
def one_face(monkeypatch):
    """One confident detection in every frame."""
    monkeypatch.setattr(facetraits, "detect", lambda frame: [ROW.copy()])


@pytest.fixture
def clip(tmp_path):
    """Write a short real video file and return its path."""
    def make(frames, name="clip.avi"):
        path = str(tmp_path / name)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"),
                                 10.0, (640, 480))
        assert writer.isOpened(), "no MJPG encoder available"
        for frame in frames:
            writer.write(frame)
        writer.release()
        assert os.path.getsize(path) > 0
        return path
    return make


# ------------------------------------------------------------ reading input


def test_a_video_is_read_frame_by_frame(clip):
    path = clip([photo(i) for i in range(10)])
    got = list(sampleframes.frames(path, stride=1))
    assert len(got) == 10, f"decoded {len(got)} of 10"
    assert got[0].shape == (480, 640, 3)


def test_the_stride_thins_the_stream(clip):
    """Thirty near-identical frames a second, each costing a full quality
    read, is the thing the stride exists to avoid."""
    path = clip([photo(i) for i in range(10)])
    assert len(list(sampleframes.frames(path, stride=5))) == 2


def test_a_stride_of_zero_does_not_divide_by_zero(clip):
    path = clip([photo(0)])
    assert len(list(sampleframes.frames(path, stride=0))) == 1


def test_a_folder_of_images_reads_in_name_order(tmp_path):
    for name in ("b.png", "a.png", "c.jpg"):
        cv2.imwrite(str(tmp_path / name), photo())
    got = list(sampleframes.frames(str(tmp_path)))
    assert len(got) == 3


def test_a_folder_ignores_what_is_not_an_image(tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), photo())
    (tmp_path / "notes.txt").write_text("not an image")
    assert len(list(sampleframes.frames(str(tmp_path)))) == 1


def test_a_missing_source_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(sampleframes.frames(str(tmp_path / "nope.mp4")))


def test_a_file_that_is_not_a_video_says_so(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        list(sampleframes.frames(str(bad)))


# -------------------------------------------------------------- the filter


def test_a_frame_with_no_face_is_rejected(monkeypatch):
    monkeypatch.setattr(facetraits, "detect", lambda frame: [])
    cand, reason = sampleframes.assess(photo())
    assert cand is None and reason == "no face"


def test_a_frame_with_two_people_is_rejected(monkeypatch):
    """A sample frame with a bystander in it tests the wrong thing."""
    second = ROW.copy()
    second[0] = 20.0
    monkeypatch.setattr(facetraits, "detect",
                        lambda frame: [ROW.copy(), second])
    cand, reason = sampleframes.assess(photo())
    assert cand is None and reason == "more than one person"


def test_a_weak_background_detection_does_not_count_as_a_person(monkeypatch):
    """count_people sets a far higher bar than "there is a face here", so a
    low-confidence background hit must not disqualify the frame."""
    faint = ROW.copy()
    faint[14] = 0.05
    monkeypatch.setattr(facetraits, "detect",
                        lambda frame: [ROW.copy(), faint])
    cand, reason = sampleframes.assess(photo())
    assert cand is not None, reason


def test_a_badly_photographed_frame_is_rejected_with_its_reasons(one_face):
    flat = np.full((480, 640, 3), 90, np.uint8)
    cand, reason = sampleframes.assess(flat)
    assert cand is None
    assert "blurry" in reason, reason


def test_pose_is_filtered_here_unlike_registration(monkeypatch):
    """Registration wants off-axis frames -- CAPTURE_PLAN asks for them. A
    sample frame should be a good frontal portrait, so the pose flags that
    camera.weak_photograph drops do apply here."""
    turned = ROW.copy()
    turned[8] = 360.0                     # nose well off the eye midline
    monkeypatch.setattr(facetraits, "detect", lambda frame: [turned])
    cand, reason = sampleframes.assess(photo())
    assert cand is None
    assert "turned away" in reason, reason


def test_a_face_off_the_edge_is_rejected(monkeypatch):
    monkeypatch.setattr(facetraits, "detect", lambda frame: [ROW.copy()])
    tiny = np.full((40, 40, 3), 90, np.uint8)
    cand, reason = sampleframes.assess(tiny)
    assert cand is None
    assert reason == "face outside the frame"


def test_a_good_frame_is_scored(one_face):
    cand, reason = sampleframes.assess(photo())
    assert cand is not None, reason
    assert cand.score > 0


def test_sharpness_stands_in_when_the_quality_model_is_absent(one_face,
                                                              monkeypatch):
    """Worse ordering, but it still prefers the crisper frame to the
    blurrier one rather than giving up on ranking."""
    real = facetraits.quality_metrics

    def without_model(crop):
        metrics = dict(real(crop))
        metrics["qualityScore"] = None
        return metrics

    monkeypatch.setattr(facetraits, "quality_metrics", without_model)
    crisper, why_a = sampleframes.assess(photo(crisp=True))
    softer, why_b = sampleframes.assess(photo())
    assert crisper is not None, why_a
    assert softer is not None, why_b
    assert crisper.score > softer.score


# -------------------------------------------------------------- the choice


def test_the_best_scoring_frames_are_kept(one_face, monkeypatch):
    monkeypatch.setattr(facetraits, "embed", lambda frame, row: None)
    low = sampleframes.Candidate(photo(), 0.2, ROW.copy())
    high = sampleframes.Candidate(photo(4), 0.9, ROW.copy())
    mid = sampleframes.Candidate(photo(8), 0.5, ROW.copy())
    chosen = sampleframes.choose([low, high, mid], keep=2)
    assert [c.score for c in chosen] == [0.9, 0.5]


def test_nothing_is_chosen_from_nothing():
    assert sampleframes.choose([], keep=3) == []


def test_repeats_of_the_same_shot_are_skipped(monkeypatch):
    """Three views of one instant test the detector once, not three times."""
    same = np.zeros(128, np.float32)
    same[0] = 1.0
    monkeypatch.setattr(facetraits, "embed", lambda frame, row: same)
    monkeypatch.setattr(facetraits, "cosine", lambda a, b: 1.0)
    cands = [sampleframes.Candidate(photo(i), 0.9 - i * 0.1, ROW.copy())
             for i in range(3)]
    assert len(sampleframes.choose(cands, keep=3)) == 1


def test_distinct_shots_are_all_kept(monkeypatch):
    monkeypatch.setattr(facetraits, "embed",
                        lambda frame, row: np.zeros(128, np.float32))
    monkeypatch.setattr(facetraits, "cosine", lambda a, b: 0.1)
    cands = [sampleframes.Candidate(photo(i), 0.9 - i * 0.1, ROW.copy())
             for i in range(3)]
    assert len(sampleframes.choose(cands, keep=3)) == 3


def test_no_embedding_model_means_no_deduplication(monkeypatch):
    """Without SFace there is nothing to compare, so keep the best by score
    rather than silently returning one frame."""
    monkeypatch.setattr(facetraits, "embed", lambda frame, row: None)
    cands = [sampleframes.Candidate(photo(i), 0.9 - i * 0.1, ROW.copy())
             for i in range(3)]
    assert len(sampleframes.choose(cands, keep=3)) == 3


# --------------------------------------------------------------- writing


def test_the_frames_are_written_under_the_names_the_tests_look_for(tmp_path,
                                                                   one_face):
    cands = [sampleframes.Candidate(photo(i), 0.9, ROW.copy()) for i in range(3)]
    written = sampleframes.write(cands, directory=str(tmp_path))
    assert [os.path.basename(p) for p in written] == list(sampleframes.NAMES)
    for path in written:
        assert cv2.imread(path) is not None, f"{path} did not decode"


def test_the_directory_is_created_if_absent(tmp_path, one_face):
    target = tmp_path / "frames" / "deeper"
    written = sampleframes.write(
        [sampleframes.Candidate(photo(), 0.9, ROW.copy())],
        directory=str(target))
    assert written and os.path.isdir(str(target))


def test_fewer_frames_than_names_writes_fewer_files(tmp_path, one_face):
    written = sampleframes.write(
        [sampleframes.Candidate(photo(), 0.9, ROW.copy())],
        directory=str(tmp_path))
    assert [os.path.basename(p) for p in written] == ["best.png"]


# ------------------------------------------------------------- end to end


def test_a_clip_becomes_sample_frames(clip, tmp_path, one_face, monkeypatch):
    monkeypatch.setattr(facetraits, "embed", lambda frame, row: None)
    path = clip([photo(i * 3) for i in range(6)])
    out = tmp_path / "out"

    written, rejected = sampleframes.build(path, stride=1,
                                           directory=str(out))

    assert len(written) == len(sampleframes.NAMES), rejected
    assert rejected == {}, rejected
    # And the fixture that has been skipping can now find one.
    assert os.path.exists(str(out / "best.png"))


def test_a_clip_of_nothing_writes_nothing_and_says_why(clip, tmp_path,
                                                       monkeypatch):
    monkeypatch.setattr(facetraits, "detect", lambda frame: [])
    path = clip([photo() for _ in range(3)])
    written, rejected = sampleframes.build(path, stride=1,
                                           directory=str(tmp_path / "out"))
    assert written == []
    assert rejected == {"no face": 3}


def test_the_summary_names_the_files_and_the_reasons():
    lines = "\n".join(sampleframes.summarise(
        [os.path.join("somewhere", "best.png")],
        collections.Counter({"no face": 4, "blurry": 1})))
    assert "best.png" in lines
    assert "no face" in lines and "4" in lines
    assert "blurry" in lines


def test_the_summary_is_honest_when_nothing_was_kept():
    lines = "\n".join(sampleframes.summarise([], collections.Counter()))
    assert "no frame was good enough" in lines


def test_the_default_target_is_its_own_directory():
    """Not the bare temporary directory, which is shared with every other
    program on the machine."""
    assert (os.path.normpath(corpus_paths.sample_frame_dir())
            != os.path.normpath(corpus_paths.corpora_dir()))
    assert corpus_paths.sample_frame_dir().startswith(
        corpus_paths.corpora_dir())


def test_scan_separates_the_usable_from_the_rejected(clip, one_face):
    """What --dry-run reports. It had its own copy of this loop, which is
    one more place for the two to disagree about what counts as usable."""
    path = clip([photo(i * 3) for i in range(4)])
    candidates, rejected = sampleframes.scan(path, stride=1)
    assert len(candidates) == 4, rejected
    assert rejected == {}


def test_scan_counts_each_reason(clip, monkeypatch):
    monkeypatch.setattr(facetraits, "detect", lambda frame: [])
    path = clip([photo() for _ in range(3)])
    candidates, rejected = sampleframes.scan(path, stride=1)
    assert candidates == []
    assert rejected == {"no face": 3}


def test_a_nested_folder_is_walked(tmp_path):
    """The corpora this is meant for are one folder per identity -- LFW is
    lfw/Person_Name/Person_Name_0001.jpg -- so a top-level glob found
    nothing in exactly the advertised case."""
    for person in ("Ada_Lovelace", "Grace_Hopper"):
        folder = tmp_path / "lfw" / person
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / f"{person}_0001.jpg"), photo())
    assert len(list(sampleframes.frames(str(tmp_path / "lfw")))) == 2


def test_upper_case_extensions_are_found(tmp_path):
    """Cameras write .JPG."""
    cv2.imwrite(str(tmp_path / "shot.JPG"), photo())
    assert len(list(sampleframes.frames(str(tmp_path)))) == 1


def test_a_folder_is_not_thinned_by_the_stride(tmp_path):
    """The stride is for video. A folder is already a selection."""
    for i in range(4):
        cv2.imwrite(str(tmp_path / f"{i}.png"), photo(i))
    assert len(list(sampleframes.frames(str(tmp_path), stride=5))) == 4


def test_asking_for_more_frames_than_there_are_names(one_face, monkeypatch):
    """--keep 10 used to write three and say nothing, because zip
    truncates. The cap is explicit now."""
    monkeypatch.setattr(facetraits, "embed", lambda frame, row: None)
    cands = [sampleframes.Candidate(photo(i), 0.9 - i * 0.05, ROW.copy())
             for i in range(8)]
    assert len(sampleframes.choose(cands, keep=10)) == len(sampleframes.NAMES)


def test_the_quality_read_happens_once_per_frame(one_face, monkeypatch):
    """quality_metrics runs a network forward pass; calling it for the
    flags and again for the score doubled the cost of the most expensive
    step on every frame."""
    calls = []
    real = facetraits.quality_metrics
    monkeypatch.setattr(facetraits, "quality_metrics",
                        lambda crop: calls.append(1) or real(crop))
    cand, reason = sampleframes.assess(photo())
    assert cand is not None, reason
    assert len(calls) == 1, f"measured {len(calls)} times"


# ------------------------------------------------------------ corpus shards


try:
    import pyarrow.parquet          # noqa: F401
    HAVE_PYARROW = True
except Exception:                   # pragma: no cover - depends on the env
    HAVE_PYARROW = False

needs_parquet = pytest.mark.skipif(
    not HAVE_PYARROW,
    reason="pyarrow is not installed; the parquet path exists for the "
           "benchmark corpora and imports it on demand")


def parquet_of(images, path):
    """A shard shaped like the HuggingFace image columns the corpora use."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    cells = []
    for img in images:
        ok, buf = cv2.imencode(".png", img)
        assert ok
        cells.append({"bytes": buf.tobytes(), "path": "x.png"})
    table = pa.table({"image": pa.array(cells)})
    pq.write_table(table, path)
    return path


@needs_parquet
def test_a_corpus_shard_is_read_directly(tmp_path):
    """The corpora are parquet, not folders. Handing the path to
    cv2.VideoCapture, which cannot read one, was advice that did not
    work."""
    path = parquet_of([photo(i) for i in range(4)],
                      str(tmp_path / "corpus.parquet"))
    got = list(sampleframes.frames(path))
    assert len(got) == 4
    assert got[0].shape == (480, 640, 3)


@needs_parquet
def test_a_shard_without_an_image_column_says_which_columns_it_has(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = str(tmp_path / "wrong.parquet")
    pq.write_table(pa.table({"label": pa.array([1, 2])}), path)
    with pytest.raises(ValueError) as raised:
        list(sampleframes.frames(path))
    assert "label" in str(raised.value)


def test_a_missing_shard_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(sampleframes.frames(str(tmp_path / "absent.parquet")))


@needs_parquet
def test_an_undecodable_cell_is_skipped(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ok, buf = cv2.imencode(".png", photo())
    cells = [{"bytes": buf.tobytes(), "path": "good.png"},
             {"bytes": b"not an image", "path": "bad.png"},
             None]
    path = str(tmp_path / "mixed.parquet")
    pq.write_table(pa.table({"image": pa.array(cells)}), path)
    assert len(list(sampleframes.frames(path))) == 1


# -------------------------------------------------------------- the limit


def test_the_limit_stops_the_scan(clip, one_face):
    """A corpus shard holds thousands of rows and only three are kept, so
    examining the lot is hours of network forward passes for no better
    answer."""
    path = clip([photo(i) for i in range(10)])
    candidates, _ = sampleframes.scan(path, stride=1, limit=4)
    assert len(candidates) == 4


def test_a_limit_of_zero_examines_everything(clip, one_face):
    path = clip([photo(i) for i in range(6)])
    candidates, _ = sampleframes.scan(path, stride=1, limit=0)
    assert len(candidates) == 6


@needs_parquet
def test_an_empty_cell_is_skipped(tmp_path):
    """A row present but with no bytes in it: not a decode failure, just
    nothing to decode."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    ok, buf = cv2.imencode(".png", photo())
    cells = [{"bytes": b"", "path": "empty.png"},
             {"bytes": buf.tobytes(), "path": "good.png"}]
    path = str(tmp_path / "empty.parquet")
    pq.write_table(pa.table({"image": pa.array(cells)}), path)
    assert len(list(sampleframes.frames(path))) == 1
