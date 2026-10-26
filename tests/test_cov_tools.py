"""The corpus scripts in tools/, on corpora built at test time.

tools/sample_frames.py, tools/scale_benchmark.py, tools/trials.py and
tools/build_gallery.py each expect a multi-gigabyte corpus in the system
temporary directory, and build_gallery downloads its own. None of that is
here: every corpus below is a handful of rows written into tmp_path (the
parquet ones skip without pyarrow, which CI installs), FACE_CORPORA and
CASIA_DIR point the scripts at them, the detector and the recogniser are
replaced with scripted answers, and curl is a function that writes a file.
Nothing leaves the process.
"""
import collections
import json
import os
import sys

import cv2
import numpy as np
import pytest

from analysis import calibration
from core import corpus_paths
from pipeline import recognition, sampleframes
from tools import benchmark_loop, build_gallery, sample_frames
from tools import scale_benchmark, trials


@pytest.fixture
def corpora(tmp_path, monkeypatch):
    """Every corpus location under one temporary folder."""
    monkeypatch.setenv("FACE_CORPORA", str(tmp_path))
    monkeypatch.delenv("CASIA_DIR", raising=False)
    return tmp_path


def png(value, size=32):
    """A solid grey square, encoded: PNG so the value survives the trip."""
    ok, data = cv2.imencode(".png", np.full((size, size, 3), value, np.uint8))
    assert ok
    return data.tobytes()


def lfw_shard(path, names, labels, cells):
    """A parquet shard laid out the way the Hugging Face LFW export is."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"image": cells, "label": labels})
    meta = {"info": {"features": {"label": {"names": names}}}}
    table = table.replace_schema_metadata(
        {b"huggingface": json.dumps(meta).encode()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)
    return path


class ScriptedRng(object):
    """The four draws `degrade` makes, answered from lists instead of chance,
    so each branch is chosen rather than hoped for."""

    def __init__(self, uniform, choice, random):
        self._uniform = list(uniform)
        self._choice = list(choice)
        self._random = list(random)
        self.noise = []

    def uniform(self, low, high):
        value = self._uniform.pop(0)
        assert low <= value <= high, "a draw outside the range asked for"
        return value

    def choice(self, options):
        value = self._choice.pop(0)
        assert value in options
        return value

    def random(self):
        return self._random.pop(0)

    def normal(self, mean, sigma, shape):
        self.noise.append(sigma)
        return np.full(shape, 40.0)

    def integers(self, low, high):
        return 90


# ------------------------------------------------------------ degrade

def edge():
    """Black left half, grey right half: a vertical edge at column 16."""
    img = np.zeros((32, 32, 3), np.uint8)
    img[:, 16:] = 200
    return img


@pytest.mark.parametrize("module", [benchmark_loop, trials])
@pytest.mark.parametrize("horizontal", [True, False])
def test_motion_blur_runs_along_the_axis_it_draws(module, horizontal):
    """No zoom, no rotation, unit gamma: only the blur moves pixels. A
    horizontal kernel smears a vertical edge; a vertical one leaves it."""
    rng = ScriptedRng(uniform=[1.0, 0.0, 1.0], choice=[9],
                      random=[0.1 if horizontal else 0.9, 0.9])
    out = module.degrade(edge(), rng)
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8
    step = int(out[16, 16, 0]) - int(out[16, 15, 0])
    if horizontal:
        assert step < 60, "the edge survived a horizontal blur"
    else:
        assert step > 150, "a vertical blur smeared a vertical edge"
    assert rng.noise == []


@pytest.mark.parametrize("module", [benchmark_loop, trials])
def test_no_blur_and_sensor_noise_brighten_a_dark_frame(module):
    rng = ScriptedRng(uniform=[1.0, 0.0, 1.0, 5.0], choice=[1],
                      random=[0.1])
    out = module.degrade(np.zeros((32, 32, 3), np.uint8), rng)
    assert rng.noise == [5.0], "the noise sigma is the fourth uniform draw"
    assert 30 < out.mean() < 50


@pytest.mark.parametrize("module", [benchmark_loop, trials])
def test_a_failed_jpeg_encode_returns_the_unencoded_frame(module,
                                                         monkeypatch):
    monkeypatch.setattr(cv2, "imencode", lambda *a, **k: (False, None))
    rng = ScriptedRng(uniform=[1.0, 0.0, 1.0], choice=[1], random=[0.9])
    img = np.full((32, 32, 3), 128, np.uint8)
    out = module.degrade(img, rng)
    # Unit gamma through a uint8 table truncates by at most one level.
    assert np.abs(out.astype(int) - 128).max() <= 1


# ------------------------------------------------------- sample_frames

def test_sample_frames_dry_run_reports_and_writes_nothing(corpora,
                                                          monkeypatch,
                                                          capsys):
    seen = {}

    def scan(source, stride, limit):
        seen.update(source=source, stride=stride, limit=limit)
        return ["a", "b"], collections.Counter({"too dark": 3})

    monkeypatch.setattr(sampleframes, "scan", scan)
    monkeypatch.setattr(sampleframes, "choose", lambda c, keep: c[:keep])
    monkeypatch.setattr(sampleframes, "build", pytest.fail)

    assert sample_frames.main(["clip.mp4", "--dry-run", "--keep", "1",
                               "--stride", "7", "--limit", "0"]) == 0
    out = capsys.readouterr().out
    assert seen == {"source": "clip.mp4", "stride": 7, "limit": 0}
    assert "target : %s  (dry run)" % os.path.join(str(corpora), "frames") in out
    assert "2 frame(s) usable, 1 would be kept" in out
    assert "no frame was good enough" in out and "too dark" in out


def test_sample_frames_writes_where_it_is_told(tmp_path, monkeypatch, capsys):
    target = tmp_path / "elsewhere"
    seen = {}

    def build(source, **kw):
        seen.update(kw)
        return [str(target / "best.png")], collections.Counter()

    monkeypatch.setattr(sampleframes, "build", build)
    assert sample_frames.main(["photos", "--into", str(target),
                               "--keep", "2"]) == 0
    out = capsys.readouterr().out
    assert seen == {"keep": 2, "stride": sampleframes.STRIDE,
                    "directory": str(target), "limit": sampleframes.LIMIT}
    assert "target : %s\n" % target in out
    assert "wrote 1 sample frame(s):" in out and "best.png" in out


def test_sample_frames_exits_1_when_nothing_was_kept(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(sampleframes, "build",
                        lambda source, **kw: ([], collections.Counter({"no face": 4})))
    assert sample_frames.main(["clip.mp4", "--into", str(tmp_path)]) == 1
    assert "no frame was good enough to keep." in capsys.readouterr().out


@pytest.mark.parametrize("dry", [[], ["--dry-run"]])
def test_sample_frames_a_missing_source_is_a_message_not_a_traceback(
        tmp_path, capsys, dry):
    missing = str(tmp_path / "nope.mp4")
    assert sample_frames.main([missing, "--into", str(tmp_path)] + dry) == 1
    assert "could not read the source: %s" % missing in capsys.readouterr().err


# ----------------------------------------------------- scale_benchmark

def basis(i, dims=128):
    v = np.zeros(dims, np.float32)
    v[i] = 1.0
    return v


def write_gallery(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = [label for label, _ in rows]
    np.savez(path, labels=np.array(labels), vecs=np.vstack([v for _, v in rows]))


@pytest.fixture
def casia(tmp_path, monkeypatch):
    monkeypatch.setenv("CASIA_DIR", str(tmp_path / "casia"))
    return corpus_paths.gallery_path()


def five_identities():
    """Three that are recognised, one whose probe is somebody else's face,
    one whose probe resembles nobody enough to clear the threshold, and one
    with too few embeddings to be used at all."""
    rows = []
    for k in range(5):
        rows += [("id%d" % k, basis(k))] * 3
    rows += [("id0", basis(0)), ("id1", basis(1)), ("id2", basis(2)),
             ("id3", basis(4))]
    faint = basis(10) + 0.2 * basis(0)
    rows.append(("id4", faint / np.linalg.norm(faint)))
    rows += [("short", basis(20))] * 3
    return rows


def test_scale_benchmark_measures_each_gallery_size(casia, monkeypatch,
                                                    capsys):
    write_gallery(casia, five_identities())
    monkeypatch.setattr(scale_benchmark, "SIZES", [2, 3, 10])
    monkeypatch.setattr(scale_benchmark, "BATCH", 2)

    assert scale_benchmark.main() == 0
    out = capsys.readouterr().out
    assert "loaded 23 embeddings, 6 identities" in out
    assert "5 identities have the 4 embeddings this needs" in out
    rows = {line.split()[0]: line.split() for line in out.splitlines()
            if line.strip() and line.split()[0].isdigit()}
    assert sorted(rows, key=int) == ["2", "3", "5"], "10 > 5, and 5 is added"
    n, thr, rank1, correct, rejected, misid, risk = rows["5"]
    assert float(thr) == calibration.recommend_threshold(5)[0]
    assert (rank1, correct, rejected, misid) == ("60.0%", "60.0%", "20.0%",
                                                 "20.0%")
    assert "UNREACHABLE" not in out


def test_scale_benchmark_marks_an_unreachable_target(casia, monkeypatch,
                                                     capsys):
    write_gallery(casia, five_identities())
    monkeypatch.setattr(scale_benchmark, "SIZES", [])
    monkeypatch.setattr(calibration, "recommend_threshold",
                        lambda n: (0.5, 0.2, False))
    scale_benchmark.main()
    line, = [x for x in capsys.readouterr().out.splitlines()
             if x.split()[:1] == ["5"] and "%" in x]
    assert line.endswith("20.00%  UNREACHABLE")


def test_scale_benchmark_sizes_stop_at_the_gallery():
    assert scale_benchmark.sizes_for(250) == [100, 250]
    assert scale_benchmark.sizes_for(300) == [100, 250, 300]
    assert scale_benchmark.sizes_for(20000)[-2:] == [10000, 20000]


def test_scale_benchmark_without_a_gallery_says_how_to_build_one(casia):
    with pytest.raises(SystemExit) as stop:
        scale_benchmark.main()
    assert "No gallery at %s" % casia in str(stop.value)
    assert "tools/build_gallery.py" in str(stop.value)


def test_scale_benchmark_with_no_usable_identity_says_so(casia, capsys):
    """This died inside np.vstack([]) before."""
    write_gallery(casia, [("a", basis(0))] * 3)
    with pytest.raises(SystemExit) as stop:
        scale_benchmark.main()
    assert "No identity has 4 embeddings" in str(stop.value)
    assert "0 identities have the 4 embeddings" in capsys.readouterr().out


# -------------------------------------------------------------- trials

DEMO = {
    1: {"name": "[demo] Ada Lovelace", "samples": 3},
    2: {"name": "[demo] Alan Turing", "samples": 5},
    3: {"name": "[demo] Grace Hopper", "samples": 1},
    4: {"name": "[demo] Nobody In The Corpus", "samples": 3},
    5: {"name": "Real Person", "samples": 2},
}

# The recogniser's answers, in the order it is asked: declined twice (once
# with no face, once under threshold), then Ada, then Alan.
ANSWERS = [{"ok": False},
           {"ok": True, "match": None},
           {"ok": True, "match": {"name": "[demo] Ada Lovelace"},
            "best": {"similarity": 0.8}},
           {"ok": True, "match": {"name": "[demo] Alan Turing"},
            "best": {"similarity": 0.6}}]


@pytest.fixture
def scripted_recogniser(monkeypatch):
    asked = []

    def identify(img, gal):
        assert gal is DEMO
        asked.append(img)
        return ANSWERS[(len(asked) - 1) % len(ANSWERS)]

    monkeypatch.setattr(recognition, "gallery", lambda: DEMO)
    monkeypatch.setattr(recognition, "identify", identify)
    return asked


def trials_corpus(corpora):
    """Ada has five photographs (three enrolled), Alan two (both enrolled,
    so the last is reused), Grace one."""
    labels = [0, 0, 0, 0, 0, 1, 1, 2]
    cells = [{"bytes": png(10 * i + 10), "path": None}
             for i in range(len(labels))]
    return lfw_shard(corpus_paths.lfw_parquet(),
                     ["Ada_Lovelace", "Alan_Turing", "Grace_Hopper"],
                     labels, cells)


def test_trials_scores_every_demo_identity_three_ways(corpora,
                                                      scripted_recogniser,
                                                      capsys):
    pytest.importorskip("PIL")
    trials_corpus(corpora)
    assert trials.main(["4"]) == 0
    out = capsys.readouterr().out

    assert "gallery: 5 enrolled, 4 demo identities" in out
    assert "4 randomised trials each" in out
    table = {" ".join(line.split()[:2]): line.split()[2:]
             for line in out.splitlines()
             if line.startswith("  ") and not line.startswith("   ")}
    assert table["Ada Lovelace"] == ["4", "25%", "50%", "25%", "0.800"]
    assert table["Alan Turing"] == ["4", "25%", "50%", "25%", "0.600"]
    assert table["Grace Hopper"] == ["4", "0%", "50%", "50%", "0.000"]
    assert "Nobody In" not in table, "not in the corpus, so not scored"
    assert table["ALL 12"] == ["16.7%", "50.0%", "33.3%"]
    assert "correct   2" in out and "WRONG     4" in out
    assert out.splitlines()[-3].split() == ["Grace", "Hopper", "2", "wrong",
                                            "/", "4"]
    assert len(scripted_recogniser) == 12
    assert all(img.shape == (32, 32, 3) for img in scripted_recogniser)


def test_trials_load_reads_either_cell_shape():
    pytest.importorskip("PIL")
    for cell in ({"bytes": png(77)}, png(77)):
        img = trials.load(cell)
        assert img.shape == (32, 32, 3) and int(img[0, 0, 0]) == 77


def test_trials_holds_out_the_images_a_person_was_enrolled_from():
    assert trials.heldout([4, 5, 6, 7], 3) == [7]
    assert trials.heldout([4, 5], 3) == [5]


def test_trials_without_the_corpus_names_where_it_looked(corpora):
    with pytest.raises(SystemExit) as stop:
        trials.main([])
    assert "No LFW corpus at %s" % corpus_paths.lfw_parquet() in str(stop.value)


def test_trials_refuses_a_trial_count_it_cannot_divide_by(corpora):
    with pytest.raises(SystemExit) as stop:
        trials.main(["0"])
    assert "at least 1" in str(stop.value)


def test_trials_with_nobody_to_test_says_so(corpora, monkeypatch, capsys):
    """This divided by zero on the ALL line before."""
    trials_corpus(corpora)
    monkeypatch.setattr(recognition, "gallery", lambda: {})
    monkeypatch.setattr(sys, "argv", ["trials.py"])
    with pytest.raises(SystemExit) as stop:
        trials.main()
    assert "No enrolled [demo] identity" in str(stop.value)
    assert "100 randomised trials each" in capsys.readouterr().out


# ------------------------------------------------------- build_gallery

@pytest.fixture
def fake_traits(monkeypatch):
    """A detector that finds a face in any square brighter than 50, and an
    embedder that declines one brighter than 200."""
    from pipeline import traits
    monkeypatch.setattr(traits, "detect",
                        lambda bgr: [object()] if bgr[0, 0, 0] > 50 else [])
    monkeypatch.setattr(
        traits, "embed",
        lambda bgr, row: None if bgr[0, 0, 0] > 200
        else np.full(128, bgr[0, 0, 0], np.float64))
    threads = []
    monkeypatch.setattr(cv2, "setNumThreads", threads.append)
    moved = []
    monkeypatch.setattr(os, "chdir", moved.append)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return threads, moved


def test_embed_batch_keeps_only_what_detects_and_embeds(fake_traits):
    pytest.importorskip("PIL")
    threads, moved = fake_traits
    out = build_gallery.embed_batch([
        ("bad", b"not an image"),
        ("dark", png(20)),
        ("glare", png(230)),
        ("good", png(100)),
    ])
    assert [label for label, _ in out] == ["good"]
    assert out[0][1].dtype == np.float32 and out[0][1][0] == 100
    assert threads == [1] and moved == [build_gallery.PROJ]


def test_the_parser_refuses_what_the_hand_rolled_one_accepted():
    args = build_gallery.build_parser().parse_args([])
    assert (args.shards, args.per_id, args.workers) == (20, 4, 8)
    with pytest.raises(SystemExit):
        build_gallery.build_parser().parse_args(["--shards"])


def fake_curl(monkeypatch, size, code=0):
    calls = []

    def call(argv):
        calls.append(argv)
        if size is not None:
            with open(argv[argv.index("-o") + 1], "wb") as handle:
                handle.write(b"x" * size)
        return code

    monkeypatch.setattr(build_gallery.subprocess, "call", call)
    return calls


def test_download_shard_accepts_a_plausible_file(tmp_path, monkeypatch):
    monkeypatch.setattr(build_gallery, "MIN_SHARD_BYTES", 10)
    calls = fake_curl(monkeypatch, 10)
    path = str(tmp_path / "s3.parquet")
    assert build_gallery.download_shard(3, path) is True
    assert calls[0][0] == "curl"
    assert build_gallery.BASE + "/3.parquet" in calls[0]
    assert calls[0][calls[0].index("--max-time") + 1] == "1800"


@pytest.mark.parametrize("size, code, said", [
    (None, 0, "download failed (rc=0)"),
    (10, 6, "download failed (rc=6)"),
    (9, 0, "download truncated (9 bytes)"),
])
def test_download_shard_refuses_a_failed_or_short_download(
        tmp_path, monkeypatch, capsys, size, code, said):
    monkeypatch.setattr(build_gallery, "MIN_SHARD_BYTES", 10)
    fake_curl(monkeypatch, size, code)
    assert build_gallery.download_shard(1, str(tmp_path / "s1.parquet")) is False
    assert said in capsys.readouterr().out


def casia_shard(path, labels, struct=True):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = [png(60 + 10 * i) for i in range(len(labels))]
    cells = [{"bytes": d, "path": None} for d in data] if struct else data
    pq.write_table(pa.table({"image": cells, "label": labels}), path)
    return data


@pytest.mark.parametrize("struct", [True, False])
def test_collect_images_caps_each_identity_across_batches(tmp_path,
                                                          monkeypatch,
                                                          struct):
    monkeypatch.setattr(build_gallery, "BATCH_ROWS", 2)
    path = str(tmp_path / "s0.parquet")
    data = casia_shard(path, [5, 5, 6, 5, 5], struct)
    wanted = build_gallery.collect_images(path, 7, per_id=3)
    assert dict(wanted) == {"7_5": [data[0], data[1], data[3]],
                            "7_6": [data[2]]}
    os.remove(path)     # the handle was closed, so Windows lets this go


def test_discard_removes_a_shard_and_survives_one_already_gone(tmp_path,
                                                               capsys):
    path = tmp_path / "s0.parquet"
    path.write_bytes(b"x")
    build_gallery.discard(str(path))
    assert not path.exists()
    build_gallery.discard(str(path))
    assert "(could not delete shard:" in capsys.readouterr().out


class FakePool(object):
    """multiprocessing.Pool, in this process."""

    made = []

    def __init__(self, workers):
        self.workers = workers
        self.closed = self.joined = False
        FakePool.made.append(self)

    def imap_unordered(self, fn, chunks):
        return map(fn, chunks)

    def close(self):
        self.closed = True

    def join(self):
        self.joined = True


def test_embed_all_spreads_the_work_over_every_worker(fake_traits):
    pytest.importorskip("PIL")
    chunks = []
    pool = FakePool(3)
    pool.imap_unordered = lambda fn, parts: (chunks.extend(parts)
                                             or map(fn, parts))
    wanted = {"0_1": [png(100), png(110)], "0_2": [png(120)]}
    out = build_gallery.embed_all(pool, wanted, 3)
    assert len(chunks) == 3 and all(len(c) == 1 for c in chunks)
    assert sorted(label for label, _ in out) == ["0_1", "0_1", "0_2"]


def test_save_writes_an_empty_gallery_that_still_has_128_columns(tmp_path,
                                                                 monkeypatch):
    out = str(tmp_path / "gallery.npz")
    monkeypatch.setattr(build_gallery, "OUT", out)
    build_gallery.save([], [])
    with np.load(out) as saved:
        assert saved["vecs"].shape == (0, 128)
        assert saved["labels"].shape == (0,)


def test_build_gallery_end_to_end_until_a_download_fails(tmp_path,
                                                         monkeypatch,
                                                         fake_traits,
                                                         capsys):
    """Shard 0 is already on disk, shard 1 downloads, shard 2 does not:
    the run stops there, keeps what it has, and deletes both shards."""
    pytest.importorskip("PIL")
    home = tmp_path / "casia"
    monkeypatch.setattr(build_gallery, "D", str(home))
    monkeypatch.setattr(build_gallery, "OUT", str(home / "gallery.npz"))
    FakePool.made = []
    monkeypatch.setattr(build_gallery.mp, "Pool", FakePool)
    os.makedirs(str(home))
    casia_shard(str(home / "s0.parquet"), [1, 1, 2])
    fetched = []

    def download(index, path):
        fetched.append(index)
        if index == 2:
            return False
        casia_shard(path, [1, 3], struct=False)
        return True

    monkeypatch.setattr(build_gallery, "download_shard", download)
    assert build_gallery.main(["--shards", "5", "--per-id", "2",
                               "--workers", "2"]) == 0
    out = capsys.readouterr().out

    assert fetched == [1, 2], "shard 0 was on disk; nothing after 2 is tried"
    assert "[ 2] stopping here" in out
    assert "done: 4 identities, 5 embeddings" in out
    assert not list(home.glob("*.parquet")), "every shard is deleted"
    with np.load(str(home / "gallery.npz")) as saved:
        assert sorted(saved["labels"].tolist()) == ["0_1", "0_1", "0_2",
                                                    "1_1", "1_3"]
        assert saved["vecs"].shape == (5, 128)
    pool, = FakePool.made
    assert pool.workers == 2 and pool.closed and pool.joined


def test_build_gallery_runs_every_shard_it_was_asked_for(tmp_path,
                                                         monkeypatch,
                                                         fake_traits,
                                                         capsys):
    pytest.importorskip("PIL")
    home = tmp_path / "casia"
    monkeypatch.setattr(build_gallery, "D", str(home))
    monkeypatch.setattr(build_gallery, "OUT", str(home / "gallery.npz"))
    monkeypatch.setattr(build_gallery.mp, "Pool", FakePool)
    monkeypatch.setattr(build_gallery, "download_shard", pytest.fail)
    os.makedirs(str(home))
    casia_shard(str(home / "s0.parquet"), [4, 9])
    assert build_gallery.main(["--shards", "1", "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "stopping here" not in out
    assert "done: 2 identities, 2 embeddings" in out
