"""analysis/fairness_benchmark.py on a tiny labelled corpus.

The statistics (Wilson intervals, stratified draws, the verdict) are tested
on plain arrays. main() runs twice over: with the parquet reader and the
analysis replaced, so it runs in CI, and once end to end against a parquet
file of small geometric PNGs written in tmp_path, with traits.analyze
replaced -- the detector is mocked, nothing is asked to find a real face.
That one needs pyarrow and Pillow, which requirements.txt does not install,
so it skips without them.
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest

from analysis import fairness_benchmark as fb


def strict_loads(text):
    def refuse(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(text, parse_constant=refuse)


# -------------------------------------------------------------- statistics

def test_wilson_interval():
    assert fb.wilson(0, 0) == (0.0, 0.0)
    lo, hi = fb.wilson(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05, "zero events still has an upper bound"
    lo, hi = fb.wilson(100, 100)
    assert 0.95 < lo < 1.0 and hi == pytest.approx(1.0)
    lo, hi = fb.wilson(50, 100)
    assert lo < 0.5 < hi and lo == pytest.approx(1 - hi)


def test_stratified_indices_draw_equally_and_deterministically():
    groups = np.array([0] * 50 + [1] * 10 + [2] * 30)
    idx = fb.stratified_indices(groups, 20, seed=3)
    counts = np.bincount(groups[idx])
    assert counts.tolist() == [20, 10, 20], "a small group gives all it has"
    assert np.all(np.diff(idx) > 0), "sorted, without repeats"
    assert np.array_equal(idx, fb.stratified_indices(groups, 20, seed=3))
    assert not np.array_equal(idx, fb.stratified_indices(groups, 20, seed=4))


def rate(hits_per_group, n=100, names=("A", "B", "C")):
    groups = np.repeat(np.arange(len(hits_per_group)), n)
    hits = np.zeros(len(groups), bool)
    for g, k in enumerate(hits_per_group):
        hits[g * n:g * n + k] = True
    return fb.report_rate("test", hits, groups, list(names))


def test_one_group_has_no_disparity(capsys):
    out = rate([5])
    assert out["disparity"] is None and len(out["rows"]) == 1


def test_an_even_gate_is_ok(capsys):
    out = rate([10, 11])
    assert out["verdict"] == "OK" and out["disparity"] == pytest.approx(1.1)
    assert "[OK]" in capsys.readouterr().out


def test_a_wide_gap_with_overlapping_intervals_is_inconclusive(capsys):
    out = rate([1, 3])
    assert out["verdict"] == "INCONCLUSIVE" and out["ciSeparated"] is False
    assert "(CIs overlap)" in capsys.readouterr().out


def test_a_wide_separated_gap_is_biased(capsys):
    out = rate([10, 40])
    assert out["verdict"] == "BIASED" and out["ciSeparated"] is True
    assert out["worst"]["group"] == "B" and out["best"]["group"] == "A"


def test_zero_against_something_is_an_infinite_ratio(capsys):
    out = rate([0, 2])
    assert out["disparity"] == float("inf")
    assert "disparity inf" in capsys.readouterr().out
    assert rate([0, 0])["disparity"] == 1.0


def test_a_group_past_the_name_table_is_labelled_by_number():
    out = rate([5, 5, 5], names=("A",))
    assert [r["group"] for r in out["rows"]] == ["A", "group 1", "group 2"]


# ------------------------------------------------------- main, mocked I/O

def analysed(detected=True, flags=()):
    return {"detected": detected, "flags": list(flags)}


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    """A corpus folder with one (empty) parquet name in it, and a way to say
    what the labels and the per-image analysis are."""
    (tmp_path / "part.parquet").write_bytes(b"")
    state = {}

    def load_labels(files, columns):
        state["columns"] = columns
        labels = {c: np.asarray(state["labels"][c], np.int16) for c in columns}
        n = len(next(iter(labels.values())))
        return labels, [(files[0], 0, n)], n

    def analyse_sample(indices, offsets):
        return [state["result"](int(i)) for i in indices]

    monkeypatch.setattr(fb, "load_labels", load_labels)
    monkeypatch.setattr(fb, "analyse_sample", analyse_sample)
    state["dir"] = str(tmp_path)
    return state


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["fairness_benchmark", *argv])
    return fb.main()


def test_no_parquet_files(monkeypatch, tmp_path, capsys):
    assert run(monkeypatch, "--corpus", str(tmp_path)) == 1
    assert "no parquet files under" in capsys.readouterr().out


def test_an_even_corpus_passes(corpus, monkeypatch, capsys, tmp_path):
    n = 200
    corpus["labels"] = {"race": [i % 2 for i in range(n)],
                        "gender": [(i // 2) % 2 for i in range(n)]}
    # 10% flagged in each group: i % 10 == 0 is even, == 1 is odd.
    corpus["result"] = lambda i: analysed(
        flags=["blurry"] if i % 10 in (0, 1) else [])
    out_json = str(tmp_path / "out.json")
    assert run(monkeypatch, "--corpus", corpus["dir"], "--per-group", "100",
               "--json", out_json) == 0
    text = capsys.readouterr().out
    assert "every check within the 1.25x disparity budget" in text
    assert "FAIRNESS BY GENDER" in text
    report = strict_loads(open(out_json, encoding="utf-8").read())
    assert set(report["sections"]) == {"detection_failure", "flagged",
                                       "flag_blurry", "gender_flagged"}
    assert report["sample"] == 200 and report["perGroup"] == 100
    assert corpus["columns"] == ["race", "gender"]


def test_a_biased_check_fails_and_is_named(corpus, monkeypatch, capsys):
    n = 400
    corpus["labels"] = {"race": [i % 2 for i in range(n)]}
    corpus["result"] = lambda i: analysed(
        flags=["low quality"] if (i % 2 and i % 4 != 1) or i % 40 == 0 else [])
    code = run(monkeypatch, "--corpus", corpus["dir"], "--per-group", "200",
               "--extra-columns", "")
    out = capsys.readouterr().out
    assert code == 1
    assert "check(s) over the 1.25x budget" in out
    assert "  - flagged: " in out and "  - flag_low quality: " in out
    assert "FAIRNESS BY GENDER" not in out


def test_a_rare_flag_is_not_broken_down(corpus, monkeypatch):
    """Under 20 hits a per-flag breakdown is noise, so it is not reported."""
    n = 100
    corpus["labels"] = {"race": [i % 2 for i in range(n)]}
    corpus["result"] = lambda i: analysed(flags=["blurry"] if i < 5 else [])
    out_path = os.path.join(corpus["dir"], "r.json")
    run(monkeypatch, "--corpus", corpus["dir"], "--extra-columns", "",
        "--json", out_path)
    report = json.load(open(out_path, encoding="utf-8"))
    assert "flag_blurry" not in report["sections"]


def test_an_unnamed_label_column_gets_numbered_groups(corpus, monkeypatch,
                                                      capsys):
    n = 60
    corpus["labels"] = {"camera": [i % 3 for i in range(n)],
                        "site": [i % 2 for i in range(n)]}
    corpus["result"] = lambda i: analysed()
    run(monkeypatch, "--corpus", corpus["dir"], "--group-column", "camera",
        "--extra-columns", "site")
    out = capsys.readouterr().out
    assert "group 2" in out and "site 1" in out


def test_an_infinite_disparity_is_written_as_valid_json(corpus, monkeypatch,
                                                        tmp_path):
    """0% in one group against anything in another is an infinite ratio.
    json.dump wrote it as the bare token Infinity, which is not JSON: a
    strict parser -- a browser, jq -- rejects the whole report."""
    n = 400
    corpus["labels"] = {"race": [i % 2 for i in range(n)]}
    corpus["result"] = lambda i: analysed(flags=["blurry"] if i % 2 else [])
    out_json = str(tmp_path / "inf.json")
    assert run(monkeypatch, "--corpus", corpus["dir"], "--per-group", "200",
               "--extra-columns", "", "--json", out_json) == 1
    report = strict_loads(open(out_json, encoding="utf-8").read())
    flagged = report["sections"]["flagged"]
    assert flagged["disparity"] is None
    assert flagged["verdict"] == "BIASED"
    assert flagged["ciSeparated"] is True, "a boolean, not 1.0"


def test_images_that_could_not_be_analysed_are_not_detection_failures(
        corpus, monkeypatch, capsys, tmp_path):
    """An image the decoder could not read was counted as a detection
    failure (and as an unflagged image) for its group. Ten corrupt files in
    one group read as a BIASED detector that had in fact found every face
    it was shown."""
    n = 200
    corpus["labels"] = {"race": [i % 2 for i in range(n)]}
    corpus["result"] = lambda i: None if (i % 2 == 0 and i < 40) else analysed()
    out_json = str(tmp_path / "r.json")
    code = run(monkeypatch, "--corpus", corpus["dir"], "--per-group", "100",
               "--extra-columns", "", "--json", out_json)
    out = capsys.readouterr().out
    assert code == 0, out
    assert "analysed 180/200" in out
    report = json.load(open(out_json, encoding="utf-8"))
    det = report["sections"]["detection_failure"]
    assert det["verdict"] == "OK"
    assert [r["n"] for r in det["rows"]] == [80, 100]
    assert report["analysed"] == 180


# ---------------------------------------------------------- end to end

def tiny_png(value):
    ok, buf = cv2.imencode(".png", np.full((4, 4, 3), value, np.uint8))
    return buf.tobytes()


def write_corpus(folder, rows):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("PIL")
    half = len(rows) // 2
    for k, part in enumerate((rows[:half], rows[half:])):
        table = pa.table({
            "image": [{"bytes": b, "path": None} for b, _, _ in part],
            "race": [r for _, r, _ in part],
            "gender": [g for _, _, g in part]})
        pq.write_table(table, os.path.join(folder, f"{k}.parquet"))


def test_end_to_end_over_parquet(monkeypatch, tmp_path, capsys):
    rows = [(tiny_png(i % 200) if i != 7 else b"corrupt", i % 2, (i // 2) % 2)
            for i in range(520)]
    write_corpus(str(tmp_path), rows)
    seen = []

    def analyze(bgr, **kw):
        seen.append((bgr.shape, kw))
        return {"detected": True, "flags": []}
    from pipeline import traits
    monkeypatch.setattr(traits, "analyze", analyze)
    out_json = str(tmp_path / "r.json")
    assert run(monkeypatch, "--corpus", str(tmp_path), "--per-group", "260",
               "--json", out_json) == 0
    out = capsys.readouterr().out
    assert "corpus: 520 images in 2 file(s)" in out
    assert "    250/520" in out and "    500/520" in out
    assert "analysed 519/520" in out, "the corrupt image is skipped, not fatal"
    assert seen[0] == ((4, 4, 3), {"want_embedding": False,
                                   "want_demographics": False,
                                   "require_detection": True})
    report = strict_loads(open(out_json, encoding="utf-8").read())
    assert report["sections"]["detection_failure"]["verdict"] == "OK"


def test_load_labels_concatenates_files_with_offsets(tmp_path):
    rows = [(tiny_png(1), i % 3, i % 2) for i in range(10)]
    write_corpus(str(tmp_path), rows)
    files = sorted(str(p) for p in tmp_path.glob("*.parquet"))
    labels, offsets, total = fb.load_labels(files, ["race", "gender"])
    assert total == 10
    assert offsets == [(files[0], 0, 5), (files[1], 5, 10)]
    assert labels["race"].tolist() == [i % 3 for i in range(10)]
    assert labels["race"].dtype == np.int16


def test_analyse_sample_skips_files_with_nothing_sampled(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [(tiny_png(10 * i), 0, 0) for i in range(10)]
    write_corpus(str(tmp_path), rows)
    files = sorted(str(p) for p in tmp_path.glob("*.parquet"))
    _, offsets, _ = fb.load_labels(files, ["race"])
    from pipeline import traits
    means = []
    monkeypatch.setattr(traits, "analyze",
                        lambda bgr, **kw: means.append(int(bgr.mean())) or {})
    opened = []
    real = pq.read_table
    monkeypatch.setattr(pq, "read_table",
                        lambda path, **kw: opened.append(path) or real(path, **kw))
    out = fb.analyse_sample(np.array([1, 3]), offsets)
    assert out == [{}, {}] and means == [10, 30]
    assert opened == [files[0]], "the second file had nothing sampled"


def test_json_ready():
    data = {"a": np.float32(1.5), "b": (np.bool_(True), float("-inf")),
            "c": [np.int16(3), float("nan")], "d": "x"}
    assert fb.json_ready(data) == {"a": 1.5, "b": [True, None],
                                   "c": [3, None], "d": "x"}
