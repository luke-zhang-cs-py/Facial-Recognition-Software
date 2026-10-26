"""tools/benchmark_loop.py, a round at a time, on corpora built at test time.

The loop is meant to run for days over LFW, FairFace and the faceage shard.
Here each corpus is a few solid squares written into tmp_path (skipped
without pyarrow, which CI installs), the detector and the recogniser answer
from the pixel value or from a script, and --rounds keeps every run to one
or two rounds. The logs go to tmp_path, never to the repository's logs/.
"""
import json
import os

import cv2
import numpy as np
import pytest

from pipeline import recognition, traits
from tools import benchmark_loop as bl


def png(value, size=8):
    ok, data = cv2.imencode(".png", np.full((size, size, 3), value, np.uint8))
    assert ok
    return data.tobytes()


def shard(path, columns, metadata=None):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(columns)
    if metadata is not None:
        table = table.replace_schema_metadata(
            {b"huggingface": json.dumps(metadata).encode()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)
    return path


def lfw_shard(path):
    """Ada has four photographs, Alan one."""
    names = ["Ada_Lovelace", "Alan_Turing"]
    meta = {"info": {"features": {"label": {"names": names}}}}
    cells = [{"bytes": png(10 * i + 20), "path": None} for i in range(5)]
    return shard(path, {"image": cells, "label": [0, 0, 0, 0, 1]}, meta)


def by_value(answers):
    """A traits.analyze that answers from the frame's grey level."""
    def analyze(img):
        return answers[int(img[0, 0, 0])]
    return analyze


# ---------------------------------------------------------- statistics

def test_wilson_is_honest_at_the_edges():
    assert bl.wilson(0, 0) == (0.0, 0.0)
    low, high = bl.wilson(40, 40)
    assert 0.91 < low < 0.92 and high == 1.0
    low, high = bl.wilson(0, 40)
    assert low == 0.0 and 0.08 < high < 0.09


def test_decode_reads_both_cell_shapes_and_refuses_garbage():
    pytest.importorskip("PIL")
    ok, red = cv2.imencode(".png", np.full((4, 4, 3), (0, 0, 255), np.uint8))
    for cell in ({"bytes": red.tobytes()}, red.tobytes()):
        img = bl.decode(cell)
        assert img.shape == (4, 4, 3)
        assert img[0, 0].tolist() == [0, 0, 255], "BGR in, BGR out"
    assert bl.decode(b"not an image") is None
    assert bl.decode({"bytes": b""}) is None


def test_lfw_indexes_photographs_by_person(tmp_path):
    pytest.importorskip("PIL")
    lfw = bl.LFW(lfw_shard(str(tmp_path / "lfw.parquet")))
    assert len(lfw) == 5
    assert lfw.names == ["Ada_Lovelace", "Alan_Turing"]
    assert lfw.label_of == {"Ada Lovelace": 0, "Alan Turing": 1}
    assert lfw.by_label == {0: [0, 1, 2, 3], 1: [4]}
    assert int(lfw.image(2)[0, 0, 0]) == 40
    os.remove(str(tmp_path / "lfw.parquet"))     # the handle was closed


# ------------------------------------------------------ identity round

class FakeLFW(object):
    def __init__(self, broken=()):
        self.label_of = {"Ada Lovelace": 0, "Alan Turing": 1,
                         "Grace Hopper": 2}
        self.by_label = {0: [0, 1, 2, 3, 4], 1: [5], 2: [6]}
        self.broken = set(broken)
        self.asked = []

    def image(self, index):
        self.asked.append(index)
        if index in self.broken:
            return None
        return np.full((32, 32, 3), 100, np.uint8)


GALLERY = {
    1: {"name": "[demo] Ada Lovelace", "samples": 3},
    2: {"name": "[demo] Alan Turing", "samples": 4},
    3: {"name": "[demo] Grace Hopper", "samples": 1},
    4: {"name": "[demo] Not In LFW", "samples": 3},
    5: {"name": "Somebody Real", "samples": 3},
}


def scripted_identify(monkeypatch, answers):
    asked = []

    def identify(img, gallery):
        assert gallery is GALLERY and img.shape == (32, 32, 3)
        asked.append(img)
        return answers[(len(asked) - 1) % len(answers)]

    monkeypatch.setattr(recognition, "identify", identify)
    return asked


def test_identity_round_counts_correct_unknown_and_wrong(monkeypatch):
    scripted_identify(monkeypatch, [
        {"ok": False},
        {"ok": True, "match": None},
        {"ok": True, "match": {"name": "[demo] Ada Lovelace"},
         "best": {"similarity": 0.75}},
        {"ok": True, "match": {"name": "[demo] Alan Turing"},
         "best": {"similarity": 0.5}},
    ])
    # Grace's only photograph will not decode, so she has no trials at all.
    lfw = FakeLFW(broken={6})
    result = bl.identity_round(lfw, GALLERY, 4, np.random.default_rng(0))

    assert result["perPerson"] == {
        "Ada Lovelace": {"n": 4, "correct": 1, "unknown": 2, "wrong": 1},
        "Alan Turing": {"n": 4, "correct": 1, "unknown": 2, "wrong": 1},
    }
    assert (result["identities"], result["trials"], result["correct"],
            result["unknown"], result["wrong"]) == (2, 8, 2, 4, 2)
    assert result["meanSimilarity"] == 0.625
    # Ada was enrolled from her first three, so only 3 and 4 are drawn;
    # Alan's one photograph is reused because nothing else is held out.
    assert {5, 6} <= set(lfw.asked) <= {3, 4, 5, 6}


def test_identity_round_with_nothing_right_has_no_mean(monkeypatch):
    scripted_identify(monkeypatch, [{"ok": True, "match": None}])
    result = bl.identity_round(FakeLFW(), GALLERY, 2,
                               np.random.default_rng(1))
    assert result["correct"] == 0 and result["unknown"] == 6
    assert result["meanSimilarity"] is None


# ----------------------------------------------------- detection round

@pytest.fixture
def fairface(tmp_path, monkeypatch):
    folder = tmp_path / "fairface"
    monkeypatch.setattr(bl, "FAIRFACE_DIR", str(folder))
    return folder


def test_detection_round_is_skipped_without_fairface(fairface):
    assert bl.detection_round(np.random.default_rng(0)) is None


def test_detection_round_reports_each_group_and_the_disparity(fairface,
                                                              monkeypatch):
    """Twenty-five of one group (twenty found), three of another (all
    found, but too few to enter the disparity), one image that will not
    decode."""
    pytest.importorskip("PIL")
    cells = ([png(200)] * 20 + [png(100)] * 5 + [png(200)] * 3
             + [b"broken"])
    race = ["White"] * 25 + ["Black"] * 3 + ["White"]
    shard(str(fairface / "train-0.parquet"), {"image": cells, "race": race})
    monkeypatch.setattr(traits, "analyze", by_value(
        {200: {"detected": True}, 100: {"detected": False}}))

    found = bl.detection_round(np.random.default_rng(3), sample=100)
    assert found["sampled"] == 28 and found["detected"] == 23
    assert found["rate"] == round(23 / 28, 4)
    assert found["byGroup"]["White"]["n"] == 25
    assert found["byGroup"]["White"]["rate"] == 0.8
    assert found["byGroup"]["Black"] == {"n": 3, "detected": 3, "rate": 1.0,
                                         "ci": [0.4385, 1.0]}
    assert found["disparity"] == 1.0, "only White has the 20 it needs"


def test_detection_round_without_labels_or_hits_has_no_disparity(
        fairface, monkeypatch):
    pytest.importorskip("PIL")
    shard(str(fairface / "a.parquet"), {"image": [png(100)] * 22})
    monkeypatch.setattr(traits, "analyze", lambda img: None)
    found = bl.detection_round(np.random.default_rng(0), sample=22)
    assert found["sampled"] == 22 and found["detected"] == 0
    assert list(found["byGroup"]) == ["0"]
    assert found["disparity"] is None


def test_detection_round_where_nothing_decodes_has_no_rate(fairface):
    pytest.importorskip("PIL")
    shard(str(fairface / "a.parquet"), {"image": [b"x", b"y"]})
    found = bl.detection_round(np.random.default_rng(0))
    assert found == {"sampled": 0, "detected": 0, "rate": None,
                     "disparity": None, "byGroup": {}}


# ----------------------------------------------------------- age round

@pytest.fixture
def faceage(tmp_path, monkeypatch):
    path = tmp_path / "faceage" / "val.parquet"
    monkeypatch.setattr(bl, "FACEAGE_PARQUET", str(path))
    return path


def test_age_round_is_skipped_without_the_corpus(faceage):
    assert bl.age_round(np.random.default_rng(0)) is None


def test_age_round_is_skipped_without_an_age_column(faceage):
    shard(str(faceage), {"image": [png(1)]})
    assert bl.age_round(np.random.default_rng(0)) is None


def age(label):
    return {"detected": True, "demographics": {"age": {"label": label}}}


def test_age_round_scores_band_midpoints_and_skips_what_it_cannot(
        faceage, monkeypatch):
    pytest.importorskip("PIL")
    rows = [
        (png(1), 3),          # truth 25, said 25-32 (28): off by 3
        (png(2), 0),          # truth 1, said 0-2 (1): exact
        (png(3), 8),          # truth 75, said 60+ (70): off by 5
        (png(1), 12),         # a label FairFace does not have
        (png(1), None),       # no label at all
        (b"broken", 3),       # will not decode
        (png(4), 3),          # no face found
        (png(5), 3),          # a face, but no demographics
        (png(6), 3),          # a band this table does not know
        (png(7), 3),          # analyze returned nothing
    ]
    shard(str(faceage), {"image": [r[0] for r in rows],
                         "age": [r[1] for r in rows]})
    monkeypatch.setattr(traits, "analyze", by_value({
        1: age("25-32"), 2: age("0-2"), 3: age("60+"),
        4: {"detected": False}, 5: {"detected": True, "demographics": None},
        6: age("99-100"), 7: None}))

    found = bl.age_round(np.random.default_rng(0), sample=50)
    assert found == {"sampled": 3, "mae": 2.67, "median": 3.0,
                     "within2": 0.3333, "within5": 1.0, "within10": 1.0}


def test_age_round_with_nothing_scorable_returns_none(faceage, monkeypatch):
    pytest.importorskip("PIL")
    shard(str(faceage), {"image": [png(4)], "age": [3]})
    monkeypatch.setattr(traits, "analyze", lambda img: {"detected": False})
    assert bl.age_round(np.random.default_rng(0)) is None


# ------------------------------------------------- state and reporting

def test_summarise_before_and_after_any_trials():
    state = bl.fresh_state()
    assert bl.summarise(state) == "  no identity trials yet"
    state["rounds"] = 2
    state["cumulative"]["identity"].update(trials=1000, correct=900,
                                           unknown=80, wrong=20)
    text = bl.summarise(state)
    assert "cumulative over 1,000 trials / 2 rounds" in text
    assert "correct   90.00%" in text and "unknown    8.00%" in text
    assert "WRONG      2.00%" in text


def test_the_parser_defaults_are_the_named_constants():
    args = bl.build_parser().parse_args([])
    assert (args.rounds, args.trials, args.detection_every,
            args.age_every) == (0, bl.DEFAULT_TRIALS,
                                bl.DEFAULT_DETECTION_EVERY,
                                bl.DEFAULT_AGE_EVERY)


@pytest.mark.parametrize("content", ["{trunc", "[1, 2]", '{"cumulative": {}}'])
def test_an_unusable_state_file_starts_fresh(tmp_path, capsys, content):
    """The last case escaped as a KeyError before: valid JSON, no totals."""
    path = tmp_path / "state.json"
    path.write_text(content, encoding="utf-8")
    state = bl.load_state(str(path))
    assert state["rounds"] == 0 and "resuming" not in capsys.readouterr().out


def test_a_good_state_file_is_resumed(tmp_path, capsys):
    path = tmp_path / "state.json"
    assert bl.load_state(str(path))["rounds"] == 0, "no file: fresh"
    saved = dict(bl.fresh_state(), rounds=4)
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert bl.load_state(str(path)) == saved
    assert "resuming from 4 previous rounds" in capsys.readouterr().out


@pytest.fixture
def passes(monkeypatch):
    ran = []
    monkeypatch.setattr(bl, "identity_round",
                        lambda lfw, gal, trials, rng: ran.append("identity")
                        or {"trials": trials})
    monkeypatch.setattr(bl, "detection_round",
                        lambda rng: ran.append("detection") or {"d": 1})
    monkeypatch.setattr(bl, "age_round",
                        lambda rng: ran.append("age") or {"a": 1})
    return ran


@pytest.mark.parametrize("round_no, every, expected", [
    (6, (2, 3), ["identity", "detection", "age"]),
    (4, (2, 3), ["identity", "detection"]),
    (3, (2, 3), ["identity", "age"]),
    (1, (2, 3), ["identity"]),
    (6, (0, 0), ["identity"]),
])
def test_each_round_runs_the_slow_passes_only_on_their_schedule(
        passes, round_no, every, expected):
    args = bl.build_parser().parse_args(
        ["--trials", "7", "--detection-every", str(every[0]),
         "--age-every", str(every[1])])
    record = bl.run_round(None, {}, args, round_no, None)
    assert passes == expected
    assert record["round"] == round_no and record["identity"] == {"trials": 7}
    assert ("detection" in record) == ("detection" in expected)
    assert ("age" in record) == ("age" in expected)
    assert record["seconds"] >= 0


def identity(trials, correct, unknown, wrong):
    return {"trials": trials, "correct": correct, "unknown": unknown,
            "wrong": wrong}


def test_accumulate_keeps_an_error_sum_not_a_mean_of_means():
    state = bl.fresh_state()
    bl.accumulate(state, {"round": 1, "at": "t1",
                          "identity": identity(10, 7, 2, 1),
                          "detection": {"sampled": 30, "detected": 27},
                          "age": {"sampled": 10, "mae": 4.0}})
    bl.accumulate(state, {"round": 2, "at": "t2",
                          "identity": identity(10, 9, 1, 0),
                          "detection": None,
                          "age": {"sampled": 30, "mae": 8.0}})
    total = state["cumulative"]
    assert total["identity"] == identity(20, 16, 3, 1)
    assert total["detection"] == {"sampled": 30, "detected": 27}
    assert total["age"] == {"sampled": 40, "errorSum": 280.0}
    assert (state["rounds"], state["updatedAt"]) == (2, "t2")


def test_report_round_prints_and_logs_every_pass(tmp_path, capsys):
    log = tmp_path / "readable.log"
    bl.report_round({"round": 3, "seconds": 1.5,
                     "identity": identity(50, 40, 8, 2),
                     "detection": {"rate": 0.95, "sampled": 300,
                                   "disparity": 1.1},
                     "age": {"mae": 6.5, "within2": 0.25, "within10": 0.8,
                             "sampled": 120}}, str(log))
    out = capsys.readouterr().out
    lines = log.read_text(encoding="utf-8").splitlines()
    assert out.splitlines() == lines and len(lines) == 3
    assert lines[0].split() == ["round", "3", "50", "trials", "correct",
                                "80.0%", "unknown", "16.0%", "WRONG",
                                "4.0%", "(1.5s)"]
    assert "detection 95.0% over 300 FairFace images, disparity 1.1" in lines[1]
    assert "age MAE 6.5y, within2 25%, within10 80% over 120" in lines[2]


def test_report_round_survives_a_detection_pass_where_nothing_decoded(
        tmp_path, capsys):
    """`100 * None` raised TypeError here and ended the loop."""
    log = tmp_path / "readable.log"
    bl.report_round({"round": 1, "seconds": 0.1,
                     "identity": identity(0, 0, 0, 0),
                     "detection": {"rate": None, "sampled": 0,
                                   "disparity": None}}, str(log))
    assert "detection n/a over 0 FairFace images" in capsys.readouterr().out


# ---------------------------------------------------------------- main

@pytest.fixture
def loop_home(tmp_path, monkeypatch):
    """Logs, the stop file and every corpus under tmp_path."""
    monkeypatch.setattr(bl, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(bl, "STOP_FILE", str(tmp_path / "STOP_BENCHMARK"))
    monkeypatch.setattr(bl, "LFW_PARQUET", str(tmp_path / "lfw.parquet"))
    monkeypatch.setattr(bl, "FAIRFACE_DIR", str(tmp_path / "fairface"))
    monkeypatch.setattr(bl, "FACEAGE_PARQUET", str(tmp_path / "age.parquet"))
    return tmp_path


def test_main_without_lfw_has_nothing_to_test(loop_home, capsys):
    assert bl.main([]) == 1
    assert "LFW not cached at" in capsys.readouterr().out


def test_main_with_nobody_enrolled_says_how_to_fix_it(loop_home, monkeypatch,
                                                      capsys):
    (loop_home / "lfw.parquet").write_bytes(b"")
    monkeypatch.setattr(recognition, "gallery", lambda: {})
    assert bl.main([]) == 1
    assert "python -m cli.seed_demo" in capsys.readouterr().out


@pytest.fixture
def enrolled(loop_home, monkeypatch):
    pytest.importorskip("PIL")
    lfw_shard(str(loop_home / "lfw.parquet"))
    gallery = {1: {"name": "[demo] Ada Lovelace", "samples": 2}}
    monkeypatch.setattr(recognition, "gallery", lambda: gallery)
    monkeypatch.setattr(recognition, "identify", lambda img, gal: {
        "ok": True, "match": {"name": "[demo] Ada Lovelace"},
        "best": {"similarity": 0.9}})
    return loop_home


def test_main_runs_the_rounds_asked_for_and_resumes_after(enrolled,
                                                         monkeypatch,
                                                         capsys):
    monkeypatch.setattr(bl, "SUMMARY_EVERY", 2)
    logs = enrolled / "logs"

    assert bl.main(["--rounds", "2", "--trials", "3"]) == 0
    out = capsys.readouterr().out
    assert "5 photographs, 2 identities" in out and "1 enrolled" in out
    records = [json.loads(line) for line in
               (logs / "benchmark.jsonl").read_text("utf-8").splitlines()]
    assert [r["round"] for r in records] == [1, 2]
    assert all(r["identity"]["correct"] == 3 for r in records)
    # FairFace and faceage are absent, so the slow passes record nothing.
    assert records[0].get("detection") is None
    state = json.loads((logs / "benchmark_state.json").read_text("utf-8"))
    assert state["rounds"] == 2
    assert state["cumulative"]["identity"]["trials"] == 6
    assert out.count("cumulative over 6 trials / 2 rounds") == 2, (
        "once at the SUMMARY_EVERY round, once at the end")

    # --rounds counts this session, not the total.
    assert bl.main(["--rounds", "1", "--trials", "3"]) == 0
    out = capsys.readouterr().out
    assert "resuming from 2 previous rounds" in out
    assert "cumulative over 9 trials / 3 rounds" in out
    readable = (logs / "benchmark.log").read_text("utf-8")
    assert readable.count("=== benchmark loop started") == 2
    assert readable.count("=== stopped") == 2


def test_main_with_no_round_limit_runs_until_the_stop_file(enrolled,
                                                          monkeypatch,
                                                          capsys):
    real = bl.run_round

    def run_round(*args):
        record = real(*args)
        if record["round"] == 2:
            open(bl.STOP_FILE, "w").close()
        return record

    monkeypatch.setattr(bl, "run_round", run_round)
    assert bl.main(["--trials", "1"]) == 0
    out = capsys.readouterr().out
    assert "round    2" in out and "round    3" not in out
    assert "STOP_BENCHMARK found -- stopping." in out
    readable = (enrolled / "logs" / "benchmark.log").read_text("utf-8")
    assert "stopped: STOP_BENCHMARK present" in readable
