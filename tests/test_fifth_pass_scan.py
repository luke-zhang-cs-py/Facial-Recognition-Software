"""The fifth pass (a scan of the fourth): one regression test per verified finding.

Each of these fails on the code as it was at e7a8ecd. No face images, and
nothing from the network: the databases are built here, in tmp_path, and the
parquet corpora are a few grey squares.

1.  db._one_mark_per_day was quadratic, and concurrent starters died on a big
    database with "database is locked".
2.  paths.folder_id: an id of 2**63 or more made registration fail for good.
3.  fairness_benchmark: a crash when no image was analysed, and failures
    swallowed by `except Exception: pass`.
5.  conftest.isolated_root leaked a temp directory per test.
6.  pad_eval: an Excel BOM and spaced headers, and float residue in the bounds.
7.  seed_demo: `--keep --names X` enrolled somebody already enrolled again.
9.  fetch_models --verify, for files that were not just downloaded.
10. test_properties: reproducible in CI.
"""
import hashlib
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import date, timedelta

import cv2
import numpy as np
import pytest

from analysis import fairness_benchmark as fb
from cli import fetch_models, seed_demo
from core import db, paths
from pipeline import train_model
from tools import pad_eval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OLD_SCHEMA = """
    CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                        created_at TEXT NOT NULL);
    CREATE TABLE attendance (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             user_id INTEGER NOT NULL, timestamp TEXT NOT NULL,
                             confidence REAL, method TEXT NOT NULL DEFAULT 'lbph',
                             FOREIGN KEY (user_id) REFERENCES users(id));
"""


def old_database(users=50, days=250, per_day=4):
    """An attendance.db from before the unique index: `per_day` marks for each
    person on each of `days` days. users * days * per_day rows (50,000 by
    default), of which users * days should survive."""
    start = date(2020, 1, 1)
    rows = []
    for user in range(1, users + 1):
        for d in range(days):
            day = (start + timedelta(days=d)).isoformat()
            for k in range(per_day):
                rows.append((user, f"{day}T{8 + k:02d}:{user % 60:02d}:00", 1.0))
    with sqlite3.connect(paths.db_path()) as conn:
        conn.executescript(OLD_SCHEMA)
        conn.executemany("INSERT INTO users (id, name, created_at) VALUES (?, ?, 'x')",
                         [(u, f"u{u}") for u in range(1, users + 1)])
        conn.executemany("INSERT INTO attendance (user_id, timestamp, confidence) "
                         "VALUES (?, ?, ?)", rows)
    conn = sqlite3.connect(paths.db_path())
    try:
        return conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    finally:
        conn.close()


def count_rows():
    conn = sqlite3.connect(paths.db_path())
    try:
        return conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    finally:
        conn.close()


# ================================ 1. the one-mark-per-day migration

def test_the_migration_is_one_pass_not_one_scan_per_row(isolated_root):
    """50,000 rows took about 40 s with the correlated DELETE (it compared every
    row with every other of the same person) and a fraction of a second as one
    window pass. The bound is a generous 15 s so a slow CI runner does not
    flake, and still far under the old time."""
    assert old_database() == 50_000
    started = time.perf_counter()
    db.init_db()
    elapsed = time.perf_counter() - started
    assert elapsed < 15, f"the migration took {elapsed:.1f} s on 50,000 rows"
    assert count_rows() == 50 * 250, "one row per person per day"


def test_starters_running_at_once_on_a_big_database_all_succeed(isolated_root):
    """Two processes starting together on an old database: the first migrates
    for as long as it takes, the rest wait on the lock. Taking the write lock
    only at the DELETE let the rest sit past the 10 s busy timeout and die
    with "database is locked" -- here the same on threads."""
    old_database()
    barrier = threading.Barrier(6, timeout=60)
    errors = []

    def start():
        try:
            barrier.wait()
            db.init_db()
        except Exception as exc:                    # noqa: BLE001 - reported below
            errors.append(f"{type(exc).__name__}: {exc}")
    threads = [threading.Thread(target=start) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert errors == []
    assert count_rows() == 50 * 250
    conn = sqlite3.connect(paths.db_path())
    try:
        per_day = conn.execute("SELECT MAX(n) FROM (SELECT COUNT(*) AS n FROM attendance "
                               "GROUP BY user_id, DATE(timestamp))").fetchone()[0]
    finally:
        conn.close()
    assert per_day == 1


def test_the_migration_says_how_many_marks_it_removed_and_how_many_it_cannot_date(
        isolated_root, capsys):
    with sqlite3.connect(paths.db_path()) as conn:
        conn.executescript(OLD_SCHEMA)
        conn.execute("INSERT INTO users VALUES (1, 'Ada', 'x')")
        conn.executemany(
            "INSERT INTO attendance (user_id, timestamp) VALUES (1, ?)",
            [("2026-10-01T09:00:00",), ("2026-10-01T10:00:00",),
             ("2026-10-01T11:00:00",),                       # two removed
             ("garbage",), ("",), ("26-10-01 9am",)])        # three undatable
    db.init_db()
    err = capsys.readouterr().err
    assert "removed 2 duplicate mark(s)" in err
    assert "3 row(s) have a timestamp that cannot be dated" in err
    assert count_rows() == 4, "the undatable rows are kept, not deleted"

    db.init_db()                                              # nothing left to say
    assert capsys.readouterr().err == ""


def test_a_clean_database_migrates_silently(isolated_root, capsys):
    db.init_db()
    assert capsys.readouterr().err == ""


def test_offset_timestamps_collapse_by_their_utc_date(isolated_root):
    """Documented in core/db.py: DATE() reads an offset in UTC, so a mark written
    as 23:30+05:00 is the same day as one at 10:00 (UTC 18:30 against 10:00),
    and one at 01:00+05:00 is the day before. log_attendance never writes an
    offset, so this concerns rows imported from elsewhere."""
    with sqlite3.connect(paths.db_path()) as conn:
        conn.executescript(OLD_SCHEMA)
        conn.execute("INSERT INTO users VALUES (1, 'Ada', 'x')")
        conn.executemany(
            "INSERT INTO attendance (id, user_id, timestamp) VALUES (?, 1, ?)",
            [(1, "2026-10-01T23:30:00+05:00"),     # 2026-10-01 18:30 UTC
             (2, "2026-10-01T10:00:00"),           # the earlier of the two
             (3, "2026-10-01T01:00:00+05:00")])    # 2026-09-30 20:00 UTC: its own day
    db.init_db()
    conn = sqlite3.connect(paths.db_path())
    try:
        assert conn.execute("SELECT id FROM attendance ORDER BY id").fetchall() == [(2,), (3,)]
    finally:
        conn.close()


# ================================ 2. a folder id past LBPH's label range

@pytest.mark.parametrize("head, expected", [
    (str(2**31 - 1), 2**31 - 1),            # the largest int32 label
    (str(2**31), None),
    (str(2**63), None),                     # past SQLite's INTEGER
    ("99999999999999999999", None),
])
def test_an_id_past_int32_claims_nothing(head, expected):
    assert paths.folder_id(f"{head}_Big") == expected


def test_a_folder_with_a_huge_id_blocks_neither_registration_nor_training(isolated_root):
    """folder_ids returned the 20-digit id, add_user raised the sequence past
    SQLite's INTEGER ("Python int too large"), and the folder is still there
    for the next registration: it never recovered."""
    db.init_db()
    folder = os.path.join(paths.dataset_dir(), "99999999999999999999_Big")
    os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, "1.png"), np.full((20, 20), 128, np.uint8))

    assert paths.folder_ids() == set()
    faces, labels = train_model.load_training_data()
    assert (faces, labels) == ([], [])
    ada = db.add_user("Ada")
    assert db.get_user_name(ada) == "Ada"


# ================================ 3. fairness_benchmark

def analysed(detected=True, flags=()):
    return {"detected": detected, "flags": list(flags)}


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    (tmp_path / "part.parquet").write_bytes(b"")
    state = {}

    def load_labels(files, columns):
        labels = {c: np.asarray(state["labels"][c], np.int16) for c in columns}
        n = len(next(iter(labels.values())))
        return labels, [(files[0], 0, n)], n

    def analyse_sample(indices, offsets):
        out = fb.Results(state["result"](int(i)) for i in indices)
        out.failures = Counter(state.get("failures", {}))
        return out

    monkeypatch.setattr(fb, "load_labels", load_labels)
    monkeypatch.setattr(fb, "analyse_sample", analyse_sample)
    state["dir"] = str(tmp_path)
    return state


def run_fb(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["fairness_benchmark", *argv])
    return fb.main()


def test_nothing_analysed_is_an_error_that_says_so_not_a_crash(corpus, monkeypatch, capsys):
    """`DEFAULT_NAMES.get(col, [... vals.max() ...])` built its default eagerly,
    and max() of an empty array raised, so the default `--extra-columns gender`
    died with a ValueError traceback instead of reporting that every image
    had failed."""
    corpus["labels"] = {"race": [i % 2 for i in range(40)],
                        "gender": [i % 2 for i in range(40)]}
    corpus["result"] = lambda i: None
    corpus["failures"] = {"decode: UnidentifiedImageError": 40}
    assert run_fb(monkeypatch, "--corpus", corpus["dir"], "--per-group", "20") == 1
    out = capsys.readouterr().out
    assert "analysed 0 of 40" in out
    assert "decode: UnidentifiedImageError x40" in out


def test_a_group_with_no_analysed_image_is_warned_about(corpus, monkeypatch, capsys):
    corpus["labels"] = {"race": [i % 2 for i in range(200)]}
    corpus["result"] = lambda i: analysed() if i % 2 == 0 else None
    run_fb(monkeypatch, "--corpus", corpus["dir"], "--per-group", "100",
           "--extra-columns", "")
    out = capsys.readouterr().out
    assert "WARNING: race group 'Indian' has n=0" in out
    assert "none of its 100 sampled images" in out


def test_failures_are_reported_by_reason(corpus, monkeypatch, capsys):
    corpus["labels"] = {"race": [i % 2 for i in range(100)]}
    corpus["result"] = lambda i: analysed() if i % 10 else None
    corpus["failures"] = {"decode: OSError": 7, "analyse: RuntimeError": 3}
    run_fb(monkeypatch, "--corpus", corpus["dir"], "--per-group", "50",
           "--extra-columns", "")
    out = capsys.readouterr().out
    assert "analyse: RuntimeError x3, decode: OSError x7" in out


def test_analyse_sample_counts_decode_failures_and_names_analyser_errors(
        tmp_path, monkeypatch, capsys):
    """It was `except Exception: pass`: a corrupt file and an analyser that raises
    on every image looked the same, which is nothing at all."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("PIL")
    ok, buf = cv2.imencode(".png", np.full((4, 4, 3), 90, np.uint8))
    good = buf.tobytes()
    images = [b"not an image", good, good, b"also not an image"]
    pq.write_table(pa.table({"image": [{"bytes": b, "path": None} for b in images],
                             "race": [0, 0, 0, 0]}),
                   str(tmp_path / "a.parquet"))
    from pipeline import traits

    def analyze(bgr, **kw):
        raise RuntimeError("detector exploded")
    monkeypatch.setattr(traits, "analyze", analyze)
    _, offsets, _ = fb.load_labels([str(tmp_path / "a.parquet")], ["race"])

    out = fb.analyse_sample(np.arange(4), offsets)
    assert out == [None] * 4
    assert out.failures == {"decode: UnidentifiedImageError": 2,
                            "analyse: RuntimeError": 2}
    err = capsys.readouterr().err
    assert err.count("detector exploded") == 1, "once per reason, not once per image"


# ================================ 5. the isolated_root fixture

def test_isolated_root_lives_under_pytests_own_temp_tree(isolated_root, tmp_path_factory):
    """mkdtemp() put one directory per test straight into %TEMP%, for ever."""
    base = os.path.realpath(str(tmp_path_factory.getbasetemp()))
    assert os.path.realpath(isolated_root).startswith(base + os.sep)


# ================================ 6. pad_eval

def test_an_excel_csv_with_a_byte_order_mark_is_read(tmp_path, capsys):
    path = tmp_path / "attempts.csv"
    path.write_text("attack_type,ground_truth,verdict\nprint,attack,spoof\n"
                    "none,bona_fide,live\n", encoding="utf-8-sig", newline="")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert pad_eval.main([str(path)]) == 0
    assert "APCER print" in capsys.readouterr().out


def test_headers_with_spaces_after_the_commas_are_read(tmp_path, capsys):
    path = tmp_path / "attempts.csv"
    path.write_text("attack_type, ground_truth, verdict \nprint,attack,spoof\n",
                    encoding="utf-8")
    assert pad_eval.main([str(path)]) == 0
    assert "APCER print" in capsys.readouterr().out


def test_a_bom_in_a_string_is_dropped_by_read_rows():
    text = "﻿attack_type, ground_truth,verdict\nprint,attack,live\n"
    import io
    assert pad_eval.read_rows(io.StringIO(text)) == [("print", "attack", "live")]


def test_the_wilson_bounds_are_exactly_zero_and_one_at_the_ends():
    """The arithmetic gave 3.5e-18 for no errors in some n and 0.9999999999999999
    for all errors in others."""
    for n in range(1, 600):
        low, high = pad_eval.wilson(0, n)
        assert low == 0.0 and 0.0 < high < 1.0, (n, low, high)
        low, high = pad_eval.wilson(n, n)
        assert high == 1.0 and 0.0 < low < 1.0, (n, low, high)
    for n in (1, 2, 7, 40):
        for k in range(n + 1):
            low, high = pad_eval.wilson(k, n)
            assert 0.0 <= low <= k / n <= high <= 1.0


# ================================ 7. seed_demo --keep --names

NAMES = ["Ada_Lovelace", "Grace_Hopper", "Alan_Turing"]
LABELS = [0, 1, 0, 2, 0, 1, 0, 1, 2]


def test_keep_with_names_skips_somebody_already_enrolled(capsys):
    picked = seed_demo.pick_people(LABELS, NAMES, 1, None,
                                   requested=["Ada_Lovelace", "grace hopper"],
                                   exclude=["Ada Lovelace"])
    assert [lab for lab, _ in picked] == [1]
    assert "Ada_Lovelace: already enrolled, skipped" in capsys.readouterr().out


def test_requested_names_without_exclude_are_all_picked(capsys):
    picked = seed_demo.pick_people(LABELS, NAMES, 1, None,
                                   requested=["Ada_Lovelace", "Grace_Hopper"])
    assert [lab for lab, _ in picked] == [0, 1]
    assert "already enrolled" not in capsys.readouterr().out


# ================================ 9. fetch_models --verify

@pytest.fixture
def weights(monkeypatch, tmp_path):
    """Every pinned file present, with pins that match what is on disk."""
    folder = tmp_path / "models"
    folder.mkdir()
    monkeypatch.setattr(paths, "models_dir", lambda: str(folder))
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("--verify reached the network"))
    pins = {}
    for name in fetch_models.URLS:
        body = name.encode() * 50
        (folder / name).write_bytes(body)
        pins[name] = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(fetch_models, "SHA256", pins)
    return folder


def test_verify_passes_files_that_match_their_pins(weights, monkeypatch, capsys):
    assert fetch_models.verify() == []
    monkeypatch.setattr(sys, "argv", ["fetch_models", "--verify"])
    assert fetch_models.main() == 0
    assert "Every model matches" in capsys.readouterr().out


def test_verify_fails_on_a_file_that_is_not_what_is_pinned(weights, monkeypatch, capsys):
    """What a cache hit never checked: the fetch only sees that the file is there."""
    name = "lbfmodel.yaml"
    (weights / name).write_bytes(b"a poisoned cache entry")
    bad = fetch_models.verify()
    assert [n for n, _ in bad] == [name] and "checksum" in bad[0][1]
    monkeypatch.setattr(sys, "argv", ["fetch_models", "--verify"])
    assert fetch_models.main() == 1
    assert f"MISMATCH  {name}" in capsys.readouterr().out
    assert (weights / name).read_bytes() == b"a poisoned cache entry", "verify repairs nothing"


def test_verify_fails_on_a_missing_file(weights, monkeypatch, capsys):
    (weights / "age_net.caffemodel").unlink()
    assert fetch_models.verify() == [("age_net.caffemodel", "missing")]
    monkeypatch.setattr(sys, "argv", ["fetch_models", "--verify"])
    assert fetch_models.main() == 1


def test_the_weights_workflow_verifies_after_restoring_the_cache():
    text = open(os.path.join(ROOT, ".github", "workflows", "coverage-with-weights.yml"),
                encoding="utf-8").read()
    assert "python -m cli.fetch_models --verify" in text
    assert text.index("cache/restore") < text.index("fetch_models --verify") \
        < text.index("cache/save")


# ================================ 10. reproducible property tests

def test_properties_use_a_derandomized_profile_in_ci():
    pytest.importorskip("hypothesis")
    code = ("import importlib.util, sys; from hypothesis import settings; "
            "spec = importlib.util.spec_from_file_location('props', sys.argv[1]); "
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
            "s = settings.default; print(s.derandomize, s.print_blob)")
    path = os.path.join(ROOT, "tests", "test_properties.py")

    def profile(**env):
        # GITHUB_ACTIONS too: it is what Hypothesis's own CI detection reads on a runner.
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("CI", "HYPOTHESIS_PROFILE", "GITHUB_ACTIONS")}
        clean.update(env)
        got = subprocess.run([sys.executable, "-c", code, path], env=clean, cwd=ROOT,
                             capture_output=True, text=True)
        assert got.returncode == 0, got.stderr
        return got.stdout.split()[-2:]
    assert profile(CI="true") == ["True", "True"]
    assert profile()[0] == "False", "locally the search is still free to roam"
    assert profile()[1] == "True", "and a failure still prints the blob"
