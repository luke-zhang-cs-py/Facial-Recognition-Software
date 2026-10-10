"""cli/seed_demo.py against a tiny LFW-shaped corpus built in tmp_path.

The "LFW" here is a parquet file of a few geometric PNGs (grey squares with
a bright block), labelled with made-up names, written at test time. The
detector is traits.detect replaced with a fixed box, so no image is ever
asked to look like a face. Nothing is downloaded.

pyarrow and Pillow are not in requirements.txt -- seed_demo needs them, CI
does not install them -- so the tests that read a real parquet file or
decode a real PNG skip without them. Everything else (the choice of people,
the budget, the dispatch in main, removal) runs everywhere, with the
parquet reader and the decoder replaced.
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest

from cli import seed_demo
from core import db, paths
from pipeline import traits

NAMES = ["Ada_Lovelace", "Grace_Hopper", "Alan_Turing", "Edsger_Dijkstra"]
# Ada x5, Grace x4, Alan x3, Edsger x1
LABELS = [0, 1, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 3]


def png(value):
    img = np.full((60, 60, 3), value, np.uint8)
    img[15:45, 15:45] = 255 - value
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


IMAGES = [{"bytes": png(10 + 15 * i)} for i in range(len(LABELS))]


def a_box(monkeypatch, box=(15, 15, 30, 30)):
    row = np.zeros(15, np.float32)
    row[:4] = box
    monkeypatch.setattr(traits, "detect", lambda bgr: [row])


# ------------------------------------------------------------ pick_people

def test_most_photographed_first_and_floor_on_depth():
    picked = seed_demo.pick_people(LABELS, NAMES, 3, None)
    assert [lab for lab, _ in picked] == [0, 1, 2]
    assert picked[0][1] == [0, 2, 4, 6, 9]
    assert seed_demo.pick_people(LABELS, NAMES, 3, 1) == [picked[0]]


def test_requested_names_ignore_the_floor_and_report_unknowns(capsys):
    picked = seed_demo.pick_people(LABELS, NAMES, 50, 2,
                                   requested=["edsger dijkstra", "Nobody"])
    assert picked == [(3, [12])]
    assert "!! not in LFW: Nobody" in capsys.readouterr().out


def test_requested_names_without_a_name_table_find_nobody(capsys):
    assert seed_demo.pick_people(LABELS, None, 1, 2, requested=["Ada"]) == []
    assert "not in LFW: Ada" in capsys.readouterr().out


def test_a_label_with_no_images_is_not_requested(capsys):
    names = NAMES + ["Ghost_Person"]
    assert seed_demo.pick_people(LABELS, names, 1, 2,
                                 requested=["Ghost Person"]) == []


def test_exclude_needs_names_to_compare_against():
    every = seed_demo.pick_people(LABELS, None, 1, None, exclude=["Ada"])
    assert len(every) == 4


# ------------------------------------------------------ budget and report

@pytest.mark.parametrize("available, wanted, budget", [
    (20, 12, 12), (12, 12, 12), (8, 12, 7), (3, 12, 3), (2, 12, 3), (1, 2, 2),
])
def test_sample_budget(available, wanted, budget):
    assert seed_demo.sample_budget(available, wanted) == budget


def test_should_report():
    assert seed_demo.should_report(0, 5, 999, 1) is True
    big = seed_demo.SMALL_RUN + 1
    assert seed_demo.should_report(big - 1, big, 7, 1) is True, "the last one"
    assert seed_demo.should_report(3, big, 405, 12) is True, "crossed 400"
    assert seed_demo.should_report(3, big, 420, 12) is False


# -------------------------------------------------------- removal / listing

def test_remove_all_with_nothing_to_remove(isolated_db, capsys):
    assert seed_demo.remove_all() == 0
    assert "no demo entries to remove" in capsys.readouterr().out


def test_remove_all_takes_rows_and_folders_and_spares_real_people(
        isolated_db, capsys):
    real = db.add_user("Real Person")
    demo = db.add_user(seed_demo.PREFIX + "Ada Lovelace")
    os.makedirs(paths.user_folder(real, "Real Person"))
    os.makedirs(paths.user_folder(demo, "Ada Lovelace"))
    assert seed_demo.remove_all() == 1
    assert db.get_all_users() == [(real, "Real Person")]
    assert os.listdir(paths.dataset_dir()) == [f"{real}_Real_Person"]
    assert "removed [demo] Ada Lovelace" in capsys.readouterr().out


def test_remove_all_without_a_dataset_folder(isolated_db):
    db.add_user(seed_demo.PREFIX + "Ada")
    assert seed_demo.remove_all() == 1 and db.get_all_users() == []


@pytest.mark.parametrize("n, heading", [(0, "0 demo entries:"),
                                        (1, "1 demo entry:"),
                                        (2, "2 demo entries:")])
def test_list_demo_users(isolated_db, capsys, n, heading):
    db.add_user("Not A Demo")
    for i in range(n):
        db.add_user(f"{seed_demo.PREFIX}P{i}")
    seed_demo.list_demo_users()
    out = capsys.readouterr().out
    assert out.splitlines()[0] == heading
    assert "Not A Demo" not in out


# ----------------------------------------------------------- enroll_person

def test_enroll_person_writes_crops_up_to_the_budget(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    a_box(monkeypatch)
    images = [IMAGES[0], IMAGES[1]["bytes"], IMAGES[2]]   # dict or raw bytes
    kept = seed_demo.enroll_person(str(tmp_path), [0, 1, 2], images, 2)
    assert kept == 2
    assert sorted(os.listdir(tmp_path)) == ["1.jpg", "2.jpg"]
    crop = cv2.imread(str(tmp_path / "1.jpg"), cv2.IMREAD_GRAYSCALE)
    assert crop.shape == seed_demo.decision.vision.LBPH_INPUT_SIZE[::-1]
    assert crop.mean() > 200, "the crop is the bright block the box covers"


def test_enroll_person_skips_images_with_no_face_or_no_crop(tmp_path,
                                                            monkeypatch):
    pytest.importorskip("PIL")
    answers = [[], [np.r_[[500, 500, 10, 10], np.zeros(11)].astype(np.float32)]]
    monkeypatch.setattr(traits, "detect",
                        lambda bgr: answers.pop(0) if answers else [])
    assert seed_demo.enroll_person(str(tmp_path), [0, 1, 2], IMAGES, 5) == 0
    assert os.listdir(tmp_path) == []


# --------------------------------------------------------------- load_lfw

def write_lfw(path, with_names=True):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"label": LABELS,
                      "image": [{"bytes": im["bytes"], "path": None}
                                for im in IMAGES]})
    if with_names:
        meta = {"info": {"features": {"label": {"names": NAMES}}}}
        table = table.replace_schema_metadata(
            {"huggingface": json.dumps(meta)})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)


@pytest.fixture
def no_pyarrow(monkeypatch):
    """A fresh install: pyarrow is not in requirements.txt. A None entry in
    sys.modules makes the import raise ImportError, installed or not."""
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)


def test_load_lfw_without_the_corpus_says_how_to_get_it(tmp_path, monkeypatch,
                                                        capsys, no_pyarrow):
    """Regression: pyarrow was imported before the corpus check, so a fresh
    install without it got ImportError instead of these instructions."""
    missing = str(tmp_path / "lfw" / "lfw.parquet")
    monkeypatch.setattr(seed_demo, "LFW", missing)
    assert seed_demo.load_lfw(3, 2) is None
    out = capsys.readouterr().out
    assert f"LFW parquet not found at {missing}" in out
    assert "curl -L https://huggingface.co" in out


def test_load_lfw_with_the_corpus_but_no_pyarrow_says_what_to_install(
        tmp_path, monkeypatch, capsys, no_pyarrow):
    present = tmp_path / "lfw.parquet"
    present.write_bytes(b"not read")
    monkeypatch.setattr(seed_demo, "LFW", str(present))
    assert seed_demo.load_lfw(3, 2) is None
    assert "pip install pyarrow" in capsys.readouterr().out


def test_load_lfw_reads_names_labels_and_images(tmp_path, monkeypatch):
    path = str(tmp_path / "lfw" / "lfw.parquet")
    write_lfw(path)
    monkeypatch.setattr(seed_demo, "LFW", path)
    names, images, picked = seed_demo.load_lfw(3, 2)
    assert names == NAMES
    assert images[0]["bytes"] == IMAGES[0]["bytes"]
    assert [lab for lab, _ in picked] == [0, 1]


def test_load_lfw_without_label_names(tmp_path, monkeypatch):
    path = str(tmp_path / "lfw.parquet")
    write_lfw(path, with_names=False)
    monkeypatch.setattr(seed_demo, "LFW", path)
    names, _, picked = seed_demo.load_lfw(4, None)
    assert names is None and [lab for lab, _ in picked] == [0, 1]


# ------------------------------------------------------------------- main

@pytest.fixture
def corpus(isolated_db, monkeypatch):
    """main() over the in-memory corpus: load_lfw keeps its real choice of
    people (pick_people) but reads LABELS instead of a parquet file, and
    enroll_person writes one placeholder file per budgeted sample."""
    loads, enrolled = [], []

    def load_lfw(min_images, wanted, requested=None, exclude=()):
        loads.append({"min": min_images, "wanted": wanted,
                      "requested": requested, "exclude": list(exclude)})
        return NAMES, IMAGES, seed_demo.pick_people(
            LABELS, NAMES, min_images, wanted, requested, exclude)

    def enroll_person(folder, idxs, images, budget):
        enrolled.append((os.path.basename(folder), budget))
        for i in range(budget):
            open(os.path.join(folder, f"{i + 1}.jpg"), "wb").close()
        return budget

    monkeypatch.setattr(seed_demo, "load_lfw", load_lfw)
    monkeypatch.setattr(seed_demo, "enroll_person", enroll_person)
    from pipeline import recognition, train_model
    trained = []
    monkeypatch.setattr(recognition, "refresh_gallery", lambda: 0)
    monkeypatch.setattr(train_model, "train", lambda: trained.append(1))
    return {"loads": loads, "enrolled": enrolled, "trained": trained}


def seed(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["seed_demo", *argv])
    return seed_demo.main()


def demo_names():
    return sorted(n for _, n in db.get_all_users())


def test_default_seed_enrolls_the_deepest_people_and_trains(corpus,
                                                            monkeypatch,
                                                            capsys):
    assert seed(monkeypatch, "--people", "2", "--samples", "4") == 0
    assert demo_names() == ["[demo] Ada Lovelace", "[demo] Grace Hopper"]
    assert corpus["loads"] == [{"min": 4, "wanted": 2, "requested": [],
                                "exclude": []}]
    assert corpus["trained"] == [1]
    out = capsys.readouterr().out
    assert "8 images written" in out


def test_reseeding_replaces_rather_than_duplicates(corpus, monkeypatch):
    seed(monkeypatch, "--people", "2", "--samples", "3")
    seed(monkeypatch, "--people", "1", "--samples", "3")
    assert demo_names() == ["[demo] Ada Lovelace"]


def test_keep_adds_only_people_not_yet_enrolled(corpus, monkeypatch):
    seed(monkeypatch, "--people", "1", "--samples", "3")
    seed(monkeypatch, "--keep", "--people", "1", "--samples", "3")
    assert demo_names() == ["[demo] Ada Lovelace", "[demo] Grace Hopper"]
    assert corpus["loads"][1]["exclude"] == ["Ada Lovelace"]


def test_keep_with_all_with_does_not_enroll_anyone_twice(corpus, monkeypatch):
    """--keep promises to add to what is enrolled. On the --all-with path
    the exclusion was never passed, so every person already enrolled was
    enrolled again under a second id -- one face split across two users,
    the bug pick_people's `exclude` was written to stop."""
    seed(monkeypatch, "--people", "1", "--samples", "3")
    seed(monkeypatch, "--keep", "--all-with", "3")
    names = demo_names()
    assert names == ["[demo] Ada Lovelace", "[demo] Alan Turing",
                     "[demo] Grace Hopper"]
    assert len(names) == len(set(names))
    assert corpus["loads"][1]["exclude"] == ["Ada Lovelace"]


def test_all_with_enrolls_everyone_deep_enough(corpus, monkeypatch):
    seed(monkeypatch, "--all-with", "4", "--samples", "10")
    assert demo_names() == ["[demo] Ada Lovelace", "[demo] Grace Hopper"]
    assert corpus["loads"][0]["wanted"] is None
    # Grace has 4 images and a budget of 10: one is held back to test with.
    assert sorted(corpus["enrolled"])[1][1] == 3


def test_explicit_names(corpus, monkeypatch, capsys):
    seed(monkeypatch, "--names", "Edsger_Dijkstra, ,Nobody")
    assert demo_names() == ["[demo] Edsger Dijkstra"]
    assert corpus["loads"][0]["requested"] == ["Edsger_Dijkstra", "Nobody"]


def test_list_and_remove_change_nothing_else(corpus, monkeypatch, capsys):
    seed(monkeypatch, "--people", "1", "--samples", "3")
    capsys.readouterr()
    assert seed(monkeypatch, "--list") == 0
    assert "1 demo entry:" in capsys.readouterr().out
    assert seed(monkeypatch, "--remove") == 0
    assert db.get_all_users() == []
    assert len(corpus["loads"]) == 1, "--list and --remove load nothing"


def test_a_missing_corpus_fails_without_training(corpus, monkeypatch):
    monkeypatch.setattr(seed_demo, "load_lfw", lambda *a, **k: None)
    assert seed(monkeypatch) == 1
    assert corpus["trained"] == []


def test_enroll_all_without_names_uses_the_label(isolated_db, monkeypatch,
                                                 capsys):
    monkeypatch.setattr(seed_demo, "enroll_person",
                        lambda folder, idxs, images, budget: 1)
    total = seed_demo.enroll_all([(7, [0, 1])], None, IMAGES, 5)
    assert total == 1 and demo_names() == ["[demo] 7"]
    assert "[demo] 7" in capsys.readouterr().out


def test_end_to_end_from_a_parquet_file(isolated_db, tmp_path, monkeypatch):
    """The real reader and the real decoder, still with a fixed detector."""
    pytest.importorskip("PIL")
    path = str(tmp_path / "lfw" / "lfw.parquet")
    write_lfw(path)
    monkeypatch.setattr(seed_demo, "LFW", path)
    a_box(monkeypatch)
    from pipeline import recognition, train_model
    monkeypatch.setattr(recognition, "refresh_gallery", lambda: 0)
    monkeypatch.setattr(train_model, "train", lambda: None)
    assert seed(monkeypatch, "--people", "2", "--samples", "3") == 0
    folders = sorted(os.listdir(paths.dataset_dir()))
    assert folders == ["1_Ada_Lovelace", "2_Grace_Hopper"]
    for folder in folders:
        assert len(os.listdir(os.path.join(paths.dataset_dir(), folder))) == 3


def test_a_large_run_reports_progress_sparingly(isolated_db, monkeypatch,
                                                capsys):
    monkeypatch.setattr(seed_demo, "enroll_person",
                        lambda folder, idxs, images, budget: 1)
    people = [(i, [0]) for i in range(seed_demo.SMALL_RUN + 5)]
    total = seed_demo.enroll_all(people, None, IMAGES, 1)
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "images so far" in ln]
    assert total == len(people)
    assert lines == [f"  {'[demo] ' + str(len(people) - 1):<34} 1 samples   "
                     f"({len(people)} images so far)"], "only the last person"
