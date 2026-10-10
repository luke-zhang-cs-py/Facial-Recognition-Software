"""Attendance decided by SFace, with LBPH only as the fallback.

On held-out LFW photos SFace named 15/16 correctly, named nobody wrongly and
accepted no strangers; LBPH at its threshold got 9/16, named 5 people wrongly
and accepted 12 of 16 strangers (notes/CODE_AUDIT_2026-10.md). So the web
camera and the CLI both decide by SFace whenever the weights are present.

Embedding is replaced at its boundary (traits.embed) so these run without
weights or a face; the decision after it -- calibrated threshold, margin
over the runner-up, liveness, one mark a day, the method recorded with the
number -- is the real code.
"""
import os
import sqlite3

import cv2
import numpy as np
import pytest

from cli import attendance
from core import db as core_db
from pipeline import camera, decision, liveness, recognition, traits

W, H = 640, 480
# YuNet order: box, right eye, left eye, nose, right mouth, left mouth, score.
# "Right eye" is the subject's, so it sits at the smaller x on the image.
ROW = np.array([200, 150, 200, 200, 250, 210, 350, 210, 300, 260,
                265, 310, 335, 310, 0.99], np.float32)


def unit(*v):
    v = np.array(v + (0.0,) * (128 - len(v)), np.float32)
    return v / np.linalg.norm(v)


ADA, BEN = unit(1.0), unit(0.0, 1.0)


def frame():
    rng = np.random.RandomState(5)
    f = rng.randint(0, 255, (H, W, 3)).astype(np.uint8)
    f[:, : W // 3] //= 3                 # not left-right symmetric
    return f


@pytest.fixture
def people(isolated_db):
    ada, ben = isolated_db.add_user("Ada"), isolated_db.add_user("Ben")
    gallery = {ada: {"name": "Ada", "centroid": ADA, "samples": 30},
               ben: {"name": "Ben", "centroid": BEN, "samples": 30}}
    return ada, ben, gallery


def rows_logged():
    with core_db.connection() as conn:
        return conn.execute(
            "SELECT u.name, a.confidence, a.method FROM attendance a "
            "JOIN users u ON u.id = a.user_id").fetchall()


# ------------------------------------------------------------- the decision

def test_sface_names_the_closest_person_above_the_threshold(people, monkeypatch):
    ada, _, gallery = people
    monkeypatch.setattr(traits, "embed", lambda bgr, row=None: ADA)
    user_id, similarity, accepted = decision.identify(frame(), ROW, gallery)
    assert (user_id, accepted) == (ada, True) and similarity == pytest.approx(1.0)


def test_sface_refuses_a_stranger_and_a_too_close_call(people, monkeypatch):
    _, _, gallery = people
    monkeypatch.setattr(traits, "embed", lambda bgr, row=None: unit(0.0, 0.0, 1.0))
    assert decision.identify(frame(), ROW, gallery)[2] is False
    # Half-way between Ada and Ben: well above the threshold for both, and
    # told apart by nothing -- a coin toss is not an identification.
    monkeypatch.setattr(traits, "embed", lambda bgr, row=None: unit(1.0, 1.0))
    assert decision.identify(frame(), ROW, gallery)[2] is False


def test_a_face_that_cannot_be_embedded_is_nobody(people, monkeypatch):
    monkeypatch.setattr(traits, "embed", lambda bgr, row=None: None)
    assert decision.identify(frame(), ROW, people[2]) == (None, None, False)


def test_unmirroring_a_row_puts_the_face_back_where_the_camera_saw_it():
    mirrored = decision.unmirror_row(ROW, W)      # the same maths both ways
    assert mirrored[0] == W - 200 - 200
    assert mirrored[4] < mirrored[6], "right eye must stay on the image left"
    assert mirrored[10] < mirrored[12], "and so must the right mouth corner"
    np.testing.assert_allclose(decision.unmirror_row(mirrored, W), ROW)


def test_without_the_weights_attendance_falls_back_to_lbph_and_says_why(
        isolated_db, monkeypatch):
    monkeypatch.setattr(decision.facemodels, "have", lambda name: False)
    assert decision.sface_gallery() is None
    assert "fetch_models" in decision.lbph_reason()
    monkeypatch.setattr(decision.facemodels, "have", lambda name: True)
    monkeypatch.setattr(recognition, "refresh_gallery", lambda: 0)
    assert decision.sface_gallery() is None, "nobody has an embedding yet"
    assert "embedding" in decision.lbph_reason()


# ------------------------------------------------------------- the camera

@pytest.fixture
def sface_camera(people, monkeypatch):
    ada, ben, gallery = people
    m = camera.CameraManager(capture_factory=lambda: None)
    monkeypatch.setattr(camera.facelandmarks, "fit", lambda gray, box: None)
    monkeypatch.setattr(camera.facetraits, "detect", lambda f: [ROW.copy()])
    monkeypatch.setattr(m._liveness, "verdict", lambda: "live")
    monkeypatch.setattr(camera.faceliveness, "score", lambda f, b: 0.9)
    m._mode, m._gallery, m._recognizer = camera.MODE_ATTENDANCE, gallery, None
    return m, ada


def test_the_camera_marks_by_sface_and_records_the_method(sface_camera, monkeypatch):
    m, ada = sface_camera
    seen = []
    monkeypatch.setattr(traits, "embed", lambda bgr, row=None: seen.append((bgr.copy(), row)) or ADA)
    shown = frame()
    m._handle_attendance(shown.copy())
    assert rows_logged() == [("Ada", pytest.approx(1.0), "sface")]
    # Embedded from the unmirrored frame, by the unmirrored row.
    bgr, row = seen[0]
    np.testing.assert_array_equal(bgr, cv2.flip(shown, 1))
    np.testing.assert_allclose(row, decision.unmirror_row(ROW, W))


def test_the_camera_never_marks_a_stranger_by_sface(sface_camera, monkeypatch):
    m, _ = sface_camera
    monkeypatch.setattr(traits, "embed", lambda bgr, row=None: unit(0.0, 0.0, 1.0))
    m._handle_attendance(frame())
    assert rows_logged() == []


def test_the_camera_still_refuses_a_photograph_by_sface(sface_camera, monkeypatch):
    m, _ = sface_camera
    monkeypatch.setattr(m._liveness, "verdict", lambda: "spoof")
    monkeypatch.setattr(traits, "embed", lambda bgr, row=None: ADA)
    m._handle_attendance(frame())
    assert rows_logged() == []


def test_starting_attendance_says_which_recogniser_decides(people, monkeypatch, isolated_root):
    _, _, gallery = people
    m = camera.CameraManager(capture_factory=lambda: None)
    monkeypatch.setattr(m, "start", lambda: None)
    monkeypatch.setattr(m, "_load_recognizer", lambda: None)
    open(os.path.join(isolated_root, "trainer.yml"), "w").close()
    monkeypatch.setattr(camera.paths, "model_path", lambda: os.path.join(isolated_root, "trainer.yml"))
    core_db.add_user("Cy")                               # enrolled, never embedded
    monkeypatch.setattr(decision, "sface_gallery", lambda: gallery)
    m.start_attendance()
    events = " | ".join(e["message"] for e in m.status()["events"])
    assert "SFace: 2 people" in events and "Cy" in events


# ------------------------------------------------------------- the CLI

def test_the_cli_marks_by_sface_once(people, monkeypatch, capsys):
    ada, _, gallery = people
    for name, fn in (("imshow", lambda *a: None), ("waitKey", lambda *a: -1),
                     ("destroyAllWindows", lambda: None)):
        monkeypatch.setattr(cv2, name, fn)
    monkeypatch.setattr(attendance, "_load_recognizer", lambda: object())
    monkeypatch.setattr(decision, "sface_gallery", lambda: gallery)
    monkeypatch.setattr(traits, "detect", lambda bgr: [ROW.copy()])
    monkeypatch.setattr(traits, "embed", lambda bgr, row=None: ADA)
    monkeypatch.setattr(liveness, "available", lambda: True)
    monkeypatch.setattr(liveness, "score", lambda bgr, box: 0.9)
    frames = [frame() for _ in range(liveness.VOTE_WINDOW + 3)]

    class Cap:
        def read(self):
            return (True, frames.pop()) if frames else (False, None)

        def release(self):
            pass
    monkeypatch.setattr(attendance, "_open_camera", Cap)
    attendance.run_attendance()
    assert rows_logged() == [("Ada", pytest.approx(1.0), "sface")]
    assert "similarity 1.000" in capsys.readouterr().out


# ------------------------------------------------------------- the gallery

def test_refresh_embeds_enrolled_samples_once_and_skips_orphan_folders(
        isolated_db, monkeypatch):
    from analysis import analytics
    ada = isolated_db.add_user("Ada")
    root = isolated_db.paths.dataset_dir()
    for folder in (f"{ada}_Ada", "99_Gone"):
        os.makedirs(os.path.join(root, folder))
        cv2.imwrite(os.path.join(root, folder, "1.jpg"), np.zeros((8, 8), np.uint8))
    done = {}
    def analyze_sample(uid, path, use_cache=True):
        done.setdefault(path, uid)
        return {"embedding": np.ones(4, np.float32)}
    monkeypatch.setattr(analytics, "analyze_sample", analyze_sample)
    monkeypatch.setattr(isolated_db, "get_cached_traits",
                        lambda path, mtime: {"embedding": b"cached" * 4} if path in done else None)
    assert recognition.refresh_gallery() == 1
    assert list(done.values()) == [ada]
    assert recognition.refresh_gallery() == 0, "an embedded sample is not redone"


# ------------------------------------------------------------- the record

def test_an_old_database_gains_the_method_column_and_reads_as_lbph(isolated_root):
    path = core_db.paths.db_path()
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL, confidence REAL, FOREIGN KEY (user_id) REFERENCES users(id));
        INSERT INTO users (name, created_at) VALUES ('Ada', '2026-09-01');
        INSERT INTO attendance (user_id, timestamp, confidence) VALUES (1, '2026-09-01T09:00:00', 41.5);
    """)
    conn.commit()
    conn.close()
    core_db.init_db()
    core_db.init_db()                                    # and again: idempotent
    assert core_db.get_all_attendance_with_method() == [("Ada", "2026-09-01T09:00:00", 41.5, "lbph")]


def test_the_report_says_which_way_each_number_runs(isolated_db, capsys):
    from cli import view_report
    ada, ben = isolated_db.add_user("Ada"), isolated_db.add_user("Ben")
    isolated_db.log_attendance(ada, 0.712, "sface")
    isolated_db.log_attendance(ben, 41.5)
    view_report.main()
    out = capsys.readouterr().out
    assert "similarity=0.712 (SFace)" in out and "distance=41.5 (LBPH)" in out


def test_the_cli_runs_by_sface_without_a_trained_lbph_model(people, monkeypatch, capsys):
    """SFace never reads trainer.yml, but the CLI asked for it first and
    turned away everybody who had registered without running train_model
    (notes/CODE_AUDIT.md, 2026-10-05). The web camera never had this."""
    _, _, gallery = people
    for name, fn in (("imshow", lambda *a: None), ("waitKey", lambda *a: -1),
                     ("destroyAllWindows", lambda: None)):
        monkeypatch.setattr(cv2, name, fn)
    monkeypatch.setattr(attendance, "_load_recognizer", lambda: None)   # no trainer.yml
    monkeypatch.setattr(decision, "sface_gallery", lambda: gallery)
    monkeypatch.setattr(traits, "detect", lambda bgr: [ROW.copy()])
    monkeypatch.setattr(traits, "embed", lambda bgr, row=None: ADA)
    monkeypatch.setattr(liveness, "available", lambda: True)
    monkeypatch.setattr(liveness, "score", lambda bgr, box: 0.9)
    frames = [frame() for _ in range(liveness.VOTE_WINDOW)]

    class Cap:
        def read(self):
            return (True, frames.pop()) if frames else (False, None)

        def release(self):
            pass
    monkeypatch.setattr(attendance, "_open_camera", Cap)
    attendance.run_attendance()
    assert rows_logged() == [("Ada", pytest.approx(1.0), "sface")]
