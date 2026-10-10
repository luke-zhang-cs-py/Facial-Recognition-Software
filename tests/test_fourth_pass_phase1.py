"""The third audit's open items (notes/CODE_AUDIT.md, "Left for later").

1. One per-frame decision. The web camera and `python -m cli.attendance` each
   ran their own copy of "score, identify, vote, mark", so the liveness-vote
   fixes had to be made twice. decision.decide() is that sequence once.
2. One mark per person per day, held by the database. log_attendance checked
   and then inserted on two connections, so two processes marking the same
   person at the same instant both found nothing and both wrote a row.
3. One rule for a dataset/ folder's id. train_model parsed it with int(),
   which takes " 5" and "-1"; paths.folder_ids used isdigit(), which does not.

No face images: the frames are noise, the detector and the recogniser are
stand-ins, and the database is a temporary file.
"""
import os
import sqlite3
import threading

import cv2
import numpy as np
import pytest

from core import db, paths
from pipeline import camera, decision, liveness, train_model, traits

W, H = 640, 480
BOX = (200, 150, 200, 200)


def noise_frame():
    rng = np.random.RandomState(7)
    return cv2.GaussianBlur(rng.randint(0, 255, (H, W, 3)).astype(np.uint8), (3, 3), 0)


def yunet_row(x, y, w, h):
    cx = x + w / 2
    return np.array([x, y, w, h, cx - 50, y + 60, cx + 50, y + 60, cx, y + 110,
                     cx - 35, y + 160, cx + 35, y + 160, 0.99], np.float32)


class Recogniser:
    """An LBPH stand-in: always this id at this distance."""

    def __init__(self, user_id, distance=10.0):
        self.user_id, self.distance, self.calls = user_id, distance, 0

    def predict(self, face):
        self.calls += 1
        return self.user_id, self.distance


def one_frame_vote():
    """A real vote that decides on a single frame, so one call is a verdict."""
    return liveness.LivenessVote(window=1, required=1)


@pytest.fixture
def scored(monkeypatch):
    """Set the liveness score every frame gets."""
    def set_score(value):
        monkeypatch.setattr(liveness, "score", lambda bgr, box: value)
    set_score(0.9)
    return set_score


# ======================================================= 1. decision.decide

def test_decide_logs_a_live_recognised_face(isolated_db, scored):
    ada = isolated_db.add_user("Ada")
    got = decision.decide(noise_frame(), BOX, None, one_frame_vote(),
                          recognizer=Recogniser(ada))
    assert got.outcome == decision.LOGGED
    assert (got.user_id, got.accepted, got.method, got.verdict) == (ada, True, decision.LBPH, "live")
    assert [name for name, _, _ in isolated_db.get_attendance_for_today()] == ["Ada"]


def test_decide_says_already_when_today_is_marked(isolated_db, scored):
    ada = isolated_db.add_user("Ada")
    isolated_db.log_attendance(ada, 12.0)
    got = decision.decide(noise_frame(), BOX, None, one_frame_vote(),
                          recognizer=Recogniser(ada))
    assert got.outcome == decision.ALREADY
    assert len(isolated_db.get_attendance_for_today()) == 1


def test_decide_says_no_user_for_a_label_with_no_user_row(isolated_db, scored):
    got = decision.decide(noise_frame(), BOX, None, one_frame_vote(),
                          recognizer=Recogniser(9999))
    assert got.outcome == decision.NO_USER
    assert isolated_db.get_attendance_for_today() == []


@pytest.mark.parametrize("score, verdict", [(0.01, "spoof"), (None, "unknown")])
def test_decide_marks_nobody_who_is_not_live(isolated_db, scored, score, verdict):
    ada = isolated_db.add_user("Ada")
    scored(score)
    got = decision.decide(noise_frame(), BOX, None, one_frame_vote(),
                          recognizer=Recogniser(ada))
    assert (got.outcome, got.verdict, got.accepted) == (decision.NOT_LIVE, verdict, True)
    assert isolated_db.get_attendance_for_today() == []


def test_decide_with_no_face_forgets_the_vote(isolated_db, scored):
    vote = liveness.LivenessVote()
    vote.follow(1)
    for _ in range(liveness.VOTE_WINDOW):
        vote.push(0.9)
    got = decision.decide(noise_frame(), None, None, vote, recognizer=Recogniser(1))
    assert got.outcome == decision.NO_FACE
    assert vote.samples == 0 and vote.verdict() == "unknown"


def test_decide_marks_nobody_it_does_not_recognise(isolated_db, scored):
    ada = isolated_db.add_user("Ada")
    vote = one_frame_vote()
    vote.follow(ada)
    stranger = Recogniser(ada, distance=decision.CONFIDENCE_THRESHOLD + 5)
    got = decision.decide(noise_frame(), BOX, None, vote, recognizer=stranger)
    assert (got.outcome, got.accepted) == (decision.UNRECOGNISED, False)
    assert isolated_db.get_attendance_for_today() == []
    assert vote._who == ada, "a frame recognising nobody must keep the vote"


def test_decide_does_not_write_twice_in_one_session(isolated_db, scored, monkeypatch):
    ada = isolated_db.add_user("Ada")
    writes = []
    monkeypatch.setattr(decision, "record", lambda *a: writes.append(a) or decision.LOGGED)
    got = decision.decide(noise_frame(), BOX, None, one_frame_vote(),
                          recognizer=Recogniser(ada), first_mark=lambda uid: False)
    assert got.outcome == decision.SEEN and writes == []


def test_decide_skips_a_face_with_no_pixels(isolated_db, scored):
    rec = Recogniser(isolated_db.add_user("Ada"))
    vote = one_frame_vote()
    got = decision.decide(np.zeros((40, 40, 3), np.uint8), BOX, None, vote, recognizer=rec)
    assert got.outcome == decision.UNREADABLE
    assert rec.calls == 0 and vote.samples == 0


def test_decide_embeds_a_mirrored_face_the_way_the_camera_saw_it(isolated_db, scored,
                                                                monkeypatch):
    ada = isolated_db.add_user("Ada")
    seen = []
    monkeypatch.setattr(decision, "identify",
                        lambda bgr, row, gal: seen.append((bgr, row)) or (ada, 0.9, True))
    frame, row = noise_frame(), yunet_row(*BOX)
    got = decision.decide(frame, BOX, row, one_frame_vote(), gallery={ada: None},
                          mirrored=True)
    assert (got.outcome, got.method) == (decision.LOGGED, decision.SFACE)
    np.testing.assert_array_equal(seen[0][0], cv2.flip(frame, 1))
    np.testing.assert_allclose(seen[0][1], decision.unmirror_row(row, W))


def test_both_front_ends_decide_through_the_same_function(isolated_db, monkeypatch):
    """The liveness-vote fixes of the third pass had to land twice. Both front
    ends now go through decision.decide, so the next one lands once."""
    from cli import attendance
    ada = isolated_db.add_user("Ada")
    calls = []
    real = decision.decide

    def spy(*args, **kwargs):
        calls.append("face" if args[1] is not None else "nobody")
        return real(*args, **kwargs)
    monkeypatch.setattr(decision, "decide", spy)
    monkeypatch.setattr(liveness, "score", lambda bgr, box: 0.9)
    monkeypatch.setattr(traits, "detect", lambda bgr: [yunet_row(*BOX)])

    # The web camera: one frame with a face, one without.
    mgr = camera.CameraManager(capture_factory=lambda: None)
    monkeypatch.setattr(camera.facelandmarks, "fit", lambda gray, box: None)
    mgr._mode, mgr._recognizer = camera.MODE_ATTENDANCE, Recogniser(ada)
    mgr._handle_attendance(noise_frame())
    monkeypatch.setattr(traits, "detect", lambda bgr: [])
    mgr._handle_attendance(noise_frame())
    assert calls == ["face", "nobody"], "the camera did not decide through decision.decide"

    # The CLI: the same two frames.
    calls.clear()
    rows = [[yunet_row(*BOX)], []]
    monkeypatch.setattr(traits, "detect", lambda bgr: rows.pop(0))
    monkeypatch.setattr(attendance.decision, "sface_gallery", lambda: None)
    monkeypatch.setattr(attendance, "_load_recognizer", lambda: Recogniser(ada))
    frames = [noise_frame(), noise_frame()]

    class Cap:
        def read(self):
            return (True, frames.pop(0)) if frames else (False, None)

        def release(self):
            pass
    monkeypatch.setattr(attendance, "_open_camera", Cap)
    for name, fn in (("imshow", lambda *a: None), ("waitKey", lambda *a: -1),
                     ("destroyAllWindows", lambda: None)):
        monkeypatch.setattr(cv2, name, fn)
    attendance.run_attendance()
    assert calls == ["face", "nobody"], "the CLI did not decide through decision.decide"


# =============================================== 2. one mark per person per day

def test_two_connections_marking_one_person_at_once_write_one_row(isolated_db, monkeypatch):
    """Each thread's connections open in step with the other's, so a
    check-then-insert has both threads check before either inserts."""
    ada = isolated_db.add_user("Ada")
    barrier = threading.Barrier(2, timeout=10)
    opened = db.get_connection

    def in_step():
        conn = opened()
        barrier.wait()
        return conn
    monkeypatch.setattr(db, "get_connection", in_step)

    results, errors = [], []

    def mark():
        try:
            results.append(db.log_attendance(ada, 10.0))
        except Exception as exc:        # surfaced below, not lost in the thread
            errors.append(exc)
    threads = [threading.Thread(target=mark) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    monkeypatch.undo()

    assert errors == []
    assert sorted(results) == [False, True]
    assert len(isolated_db.get_attendance_for_today()) == 1


def test_the_database_itself_refuses_a_second_mark_on_one_day(isolated_db):
    ada = isolated_db.add_user("Ada")
    conns = [sqlite3.connect(paths.db_path()) for _ in range(2)]
    try:
        conns[0].execute("INSERT INTO attendance (user_id, timestamp, confidence) "
                         "VALUES (?, '2026-10-10T09:00:00', 1)", (ada,))
        conns[0].commit()
        with pytest.raises(sqlite3.IntegrityError):
            conns[1].execute("INSERT INTO attendance (user_id, timestamp, confidence) "
                             "VALUES (?, '2026-10-10T17:30:00.123456', 2)", (ada,))
        # Another day is another mark.
        conns[1].execute("INSERT INTO attendance (user_id, timestamp, confidence) "
                         "VALUES (?, '2026-10-11T09:00:00', 3)", (ada,))
        conns[1].commit()
    finally:
        for c in conns:
            c.close()


def test_log_attendance_keeps_its_return_values(isolated_db):
    ada = isolated_db.add_user("Ada")
    assert isolated_db.log_attendance(ada, 10.0) is True
    assert isolated_db.log_attendance(ada, 11.0) is False
    with pytest.raises(sqlite3.IntegrityError):     # a missing user still raises
        isolated_db.log_attendance(9999, 10.0)


OLD_SCHEMA = """
    CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                        created_at TEXT NOT NULL);
    CREATE TABLE attendance (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             user_id INTEGER NOT NULL, timestamp TEXT NOT NULL,
                             confidence REAL, method TEXT NOT NULL DEFAULT 'lbph',
                             FOREIGN KEY (user_id) REFERENCES users(id));
"""


def test_the_migration_keeps_the_earliest_mark_of_each_day(isolated_root):
    with sqlite3.connect(paths.db_path()) as conn:
        conn.executescript(OLD_SCHEMA)
        conn.executemany("INSERT INTO users (id, name, created_at) VALUES (?, ?, 'x')",
                         [(1, "Ada"), (2, "Ben")])
        # Inserted out of order, and in both ISO spellings: "earliest" is by
        # time, not by row id or by comparing the strings.
        conn.executemany(
            "INSERT INTO attendance (id, user_id, timestamp, confidence) VALUES (?, ?, ?, ?)",
            [(1, 1, "2026-10-01T11:00:00", 1),
             (2, 1, "2026-10-01T08:30:00.250000", 2),     # Ada's earliest on the 1st
             (3, 1, "2026-10-01 10:00:00", 3),
             (4, 1, "2026-10-02T09:00:00", 4),            # another day: kept
             (5, 2, "2026-10-01T12:00:00", 5),            # another person: kept
             (6, 2, "2026-10-01T12:00:00", 6)])           # exact tie: lower id kept

    db.init_db()

    def rows():
        with sqlite3.connect(paths.db_path()) as conn:
            return conn.execute("SELECT id FROM attendance ORDER BY id").fetchall()
    assert rows() == [(2,), (4,), (5,)]

    db.init_db()                        # every startup runs it: idempotent
    assert rows() == [(2,), (4,), (5,)]
    with sqlite3.connect(paths.db_path()) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO attendance (user_id, timestamp) "
                         "VALUES (1, '2026-10-02T18:00:00')")


# ====================================================== 3. a folder's user id

HEADS = {" 5": None, "-1": None, "05": 5, "5": 5, "abc": None, "": None}


@pytest.mark.parametrize("head, expected", list(HEADS.items()))
def test_one_rule_decides_a_folders_id(head, expected):
    assert paths.folder_id(f"{head}_Ada") == expected
    assert paths.folder_id(head) == expected


def test_training_and_folder_ids_agree_on_every_folder(isolated_root):
    image = np.full((20, 20), 128, np.uint8)        # not a face; LBPH never runs
    for i, head in enumerate(HEADS):
        folder = os.path.join(paths.dataset_dir(), f"{head}_Person{i}")
        os.makedirs(folder)
        cv2.imwrite(os.path.join(folder, "1.png"), image)
    names = os.listdir(paths.dataset_dir())
    assert len(names) == len(HEADS), names          # the file system kept every name

    _faces, labels = train_model.load_training_data()
    assert set(labels) == paths.folder_ids() == {5}
    assert labels.count(5) == 2, "05_ and 5_ are both id 5"
