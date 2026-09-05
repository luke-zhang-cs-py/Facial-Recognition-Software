"""db.py had a connection leak that turned any single error into 'database is
locked' for the rest of the process. These lock that fix down."""
import sqlite3

import pytest


def test_add_and_list_users(isolated_db):
    uid = isolated_db.add_user("Alice")
    assert isolated_db.get_user_name(uid) == "Alice"
    assert (uid, "Alice") in isolated_db.get_all_users()


def test_user_exists(isolated_db):
    uid = isolated_db.add_user("Bob")
    assert isolated_db.user_exists(uid)
    assert not isolated_db.user_exists(99999)


def test_attendance_is_once_per_day(isolated_db):
    uid = isolated_db.add_user("Carol")
    assert isolated_db.log_attendance(uid, 40.0) is True
    assert isolated_db.log_attendance(uid, 41.0) is False, "second mark same day"
    assert len(isolated_db.get_attendance_for_today()) == 1


def test_save_traits_rejects_an_unknown_user_without_raising(isolated_db):
    row = dict(path="x.jpg", user_id=4242, mtime=1.0, sharpness=1.0,
               brightness=1.0, contrast=1.0, quality=0.5, face_px=10,
               yaw=0.0, roll=0.0, detected=1, flags="", embedding=None,
               age_label=None, age_conf=None, gender_label=None,
               gender_conf=None)
    assert isolated_db.save_traits(row) is False


def test_a_rejected_write_does_not_lock_the_database(isolated_db):
    """The regression that mattered: one bad row used to leave a connection
    open with its transaction, and every later write failed."""
    uid = isolated_db.add_user("Dave")
    bad = dict(path="x.jpg", user_id=4242, mtime=1.0, sharpness=1.0,
               brightness=1.0, contrast=1.0, quality=0.5, face_px=10,
               yaw=0.0, roll=0.0, detected=1, flags="", embedding=None,
               age_label=None, age_conf=None, gender_label=None,
               gender_conf=None)
    isolated_db.save_traits(bad)
    good = dict(bad, user_id=uid, path="y.jpg")
    assert isolated_db.save_traits(good) is True
    assert isolated_db.add_user("Eve") is not None
    assert isolated_db.get_cached_traits("y.jpg", 1.0) is not None


def test_an_exception_mid_transaction_rolls_back_and_releases(isolated_db):
    with pytest.raises(RuntimeError):
        with isolated_db.connection() as conn:
            conn.cursor().execute(
                "INSERT INTO users (name, created_at) VALUES (?,?)",
                ("Ghost", "now"))
            raise RuntimeError("boom")
    assert "Ghost" not in [n for _, n in isolated_db.get_all_users()]
    assert isolated_db.add_user("Frank") is not None


def test_cached_traits_are_keyed_by_mtime(isolated_db):
    uid = isolated_db.add_user("Grace")
    row = dict(path="z.jpg", user_id=uid, mtime=100.0, sharpness=1.0,
               brightness=1.0, contrast=1.0, quality=0.5, face_px=10,
               yaw=0.0, roll=0.0, detected=1, flags="", embedding=None,
               age_label=None, age_conf=None, gender_label=None,
               gender_conf=None)
    isolated_db.save_traits(row)
    assert isolated_db.get_cached_traits("z.jpg", 100.0) is not None
    assert isolated_db.get_cached_traits("z.jpg", 101.0) is None, \
        "an edited file must miss the cache"


def test_delete_user_clears_everything(isolated_db):
    uid = isolated_db.add_user("Heidi")
    isolated_db.log_attendance(uid, 30.0)
    isolated_db.delete_user(uid)
    assert isolated_db.get_user_name(uid) is None
    assert isolated_db.get_attendance_for_today() == []
