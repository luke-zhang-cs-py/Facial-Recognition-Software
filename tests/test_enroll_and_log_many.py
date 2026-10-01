"""More faces, end to end: enroll several new people, train, recognise them,
and check each lands in the attendance log -- the table, and /api/report that
the web page's log reads -- exactly once.

Real faces, not synthetic ones: the photographs come from this checkout's own
dataset/ (filled by `python -m cli.seed_demo`, LFW crops already at the LBPH
input size). Without it the test skips, as face_image does, rather than
asserting against a fake.

What it holds the system to is what should be exact: the plumbing from a new
user row to a folder, to a label in the model, to a name in the log, once per
person per day. It does not claim LBPH tells LFW strangers apart: on photos it
was not trained on, 8 people x 8 photos, it named the right person 9 times in
16 and accepted some wrong ones under the 70 threshold, so a held-out check
here would be a measurement of LBPH, not of this plumbing.
"""
import os
import shutil

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "dataset")
NEW_PEOPLE = 8          # people added on top of an empty system
TRAIN_PER_PERSON = 8    # photos enrolled per person
MIN_PHOTOS = 10         # people with fewer are not picked


def _people():
    """Folders in the checkout's dataset/ with enough photos, in a fixed order."""
    if not os.path.isdir(SOURCE):
        return []
    out = []
    for folder in sorted(os.listdir(SOURCE)):
        path = os.path.join(SOURCE, folder)
        if not os.path.isdir(path) or "_" not in folder:
            continue
        photos = sorted((f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png"))),
                        key=lambda f: (len(f), f))
        if len(photos) >= MIN_PHOTOS:
            out.append((folder.split("_", 1)[1].replace("_", " "), [os.path.join(path, f) for f in photos]))
    return out


@pytest.fixture
def people():
    found = _people()
    if len(found) < NEW_PEOPLE + 2:
        pytest.skip("no seeded dataset/ in this checkout (python -m cli.seed_demo)")
    return found


@pytest.fixture
def client(isolated_root):
    from core import db
    db.init_db()
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    return flask_app.app.test_client()


def _enroll(db, paths, name, photos):
    """What registering someone amounts to: a user row and a folder of samples."""
    uid = db.add_user(name)
    folder = paths.user_folder(uid, name)
    os.makedirs(folder, exist_ok=True)
    for i, photo in enumerate(photos, 1):
        shutil.copy(photo, os.path.join(folder, f"{i}.jpg"))
    return uid


def _gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, path
    return img


def _whole(img):
    """The photos are already face crops, so the face box is the whole image."""
    return (0, 0, img.shape[1], img.shape[0])


def _trained(people):
    """Enroll NEW_PEOPLE, train the real model, and return (recognizer, enrolled)."""
    from core import db, paths
    from pipeline import train_model
    enrolled = {}
    for name, photos in people[:NEW_PEOPLE]:
        uid = _enroll(db, paths, name, photos[:TRAIN_PER_PERSON])
        enrolled[uid] = (name, photos)
    train_model.train()
    assert os.path.exists(paths.model_path()), "training wrote no model"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(paths.model_path())
    return recognizer, enrolled


def test_more_faces_are_enrolled_recognised_and_logged_once_each(client, people):
    from core import db, vision
    from cli import attendance
    recognizer, enrolled = _trained(people)

    assert len(db.get_all_users()) == NEW_PEOPLE
    session = set()
    for uid, (name, photos) in enrolled.items():
        # two frames of the same person, as a camera would see them
        for photo in photos[:2]:
            img = _gray(photo)
            got, confidence, shown, accepted = attendance.identify(recognizer, img, _whole(img))
            assert accepted and got == uid and shown == name, \
                f"{name}: an enrolled photo came back as {shown} ({confidence:.1f})"
            attendance.mark_present(got, shown, confidence, session)

    today = db.get_attendance_for_today()
    assert sorted(n for n, _, _ in today) == sorted(name for name, _ in enrolled.values()), \
        "everyone enrolled is in today's log"
    assert len(today) == NEW_PEOPLE, "once per person, however many frames"
    assert all(c < vision.CONFIDENCE_THRESHOLD for _, _, c in today), "only accepted matches are logged"

    # the log the web page reads
    report = client.get("/api/report").get_json()
    assert sorted(u["name"] for u in report["users"]) == sorted(n for n, _ in enrolled.values())
    assert sorted(r["name"] for r in report["today"]) == sorted(n for n, _ in enrolled.values())
    assert len(report["all"]) == NEW_PEOPLE

    # a second camera session later the same day logs no one twice
    again = set()
    uid, (name, photos) = next(iter(enrolled.items()))
    img = _gray(photos[0])
    got, confidence, shown, accepted = attendance.identify(recognizer, img, _whole(img))
    attendance.mark_present(got, shown, confidence, again)
    assert len(db.get_attendance_for_today()) == NEW_PEOPLE


def test_adding_people_later_keeps_the_earlier_log_and_logs_the_new_ones(client, people):
    """Enrolling more people after attendance has run must not lose or relabel
    anyone already logged: labels are user ids, so a retrain cannot shift them."""
    from core import db, paths
    from cli import attendance
    from pipeline import train_model

    recognizer, enrolled = _trained(people)
    first_uid, (first_name, first_photos) = next(iter(enrolled.items()))
    img = _gray(first_photos[0])
    got, confidence, shown, accepted = attendance.identify(recognizer, img, _whole(img))
    assert accepted and got == first_uid
    attendance.mark_present(got, shown, confidence, set())

    late = people[NEW_PEOPLE:NEW_PEOPLE + 2]
    late_ids = {_enroll(db, paths, name, photos[:TRAIN_PER_PERSON]): name for name, photos in late}
    train_model.train()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(paths.model_path())

    session = set()
    for (name, photos), uid in zip(late, late_ids):
        img = _gray(photos[0])
        got, confidence, shown, accepted = attendance.identify(recognizer, img, _whole(img))
        assert accepted and got == uid and shown == name, f"{name} came back as {shown}"
        attendance.mark_present(got, shown, confidence, session)

    names = sorted(n for n, _, _ in db.get_attendance_for_today())
    assert names == sorted([first_name] + list(late_ids.values()))
    still = attendance.identify(recognizer, img, _whole(img))
    assert still[0] != first_uid, "a later person is not relabelled as the first"


def test_a_rejected_face_is_never_logged(client, people):
    from core import db
    from cli import attendance
    recognizer, _ = _trained(people)
    noise = np.random.default_rng(7).integers(0, 256, (200, 200), dtype=np.uint8)
    got, confidence, shown, accepted = attendance.identify(recognizer, noise, _whole(noise))
    assert not accepted and shown == "Unknown", f"noise was accepted as {shown} ({confidence:.1f})"
    assert db.get_attendance_for_today() == []
    assert client.get("/api/report").get_json()["today"] == []
