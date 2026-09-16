"""Training, loading and reporting -- the rest of what the camera manages.

The remaining uncovered block in `camera.py` after the capture loop was the
model lifecycle: retrain, ensure_trained, _load_recognizer, _build_report.
None of it needed a seam adding. `paths` resolves per call so `isolated_root`
already moves the dataset and the model, `train_model` is imported inside
`retrain` so it can be replaced, and LBPH trains on any grey image.

So these tests train a real model from a real dataset of synthetic crops
and read it back with the real recogniser. The only thing stubbed is
failure -- a training run that raises, which is hard to arrange honestly
and is exactly the path that matters when it happens.
"""
import os
import time

import cv2
import numpy as np
import pytest

import camera


@pytest.fixture
def mgr():
    return camera.CameraManager(capture_factory=lambda: None)


def seed_dataset(root, user_id=1, name="Ada", count=3):
    """A dataset folder shaped the way train_model expects: "<id>_<Name>".

    Deterministic noise rather than flat grey -- LBPH builds histograms of
    local binary patterns, and a uniform image has none.
    """
    folder = os.path.join(root, "dataset", f"{user_id}_{name}")
    os.makedirs(folder, exist_ok=True)
    for i in range(count):
        rng = np.random.RandomState(user_id * 100 + i)
        img = rng.randint(0, 255, (200, 200), dtype=np.uint8)
        cv2.imwrite(os.path.join(folder, f"{i}.jpg"), img)
    return folder


def train_a_real_model(root):
    """Produce a genuine trainer.yml at the isolated model path."""
    seed_dataset(root)
    import train_model
    train_model.train()
    assert os.path.exists(camera.paths.model_path()), "no model was written"


# ------------------------------------------------------------ loading it


def test_the_recogniser_is_read_from_disk(mgr, isolated_root):
    train_a_real_model(isolated_root)
    mgr._load_recognizer()

    assert mgr._recognizer is not None
    assert mgr._model_mtime == os.path.getmtime(camera.paths.model_path())


def test_an_unchanged_model_is_not_read_twice(mgr, isolated_root):
    """Reloading trainer.yml on every attendance frame would be pointless
    work on the capture thread, so the mtime is the guard."""
    train_a_real_model(isolated_root)
    mgr._load_recognizer()
    first = mgr._recognizer
    mgr._load_recognizer()

    assert mgr._recognizer is first, "it re-read a model that had not changed"


def test_invalidating_forces_the_next_read(mgr, isolated_root):
    train_a_real_model(isolated_root)
    mgr._load_recognizer()
    first = mgr._recognizer

    mgr.invalidate_model()
    assert mgr._recognizer is None
    assert mgr._model_mtime is None

    mgr._load_recognizer()
    assert mgr._recognizer is not None
    assert mgr._recognizer is not first, "the stale object came back"


# -------------------------------------------------------------- retraining


def test_retrain_builds_a_model_and_says_what_it_trained_on(mgr, isolated_root):
    seed_dataset(isolated_root, count=4)

    assert mgr.retrain(reason="after a test") is True
    assert os.path.exists(camera.paths.model_path())

    message = mgr.status()["events"][0]["message"]
    assert "4 images" in message, message
    assert "1 person" in message, "one user should be singular: " + message
    assert "after a test" in message, "the reason was dropped"


def test_retrain_counts_people_not_folders(mgr, isolated_root):
    seed_dataset(isolated_root, user_id=1, name="Ada", count=2)
    seed_dataset(isolated_root, user_id=2, name="Grace", count=2)

    assert mgr.retrain() is True
    message = mgr.status()["events"][0]["message"]
    assert "4 images" in message, message
    assert "2 people" in message, "two users should be plural: " + message


def test_retrain_with_nothing_to_train_on_is_not_an_error(mgr, isolated_root):
    """An empty dataset is the state of a fresh install, not a failure."""
    assert mgr.retrain() is False
    assert mgr.status()["error"] is None
    assert "Nothing to train on" in mgr.status()["events"][0]["message"]


def test_a_failed_training_run_is_reported_and_survived(mgr, isolated_root,
                                                        monkeypatch):
    """This runs on the capture thread. An exception escaping here would
    take the video feed down with it."""
    import train_model
    seed_dataset(isolated_root)
    monkeypatch.setattr(train_model, "train",
                        lambda: (_ for _ in ()).throw(RuntimeError("disk full")))

    assert mgr.retrain() is False
    status = mgr.status()
    assert "disk full" in status["error"]
    assert any("Training failed" in e["message"] for e in status["events"])


def test_retraining_invalidates_the_loaded_model(mgr, isolated_root):
    """The next attendance frame has to pick up the model just written, not
    the one held in memory from before it."""
    train_a_real_model(isolated_root)
    mgr._load_recognizer()
    assert mgr._recognizer is not None

    mgr.retrain()
    assert mgr._recognizer is None, "the old recogniser outlived the retrain"


# ---------------------------------------------------- training at startup


def test_no_dataset_means_nothing_to_do(mgr, isolated_root):
    assert mgr.ensure_trained() is False


def test_an_empty_dataset_folder_means_nothing_to_do(mgr, isolated_root):
    os.makedirs(os.path.join(isolated_root, "dataset"), exist_ok=True)
    assert mgr.ensure_trained() is False


def test_a_missing_model_with_samples_present_trains(mgr, isolated_root):
    """The point of this: the app is usable straight after a fresh clone
    with an existing dataset, instead of insisting on a manual step."""
    seed_dataset(isolated_root)
    assert not os.path.exists(camera.paths.model_path())

    assert mgr.ensure_trained() is True
    assert os.path.exists(camera.paths.model_path())


def test_a_model_newer_than_the_dataset_is_left_alone(mgr, isolated_root):
    train_a_real_model(isolated_root)
    # Push the model's timestamp past every sample.
    future = time.time() + 60
    os.utime(camera.paths.model_path(), (future, future))

    assert mgr.ensure_trained() is False, "it retrained a model already current"


def test_a_dataset_newer_than_the_model_retrains(mgr, isolated_root):
    train_a_real_model(isolated_root)
    # A new sample arrives after training.
    seed_dataset(isolated_root, user_id=2, name="Grace", count=1)
    future = time.time() + 60
    folder = os.path.join(isolated_root, "dataset", "2_Grace")
    for f in os.listdir(folder):
        os.utime(os.path.join(folder, f), (future, future))

    assert mgr.ensure_trained() is True
    assert any("dataset changed" in e["message"]
               for e in mgr.status()["events"])


# ------------------------------------------------------ the finished report


def test_the_report_is_stored_and_announced(mgr, monkeypatch):
    mgr._reg_user_id = 1
    mgr._reg_name = "Ada"
    mgr._reg_poses = [{"stage": "front", "yaw": 0.0}]
    monkeypatch.setattr(camera.enrollment, "build",
                        lambda uid, name, poses, folder: {"samples": 30})

    mgr._build_report()

    assert mgr.status()["report"] == {"samples": 30}
    assert any("report ready for Ada" in e["message"]
               for e in mgr.status()["events"])


def test_a_report_that_cannot_be_built_is_logged_not_raised(mgr, monkeypatch):
    """The last good report survives a failed rebuild.

    Asserting the report is None afterwards would have been the weaker
    test: None is also the starting value, so it passes whether the guard
    works or not. A previous report makes the two outcomes different.
    """
    mgr._reg_name = "Ada"
    mgr._reg_report = {"samples": 30, "from": "the run before"}

    def explode(uid, name, poses, folder):
        raise RuntimeError("no samples on disk")

    monkeypatch.setattr(camera.enrollment, "build", explode)

    mgr._build_report()          # must not propagate

    assert mgr.status()["report"] == {"samples": 30, "from": "the run before"}
    assert any("Could not build report" in e["message"]
               for e in mgr.status()["events"])


def test_no_report_is_not_an_empty_report(mgr, monkeypatch):
    """`build` returning None means there was nothing to summarise. Storing
    that would put a blank panel on screen and lose the previous report --
    which is what makes this distinguishable from doing nothing at all.
    """
    mgr._reg_report = {"samples": 30}
    monkeypatch.setattr(camera.enrollment, "build",
                        lambda uid, name, poses, folder: None)

    mgr._build_report()

    assert mgr.status()["report"] == {"samples": 30}, (
        "an absent report overwrote the one already there")
