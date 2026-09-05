"""
paths.py
---------
Where things live on disk, in one place.

Three modules used to each work this out for themselves at import time, which
made them independently redirectable and nothing kept them in step. A test
could point the database at a temporary file and still read the real
dataset/ off disk -- and the same split let a dataset/ folder reference a
user id that had no row in `users`, which crashed the analysis outright.

Everything resolves through here, and `use()` moves the whole set together so
the two stores cannot drift apart.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_root = BASE_DIR


def use(root):
    """Point every path at a different root. Returns the previous one."""
    global _root
    previous = _root
    _root = os.path.abspath(root)
    return previous


def root():
    return _root


def dataset_dir():
    return os.path.join(_root, "dataset")


def model_path():
    return os.path.join(_root, "trainer.yml")


def db_path():
    return os.path.join(_root, "attendance.db")


def models_dir():
    """Pretrained weights stay with the code, not with the data root -- a
    test pointing the data elsewhere still wants the real models."""
    return os.path.join(BASE_DIR, "models")


def user_folder(user_id, name):
    return os.path.join(dataset_dir(), f"{user_id}_{name.replace(' ', '_')}")
