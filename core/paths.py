"""
core/paths.py
-------------
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
import re

# The project root, which is this file's *parent's* parent: paths.py lives in
# core/, and everything it names -- dataset/, trainer.yml, attendance.db,
# models/ -- sits beside core/ rather than inside it. Written out because the
# usual one-dirname spelling is what a move into a package silently breaks:
# it keeps resolving, just to the wrong folder, and the first symptom is an
# empty dataset rather than an error.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


# A name is typed into the web form, and it used to become a directory name
# with only its spaces replaced. "x/../../../elsewhere" therefore wrote face
# samples outside dataset/, and a name with a colon or a question mark --
# legal in a name, illegal in a Windows path -- raised after the user row had
# already been inserted, leaving a person with no folder. Anything that is
# not a letter, a digit, an underscore or a hyphen becomes an underscore.
#
# Only the folder is rewritten; the database keeps the name exactly as typed.
# Nothing finds a folder by its name part -- training and analysis read the
# id before the first underscore -- so this changes what is on disk and
# nothing about who is recognised.
_UNSAFE_IN_FOLDER = re.compile(r"[^\w-]")
FOLDER_NAME_MAX = 64


def user_folder(user_id, name):
    safe = _UNSAFE_IN_FOLDER.sub("_", name)[:FOLDER_NAME_MAX]
    return os.path.join(dataset_dir(), f"{int(user_id)}_{safe}")
