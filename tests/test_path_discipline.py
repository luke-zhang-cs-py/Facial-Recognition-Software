"""paths.py exists to keep the database and the dataset in step. It did not.

Its own docstring describes the failure it was written to prevent: "A test
could point the database at a temporary file and still read the real
dataset/ off disk -- and the same split let a dataset/ folder reference a
user id that had no row in `users`, which crashed the analysis outright."

Four modules then reintroduced it, by reading their path once at import:

    db.DB_PATH            = paths.db_path()
    train_model.DATASET_DIR / MODEL_PATH
    seed_demo.DATASET_DIR

After paths.use() the dataset moved and those did not. A script pointed at a
copy would have read the copy and written the real attendance.db. Two more
modules -- attendance.py and register_user.py -- never used paths.py at all
and held bare relative names, so they only worked if you happened to be
standing in the project directory.
"""

import os
import re

import paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Filenames paths.py is the authority on. Anything else naming them is a
# second opinion about where they live.
OWNED = re.compile(r"""["'](?:trainer\.yml|attendance\.db)["']"""
                   r"""|=\s*["']dataset["']""")

# Modules that legitimately name them: paths.py decides, and the tests here
# are allowed to talk about the mistake.
ALLOWED = {"paths.py"}


def source_lines(name):
    text = open(os.path.join(ROOT, name), encoding="utf-8").read()
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield line


def test_no_module_names_a_path_that_paths_owns():
    offenders = []
    for name in sorted(os.listdir(ROOT)):
        if not name.endswith(".py") or name in ALLOWED:
            continue
        for line in source_lines(name):
            if OWNED.search(line):
                offenders.append(f"{name}: {line.strip()[:70]}")
    assert not offenders, "use paths.py instead:\n  " + "\n  ".join(offenders)


def test_no_module_holds_an_import_time_copy_of_a_path():
    """A module-level `X = paths.something()` reads the root once and then
    never moves, which is what broke use()."""
    captured = re.compile(r"^[A-Z_]+\s*=\s*paths\.\w+\(\)", re.M)
    offenders = []
    for name in sorted(os.listdir(ROOT)):
        if not name.endswith(".py") or name in ALLOWED:
            continue
        text = open(os.path.join(ROOT, name), encoding="utf-8").read()
        for match in captured.finditer(text):
            offenders.append(f"{name}: {match.group(0)}")
    assert not offenders, ("resolve per call, not at import:\n  "
                           + "\n  ".join(offenders))


def test_use_moves_every_store_together(tmp_path):
    """The contract, end to end."""
    import db
    import train_model

    previous = paths.use(str(tmp_path))
    try:
        for value in (paths.db_path(), paths.dataset_dir(), paths.model_path()):
            assert value.startswith(str(tmp_path)), value

        # And the modules follow, rather than holding an import-time copy.
        db.init_db()
        assert os.path.exists(os.path.join(str(tmp_path), "attendance.db")), \
            "the database did not move with the root"
        assert train_model.paths.dataset_dir() == paths.dataset_dir()
    finally:
        paths.use(previous)


def test_use_returns_the_previous_root_so_it_can_be_restored(tmp_path):
    original = paths.root()
    previous = paths.use(str(tmp_path))
    try:
        assert previous == original
    finally:
        paths.use(previous)
    assert paths.root() == original


def test_models_stay_with_the_code_not_the_data(tmp_path):
    """Pretrained weights are part of the program. A test pointing the data
    elsewhere still wants the real detector."""
    previous = paths.use(str(tmp_path))
    try:
        assert not paths.models_dir().startswith(str(tmp_path))
        assert paths.models_dir().startswith(paths.BASE_DIR)
    finally:
        paths.use(previous)


def test_user_folder_is_the_one_naming_rule():
    """register_user.py wrote this join out again. A dataset folder named
    differently from what the rest of the project looks for is a person who
    silently never gets recognised."""
    folder = paths.user_folder(7, "Jane Doe")
    assert folder.startswith(paths.dataset_dir())
    assert os.path.basename(folder) == "7_Jane_Doe"
    assert " " not in os.path.basename(folder)
