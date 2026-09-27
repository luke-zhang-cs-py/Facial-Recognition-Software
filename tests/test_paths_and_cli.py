"""paths.py exists so the database and the dataset cannot drift apart, and the
CLI modules had no coverage at all."""
import os

import pytest

import layout

from core import paths


def test_all_paths_move_together():
    """The defect this module fixes: redirecting one store and not the other."""
    original = paths.root()
    try:
        paths.use("/tmp/somewhere")
        assert paths.dataset_dir().startswith(paths.root())
        assert paths.db_path().startswith(paths.root())
        assert paths.model_path().startswith(paths.root())
    finally:
        paths.use(original)


def test_use_returns_the_previous_root():
    original = paths.root()
    prev = paths.use("/tmp/x")
    assert prev == original
    assert paths.use(prev) == os.path.abspath("/tmp/x")
    assert paths.root() == original


def test_models_dir_stays_with_the_code():
    """A test pointing data elsewhere still wants the real weights."""
    original = paths.root()
    try:
        before = paths.models_dir()
        paths.use("/tmp/elsewhere")
        assert paths.models_dir() == before
    finally:
        paths.use(original)


def test_user_folder_is_filesystem_safe():
    f = paths.user_folder(7, "Jane Doe")
    assert "7_Jane_Doe" in f and " " not in os.path.basename(f)


@pytest.mark.parametrize("name", ["../../escape", "a/b", "a\\b", "..",
                                  "C:\\Windows", "x" * 500])
def test_user_folder_stays_one_level_inside_the_dataset(name):
    """Every name, however hostile, is one folder directly under dataset/."""
    f = paths.user_folder(7, name)
    assert os.path.dirname(f) == paths.dataset_dir()
    base = os.path.basename(f)
    assert base.startswith("7_")
    assert "/" not in base and "\\" not in base and ".." not in base
    assert len(base) <= len("7_") + paths.FOLDER_NAME_MAX


def test_every_weight_file_the_code_loads_can_be_fetched():
    """cli.fetch_models is the only way a fresh clone gets its weights, and it
    downloaded seven of the nine files. The two it skipped were the landmark
    model and the liveness net, loaded by their own modules rather than by
    facemodels -- and without the liveness net the web attendance loop waits
    for a verdict that never comes, so nobody is ever marked present."""
    from cli import fetch_models
    from core import facemodels
    from pipeline import landmarks, liveness

    loaded = {f for files, _ in facemodels.SPECS.values() for f in files}
    own = {os.path.basename(landmarks.model_path()),
           os.path.basename(liveness.model_path())}
    assert own <= loaded, "missing from SPECS, so /api/models never names them"
    assert loaded <= set(fetch_models.URLS), sorted(loaded - set(fetch_models.URLS))


def test_train_model_reports_nothing_to_train(isolated_root, capsys):
    """Redirected with paths.use() rather than by patching train_model's own
    copy of the path -- it does not keep one any more."""
    from pipeline import train_model
    assert not os.path.exists(paths.dataset_dir())
    faces, labels = train_model.load_training_data()
    assert faces == [] and labels == []
    train_model.train()
    assert "No training images" in capsys.readouterr().out


def test_train_model_skips_folders_without_an_id(isolated_root, capsys):
    from pipeline import train_model
    os.makedirs(os.path.join(paths.dataset_dir(), "junk"))
    train_model.load_training_data()
    assert "Skipping" in capsys.readouterr().out


def test_view_report_on_an_empty_database(isolated_db, capsys):
    from cli import view_report
    view_report.main()
    out = capsys.readouterr().out
    assert "none yet" in out and "no attendance logged" in out


def test_view_report_lists_users_and_records(isolated_db, capsys):
    from cli import view_report
    uid = isolated_db.add_user("Zoe")
    isolated_db.log_attendance(uid, 33.3)
    view_report.main()
    out = capsys.readouterr().out
    assert "Zoe" in out and "33.3" in out


def test_face_attendance_requires_a_subcommand():
    """The thin front end must not silently do nothing.

    Run as `-m cli.face_attendance` from the project root, which is how it is
    invoked now that it lives in a package. The old spelling passed a bare
    filename with `cwd=os.getcwd()`, so it depended on pytest happening to be
    started from the root -- and anywhere else the failure it saw was "no
    such file", which would satisfy `returncode != 0` for the wrong reason.
    """
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "cli.face_attendance"],
                       capture_output=True, text=True, cwd=layout.ROOT)
    assert r.returncode != 0, (
        "it exited 0 with no subcommand: " + (r.stdout + r.stderr))
    assert "command" in (r.stderr + r.stdout).lower()


def test_face_attendance_defines_no_duplicated_logic():
    """It used to re-implement twelve functions that already existed."""
    import ast
    tree = ast.parse(layout.source_of("cli/face_attendance.py"))
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert funcs == {"main"}, f"logic crept back in: {funcs}"
