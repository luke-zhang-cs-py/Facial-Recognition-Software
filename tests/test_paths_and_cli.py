"""paths.py exists so the database and the dataset cannot drift apart, and the
CLI modules had no coverage at all."""
import os

import paths


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


def test_train_model_reports_nothing_to_train(isolated_root, capsys):
    """Redirected with paths.use() rather than by patching train_model's own
    copy of the path -- it does not keep one any more."""
    import train_model
    assert not os.path.exists(paths.dataset_dir())
    faces, labels = train_model.load_training_data()
    assert faces == [] and labels == []
    train_model.train()
    assert "No training images" in capsys.readouterr().out


def test_train_model_skips_folders_without_an_id(isolated_root, capsys):
    import train_model
    os.makedirs(os.path.join(paths.dataset_dir(), "junk"))
    train_model.load_training_data()
    assert "Skipping" in capsys.readouterr().out


def test_view_report_on_an_empty_database(isolated_db, capsys):
    import view_report
    view_report.main()
    out = capsys.readouterr().out
    assert "none yet" in out and "no attendance logged" in out


def test_view_report_lists_users_and_records(isolated_db, capsys):
    import view_report
    uid = isolated_db.add_user("Zoe")
    isolated_db.log_attendance(uid, 33.3)
    view_report.main()
    out = capsys.readouterr().out
    assert "Zoe" in out and "33.3" in out


def test_face_attendance_requires_a_subcommand():
    """The thin front end must not silently do nothing."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "face_attendance.py"],
                       capture_output=True, text=True, cwd=os.getcwd())
    assert r.returncode != 0
    assert "command" in (r.stderr + r.stdout).lower()


def test_face_attendance_defines_no_duplicated_logic():
    """It used to re-implement twelve functions that already existed."""
    import ast
    tree = ast.parse(open("face_attendance.py", encoding="utf-8").read())
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert funcs == {"main"}, f"logic crept back in: {funcs}"
