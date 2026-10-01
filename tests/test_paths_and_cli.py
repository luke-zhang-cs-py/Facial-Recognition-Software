"""paths.py exists so the database and the dataset cannot drift apart, and the
CLI modules had no coverage at all."""
import hashlib
import io
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


def test_every_download_has_a_checksum():
    from cli import fetch_models
    assert set(fetch_models.SHA256) == set(fetch_models.URLS)
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in fetch_models.SHA256.values())


class _Reply(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def test_a_download_that_does_not_match_its_checksum_is_refused(monkeypatch, tmp_path):
    """The weights come from third-party repos; a file swapped or corrupted at
    the source used to be saved and loaded like the real one."""
    from cli import fetch_models
    monkeypatch.setattr(fetch_models, "path_for", lambda f: str(tmp_path / f))
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", lambda url, timeout: _Reply(b"not the model" * 400))
    assert fetch_models.download("minifasnet_v2.onnx", "https://example.invalid/x.onnx") is False
    assert list(tmp_path.iterdir()) == [], "the refused file (and its .part) must not be left behind"


def test_a_download_that_matches_its_checksum_is_kept(monkeypatch, tmp_path):
    from cli import fetch_models
    body = b"a real model, for the purpose of this test" * 100
    monkeypatch.setattr(fetch_models, "path_for", lambda f: str(tmp_path / f))
    monkeypatch.setattr(fetch_models, "SHA256", {"m.onnx": hashlib.sha256(body).hexdigest()})
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", lambda url, timeout: _Reply(body))
    assert fetch_models.download("m.onnx", "https://example.invalid/m.onnx") is True
    assert (tmp_path / "m.onnx").read_bytes() == body


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


def test_register_user_help_prints_usage_and_creates_nobody(isolated_db, capsys):
    """Any one argument was the name, so `--help` registered a user called
    "--help" and made dataset/<id>_--help (Oct 2026 audit)."""
    from cli import register_user
    for argv in (["--help"], ["-h"], ["--name"], []):
        register_user.main(argv)
    assert isolated_db.get_all_users() == []
    assert "usage" in capsys.readouterr().out.lower()


def test_seed_demo_keep_skips_people_already_enrolled():
    """--keep adds people. It picked the most-photographed first -- the ones
    already enrolled -- and enrolled them again under a second id."""
    from cli import seed_demo
    names = ["George_W_Bush", "Colin_Powell", "Ada_Lovelace"]
    labels = [0] * 5 + [1] * 4 + [2] * 3
    picked = seed_demo.pick_people(labels, names, 3, 2, None, exclude=["George W Bush"])
    assert [names[lab] for lab, _ in picked] == ["Colin_Powell", "Ada_Lovelace"]
    assert [len(idx) for _, idx in picked] == [4, 3]
    everyone = seed_demo.pick_people(labels, names, 4, None)
    assert [names[lab] for lab, _ in everyone] == ["George_W_Bush", "Colin_Powell"]
