"""The small entry points: cli/face_attendance.py's dispatch,
core/corpus_paths.py's environment handling, and cli/fetch_models.py with
every download mocked.

No test here touches the network: an autouse fixture replaces urlopen with
one that fails the test, and each download test installs its own fake.
"""
import hashlib
import os
import sys
import tempfile

import pytest

from cli import face_attendance, fetch_models
from core import corpus_paths


# ------------------------------------------------------- face_attendance

@pytest.fixture
def dispatch(isolated_db, monkeypatch):
    """Record which real module each subcommand reaches."""
    calls = []
    monkeypatch.setattr(face_attendance.register_user, "register_user",
                        lambda name: calls.append(("register", name)))
    monkeypatch.setattr(face_attendance.train_model, "train",
                        lambda: calls.append(("train",)))
    monkeypatch.setattr(face_attendance.attendance, "run_attendance",
                        lambda: calls.append(("attendance",)))
    monkeypatch.setattr(face_attendance.view_report, "main",
                        lambda: calls.append(("report",)))
    return calls


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["face_attendance", *argv])
    return face_attendance.main()


@pytest.mark.parametrize("argv, expected", [
    (["train"], [("train",)]),
    (["attendance"], [("attendance",)]),
    (["report"], [("report",)]),
    # Registering trains straight after, so the person can be recognised.
    (["register", "Grace Hopper"], [("register", "Grace Hopper"), ("train",)]),
    (["register", "  Ada  "], [("register", "Ada"), ("train",)]),
])
def test_each_subcommand_reaches_its_module(dispatch, monkeypatch, argv,
                                            expected):
    assert run_cli(monkeypatch, *argv) == 0
    assert dispatch == expected


def test_dispatch_initialises_the_database(dispatch, monkeypatch):
    from core import db, paths
    os.remove(paths.db_path())
    run_cli(monkeypatch, "report")
    assert os.path.exists(paths.db_path())
    assert db.get_all_users() == []


@pytest.mark.parametrize("name", ["   ", "-Ada"])
def test_register_refuses_a_name_register_user_refuses(dispatch, monkeypatch,
                                                       capsys, name):
    """`python -m cli.register_user "  "` is refused; the same name through
    this front end created a user called "   " and trained on nothing."""
    assert run_cli(monkeypatch, "register", "--", name) == 1
    assert dispatch == []
    assert "Usage:" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [[], ["dance"], ["register"]])
def test_bad_arguments_exit_with_usage(dispatch, monkeypatch, argv):
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch, *argv)
    assert stop.value.code == 2 and dispatch == []


# ---------------------------------------------------------- corpus_paths

@pytest.fixture
def clean_env(monkeypatch):
    for var in ("FACE_CORPORA", "CASIA_DIR", "TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(var, raising=False)
    # gettempdir() caches its answer; clear it so each test recomputes.
    monkeypatch.setattr(tempfile, "tempdir", None)
    return monkeypatch


def every_path():
    return {"corpora": corpus_paths.corpora_dir(),
            "gallery": corpus_paths.gallery_dir(),
            "galleryFile": corpus_paths.gallery_path(),
            "lfw": corpus_paths.lfw_parquet(),
            "fairface": corpus_paths.fairface_dir(),
            "faceage": corpus_paths.faceage_parquet(),
            "frames": corpus_paths.sample_frame_dir()}


def test_face_corpora_moves_every_corpus(clean_env, tmp_path):
    clean_env.setenv("FACE_CORPORA", str(tmp_path))
    p = every_path()
    assert p["corpora"] == str(tmp_path)
    assert p["gallery"] == os.path.join(str(tmp_path), "casia")
    assert p["galleryFile"] == os.path.join(str(tmp_path), "casia",
                                            "gallery.npz")
    assert p["lfw"] == os.path.join(str(tmp_path), "lfw", "lfw.parquet")
    assert p["fairface"] == os.path.join(str(tmp_path), "fairface")
    assert p["faceage"] == os.path.join(str(tmp_path), "faceage",
                                        "val.parquet")
    assert p["frames"] == os.path.join(str(tmp_path), "frames")


def test_casia_dir_moves_only_the_gallery(clean_env, tmp_path):
    clean_env.setenv("FACE_CORPORA", str(tmp_path / "base"))
    clean_env.setenv("CASIA_DIR", str(tmp_path / "big_disk"))
    p = every_path()
    assert p["gallery"] == str(tmp_path / "big_disk")
    assert p["galleryFile"] == str(tmp_path / "big_disk" / "gallery.npz")
    assert p["lfw"].startswith(str(tmp_path / "base"))


def test_an_empty_variable_counts_as_unset(clean_env, tmp_path):
    clean_env.setenv("TMPDIR", str(tmp_path))
    clean_env.setenv("FACE_CORPORA", "")
    clean_env.setenv("CASIA_DIR", "")
    assert corpus_paths.corpora_dir() == str(tmp_path)
    assert corpus_paths.gallery_dir() == os.path.join(str(tmp_path), "casia")


def test_windows_style_temp_is_honoured(clean_env, tmp_path):
    """Windows sets TEMP and TMP; nothing here reads TEMP directly, which
    is what used to raise KeyError everywhere else."""
    clean_env.setenv("TEMP", str(tmp_path))
    clean_env.setenv("TMP", str(tmp_path))
    assert corpus_paths.corpora_dir() == str(tmp_path)
    assert corpus_paths.lfw_parquet() == os.path.join(str(tmp_path), "lfw",
                                                      "lfw.parquet")


def test_posix_style_temp_is_honoured_without_temp_set(clean_env, tmp_path):
    """POSIX sets TMPDIR (or nothing) and never TEMP. os.environ["TEMP"]
    raised here; os.environ.get("TEMP", ".") silently looked in the cwd."""
    clean_env.setenv("TMPDIR", str(tmp_path))
    assert "TEMP" not in os.environ
    assert corpus_paths.corpora_dir() == str(tmp_path)
    assert corpus_paths.sample_frame_dir() != os.path.join(".", "frames")


def test_no_temp_variable_at_all_still_resolves_to_a_real_directory(clean_env):
    base = corpus_paths.corpora_dir()
    assert os.path.isabs(base) and os.path.isdir(base)
    assert base == tempfile.gettempdir()


# ---------------------------------------------------------- fetch_models

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        pytest.fail("a test tried to reach the network")
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", refuse)


@pytest.fixture
def models_dir(monkeypatch, tmp_path):
    from core import paths
    folder = tmp_path / "models"
    monkeypatch.setattr(paths, "models_dir", lambda: str(folder))
    return folder


class Reply:
    """A urlopen response that serves `body` in chunks, optionally failing
    after `fail_after` bytes like a dropped connection."""

    def __init__(self, body, fail_after=None):
        self.body, self.pos, self.fail_after = body, 0, fail_after

    def read(self, n):
        if self.fail_after is not None and self.pos >= self.fail_after:
            raise ConnectionResetError("connection reset by peer")
        chunk = self.body[self.pos:self.pos + min(n, 4096)]
        self.pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def serve(monkeypatch, *replies):
    """urlopen hands out these replies (or raises these exceptions) in turn."""
    queue, urls = list(replies), []

    def urlopen(url, timeout):
        urls.append((url, timeout))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", urlopen)
    return urls


BODY = os.urandom(10_000)
NAME = "minifasnet_v2.onnx"


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setitem(fetch_models.SHA256, NAME,
                        hashlib.sha256(BODY).hexdigest())


def test_a_verified_download_is_moved_into_place(models_dir, pinned,
                                                 monkeypatch, capsys):
    models_dir.mkdir()
    urls = serve(monkeypatch, Reply(BODY))
    assert fetch_models.download(NAME, "https://example.invalid/m") is True
    assert (models_dir / NAME).read_bytes() == BODY
    assert not (models_dir / (NAME + ".part")).exists()
    assert urls == [("https://example.invalid/m", 120)]
    assert "checksum verified" in capsys.readouterr().out


def test_a_file_already_present_is_not_fetched(models_dir, monkeypatch, capsys):
    models_dir.mkdir()
    (models_dir / NAME).write_bytes(b"whatever is there")
    assert fetch_models.download(NAME, "https://example.invalid/m") is True
    assert "have    " + NAME in capsys.readouterr().out
    assert (models_dir / NAME).read_bytes() == b"whatever is there"


def test_force_replaces_a_present_file(models_dir, pinned, monkeypatch):
    models_dir.mkdir()
    (models_dir / NAME).write_bytes(b"old")
    serve(monkeypatch, Reply(BODY))
    assert fetch_models.download(NAME, "u", force=True) is True
    assert (models_dir / NAME).read_bytes() == BODY


def test_a_checksum_mismatch_is_refused_and_nothing_is_kept(
        models_dir, monkeypatch, capsys):
    models_dir.mkdir()
    serve(monkeypatch, Reply(BODY))      # the real pin, which BODY is not
    assert fetch_models.download(NAME, "u") is False
    assert os.listdir(models_dir) == []
    out = capsys.readouterr().out
    assert "checksum" in out and "refusing it" in out


def test_a_mismatch_under_force_keeps_the_good_copy(models_dir, monkeypatch):
    models_dir.mkdir()
    (models_dir / NAME).write_bytes(b"the copy that worked")
    serve(monkeypatch, Reply(BODY))
    assert fetch_models.download(NAME, "u", force=True) is False
    assert (models_dir / NAME).read_bytes() == b"the copy that worked"


def test_a_partial_download_leaves_no_part_file(models_dir, pinned,
                                                monkeypatch, capsys):
    models_dir.mkdir()
    serve(monkeypatch, Reply(BODY, fail_after=4096))
    assert fetch_models.download(NAME, "u") is False
    assert os.listdir(models_dir) == []
    assert "FAILED (connection reset by peer)" in capsys.readouterr().out


def test_a_connection_that_never_opens(models_dir, monkeypatch, capsys):
    models_dir.mkdir()
    serve(monkeypatch, OSError("name resolution failed"))
    assert fetch_models.download(NAME, "u") is False
    assert os.listdir(models_dir) == []


def test_a_retry_after_a_failure_succeeds(models_dir, pinned, monkeypatch):
    """Run it again: a stale .part from a killed run is simply overwritten,
    and the second attempt lands the verified file."""
    models_dir.mkdir()
    (models_dir / (NAME + ".part")).write_bytes(b"left by a killed run")
    serve(monkeypatch, Reply(BODY, fail_after=0), Reply(BODY))
    assert fetch_models.download(NAME, "u") is False
    assert fetch_models.download(NAME, "u") is True
    assert sorted(os.listdir(models_dir)) == [NAME]
    assert (models_dir / NAME).read_bytes() == BODY


def test_an_lfs_pointer_is_refused_before_hashing(models_dir, monkeypatch,
                                                  capsys):
    models_dir.mkdir()
    serve(monkeypatch, Reply(b"version https://git-lfs.github.com/spec/v1\n"))
    monkeypatch.setattr(fetch_models, "sha256_of",
                        lambda p: pytest.fail("hashed a pointer"))
    assert fetch_models.download(NAME, "u") is False
    assert "LFS pointer" in capsys.readouterr().out
    assert os.listdir(models_dir) == []


def test_a_small_prototxt_is_allowed_through_to_the_checksum(
        models_dir, monkeypatch):
    """A prototxt is a few KB of text by nature, so the size floor does not
    apply; the checksum still does."""
    models_dir.mkdir()
    proto = b"name: 'tiny'\n"
    monkeypatch.setitem(fetch_models.SHA256, "age_deploy.prototxt",
                        hashlib.sha256(proto).hexdigest())
    serve(monkeypatch, Reply(proto))
    assert fetch_models.download("age_deploy.prototxt", "u") is True
    assert (models_dir / "age_deploy.prototxt").read_bytes() == proto


def test_sha256_of_matches_hashlib(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(BODY * 200)            # several 1 MB chunks
    assert fetch_models.sha256_of(str(p)) == hashlib.sha256(BODY * 200).hexdigest()


def run_fetch(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["fetch_models", *argv])
    return fetch_models.main()


def test_list_reports_status_and_downloads_nothing(models_dir, monkeypatch,
                                                   capsys):
    models_dir.mkdir()
    (models_dir / "age_deploy.prototxt").write_bytes(b"x")
    (models_dir / "lbfmodel.yaml").write_bytes(b"x")
    assert run_fetch(monkeypatch, "--list") == 0
    out = capsys.readouterr().out
    assert "ready   landmarks" in out and "MISSING age" in out
    assert "age_deploy.prototxt" in out and "not downloaded" in out
    assert sorted(os.listdir(models_dir)) == ["age_deploy.prototxt",
                                              "lbfmodel.yaml"]


def test_main_creates_the_folder_and_reports_failures(models_dir, monkeypatch,
                                                      capsys):
    tried = []

    def download(name, url, force=False):
        tried.append((name, force))
        return name != "lbfmodel.yaml"
    monkeypatch.setattr(fetch_models, "download", download)
    assert run_fetch(monkeypatch, "--force") == 1
    assert models_dir.is_dir()
    assert [n for n, _ in tried] == list(fetch_models.URLS)
    assert all(force for _, force in tried)
    out = capsys.readouterr().out
    assert "1 file(s) failed: lbfmodel.yaml" in out
    assert "The app still runs" in out


def test_main_succeeds_when_every_file_lands(models_dir, monkeypatch, capsys):
    monkeypatch.setattr(fetch_models, "download",
                        lambda name, url, force=False: True)
    assert run_fetch(monkeypatch) == 0
    assert "All models present." in capsys.readouterr().out
