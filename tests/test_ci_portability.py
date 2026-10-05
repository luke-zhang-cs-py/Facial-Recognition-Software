"""Every test must run on the CI runner, which does not have what this
machine has.

Two of the three red CI runs in this repo's history were the same mistake:
a test that needed pyarrow, which is optional (only the corpus readers use
it) and is not installed on the runner. Run #12 on 2026-09-16, in
test_sample_frames.py, and run #30 on 2026-10-01, in test_paths_and_cli.py.
Both passed locally, because pyarrow is installed here. This test catches
the mistake before a push rather than after.
"""
import ast
import os
import re
import subprocess

import pytest

import layout

TESTS = os.path.join(layout.ROOT, "tests")

# Installed here, not by requirements.txt on the runner.
OPTIONAL = ("pyarrow",)


def _imports_optional(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import) and any(a.name.split(".")[0] in OPTIONAL for a in sub.names):
            return True
        if isinstance(sub, ast.ImportFrom) and (sub.module or "").split(".")[0] in OPTIONAL:
            return True
    return False


def _calls(node):
    return {sub.func.id for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)}


def _guarded(fn, source):
    """A skip decorator, or pytest.importorskip in the body."""
    if any("skip" in ast.get_source_segment(source, d) or "needs_" in ast.get_source_segment(source, d)
           for d in fn.decorator_list):
        return True
    return "importorskip" in ast.get_source_segment(source, fn)


def unguarded_tests():
    found = []
    for name in sorted(os.listdir(TESTS)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        source = open(os.path.join(TESTS, name), encoding="utf-8").read()
        funcs = {f.name: f for f in ast.parse(source).body if isinstance(f, ast.FunctionDef)}
        # A function needs the package if it imports it, or calls one that does.
        needs = {n for n, f in funcs.items() if _imports_optional(f)}
        grew = True
        while grew:
            more = {n for n, f in funcs.items() if n not in needs and _calls(f) & needs}
            grew = bool(more)
            needs |= more
        found += [f"{name}::{n}" for n in sorted(needs)
                  if n.startswith("test_") and not _guarded(funcs[n], source)]
    return found


def test_no_test_needs_an_optional_package_without_skipping_when_it_is_absent():
    assert unguarded_tests() == [], (
        "these tests import an optional package (or call a helper that does) "
        "and would fail on CI; skip them when it is absent or test without it")


def test_the_check_catches_the_mistake_it_is_for(tmp_path, monkeypatch):
    bad = tmp_path / "test_bad.py"
    bad.write_text("def helper():\n    import pyarrow\n\n"
                   "def test_direct():\n    import pyarrow.parquet\n\n"
                   "def test_via_helper():\n    helper()\n\n"
                   "@needs_parquet\ndef test_guarded():\n    helper()\n\n"
                   "def test_skips_itself():\n    pytest.importorskip('pyarrow')\n    helper()\n\n"
                   "def test_unrelated():\n    pass\n")
    monkeypatch.setattr(__import__(__name__), "TESTS", str(tmp_path))
    assert unguarded_tests() == ["test_bad.py::test_direct", "test_bad.py::test_via_helper"]


# Run #1 (2026-09-09) failed partly on a path from the machine the code was
# written on: 'c:\Users\<name>\Facial-Recognition-Software' does not exist
# on the runner, or on anybody else's computer.
SEP = r"[\\/]"                   # a backslash or a forward slash
MACHINE_PATH = re.compile(r"(?i)\b[a-z]:" + SEP + "+users" + SEP + r"|(?<![\w.])/(?:users|home)/[a-z]")
SCANNED = (".py", ".js", ".yml", ".yaml", ".ini", ".cfg", ".toml", ".html", ".txt")


def machine_paths():
    tracked = subprocess.run(["git", "ls-files"], cwd=layout.ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    hits = []
    for rel in tracked:
        if not rel.endswith(SCANNED) or rel == "tests/test_ci_portability.py":
            continue
        with open(os.path.join(layout.ROOT, rel), encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                if MACHINE_PATH.search(line):
                    hits.append(f"{rel}:{n}")
    return hits


def test_no_tracked_file_holds_a_path_from_one_machine():
    if not os.path.isdir(os.path.join(layout.ROOT, ".git")):
        pytest.skip("not a git checkout")
    assert machine_paths() == [], "derive paths from __file__ or core/paths.py instead"


def test_the_machine_path_pattern_catches_what_broke_run_1():
    for bad in (r"ROOT = 'c:\Users\dev\Facial-Recognition-Software'",
                "BASE = 'C:/Users/someone/x'", "p = '/home/runner/work'", "'/Users/me/src'"):
        assert MACHINE_PATH.search(bad), bad
    for fine in ("os.path.join(paths.dataset_dir(), 'users')", "url = 'https://x.org/home/a'",
                 "db.get_all_users()", "'api/users'"):
        assert not MACHINE_PATH.search(fine), fine
