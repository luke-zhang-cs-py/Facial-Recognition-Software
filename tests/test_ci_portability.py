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
