"""tools/refresh_figures.py, against a miniature project in tmp_path.

The real script runs the whole suite three times -- plain, under coverage,
and --collect-only -- and rewrites docs/index.html, the README and
CONTRIBUTING. Here ROOT, PAGE, README and CONTRIBUTING point into tmp_path,
and subprocess.run is a fake that prints what pytest would have printed (or
writes the JSON report coverage would have written), so each test is a
millisecond and no tracked file is touched. The page below is the data
block of docs/index.html, cut down to two modules and two test files.
"""
import json
import os
import subprocess
import sys

import pytest

from tools import refresh_figures as rf

PAGE = """<script>
var MODULES = [
  { name: 'app.py', lines: 1, stmts: 1, missed: 0, does: 'the app' },
  { name: 'core/db.py', lines: 1, stmts: 1, missed: 0, does: 'the db' }
];

var OMITTED = [
  { name: 'tools/x.py', lines: 1, why: 'a script' }
];

var TESTS = [
  { file: 'test_a.py', n: 1, of: 'the first' },
  { file: 'test_b.py', n: 1, of: 'the second' }
];

var TEST_LINES = 1;
var MEASURED_WITH = 'Python 2.7';
</script>
"""

MEASURED = {"app.py": {"lines": 10, "stmts": 5, "missed": 0},
            "core/db.py": {"lines": 20, "stmts": 9, "missed": 1}}
COUNTS = {"test_a.py": 2, "test_b.py": 1}
VERSION = "%d.%d" % sys.version_info[:2]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A root with a page, a README, a CONTRIBUTING and two test files."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.html").write_text(PAGE, encoding="utf-8")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "x.py").write_text("a\nb\nc\n", encoding="utf-8")
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "db.py").write_text("x\n" * 20, encoding="utf-8")
    (tmp_path / "app.py").write_text("x\n" * 10, encoding="utf-8")
    tests = tmp_path / "tests"
    (tests / "__pycache__").mkdir(parents=True)
    (tests / "test_a.py").write_text("1\n2\n3\n", encoding="utf-8")
    (tests / "golden").mkdir()
    (tests / "golden" / "g.py").write_text("1\n2\n", encoding="utf-8")
    (tests / "golden" / "data.json").write_text("{}\n" * 9, encoding="utf-8")
    (tests / "__pycache__" / "skip.py").write_text("1\n" * 50,
                                                   encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# x\n\n1 tests, 50% of 2 statements, needing nothing.\n",
        encoding="utf-8")
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "CONTRIBUTING.md").write_text(
        "Run them.\n\n1 tests locally; 1 pass and 0 skip here.\n",
        encoding="utf-8")
    monkeypatch.setattr(rf, "ROOT", str(tmp_path))
    monkeypatch.setattr(rf, "PAGE", str(tmp_path / "docs" / "index.html"))
    monkeypatch.setattr(rf, "README", str(tmp_path / "README.md"))
    monkeypatch.setattr(rf, "CONTRIBUTING",
                        str(tmp_path / ".github" / "CONTRIBUTING.md"))
    monkeypatch.setattr(rf, "_OUTCOME", {})
    return tmp_path


class Done(object):
    def __init__(self, stdout):
        self.stdout = stdout


def fake_run(monkeypatch, stdout="", report=None):
    calls = []

    def run(argv, cwd, capture_output, text, timeout):
        calls.append({"argv": argv, "cwd": cwd})
        for arg in argv:
            if arg.startswith("--cov-report=json:"):
                with open(arg[len("--cov-report=json:"):], "w") as handle:
                    json.dump(report, handle)
        return Done(stdout)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


# ------------------------------------------------------------ measuring

def test_read_and_write_round_trip_with_lf(tmp_path):
    path = str(tmp_path / "f.txt")
    rf.write(path, "a\nb\n")
    with open(path, "rb") as handle:
        assert handle.read() == b"a\nb\n"
    assert rf.read(path) == "a\nb\n"


def test_outcome_reads_the_pass_skip_split(project, monkeypatch):
    calls = fake_run(monkeypatch,
                     "FAILED tests/test_published_figures.py::test_x\n"
                     "2 failed, 900 passed, 7 skipped in 3s\n")
    assert rf.outcome() == {"passed": 900, "skipped": 7, "failed": 2}
    assert calls[0]["argv"][1:] == ["-m", "pytest", "-q", "-p",
                                    "no:cacheprovider"]
    assert calls[0]["cwd"] == str(project)


def test_outcome_counts_nothing_when_nothing_ran(project, monkeypatch):
    fake_run(monkeypatch, "no tests ran in 0.01s\n")
    assert rf.outcome() == {"passed": 0, "skipped": 0, "failed": 0}


def test_outcome_refuses_to_measure_a_broken_suite(project, monkeypatch):
    fake_run(monkeypatch, "FAILED tests/test_db.py::test_add - boom\n"
                          "FAILED tests/test_published_figures.py::test_y\n"
                          "2 failed, 1 passed\n")
    with pytest.raises(SystemExit) as stop:
        rf.outcome()
    assert "broken suite" in str(stop.value)
    assert "test_db.py::test_add" in str(stop.value)
    assert "test_published_figures.py::test_y" not in str(stop.value)


def test_measure_reads_the_json_report_and_deletes_it(project, monkeypatch):
    report = {"files": {
        "app.py": {"summary": {"num_statements": 5, "missing_lines": 0}},
        os.path.join("core", "db.py"): {
            "summary": {"num_statements": 9, "missing_lines": 1}}}}
    calls = fake_run(monkeypatch, report=report)
    assert rf.measure() == MEASURED
    path = [a for a in calls[0]["argv"]
            if a.startswith("--cov-report=json:")][0].split(":", 1)[1]
    assert "--cov" in calls[0]["argv"]
    assert not os.path.exists(path), "the temporary report is removed"


def test_collect_counts_tests_per_file(project, monkeypatch):
    fake_run(monkeypatch, "tests/test_a.py::test_x\n"
                          "tests/test_a.py::test_y[1]\n"
                          "tests/test_b.py::test_z\n"
                          "\n3 tests collected in 0.1s\n")
    assert rf.collect() == COUNTS


def test_collect_refuses_an_empty_collection(project, monkeypatch):
    fake_run(monkeypatch, "ERROR collecting tests/test_a.py\n")
    with pytest.raises(SystemExit) as stop:
        rf.collect()
    assert "collection produced nothing" in str(stop.value)
    assert "ERROR collecting" in str(stop.value)


# ------------------------------------------------------------- the page

def test_fix_modules_rewrites_figures_largest_first():
    changes = []
    page = rf.fix_modules(PAGE, MEASURED, changes)
    assert ("  { name: 'core/db.py', lines:  20, stmts:   9, missed:  1, "
            "does: 'the db' },\n"
            "  { name: 'app.py',     lines:  10, stmts:   5, missed:  0, "
            "does: 'the app' }\n];") in page
    assert [line.split() for line in changes] == [
        ["core/db.py", "(1,", "1,", "0)", "->", "(20,", "9,", "1)"],
        ["app.py", "(1,", "1,", "0)", "->", "(10,", "5,", "0)"]]
    again = []
    assert rf.fix_modules(page, MEASURED, again) == page and again == []


def test_fix_modules_needs_a_person_to_describe_a_new_module():
    measured = dict(MEASURED, **{"tools/new.py": {"lines": 1, "stmts": 1,
                                                  "missed": 0}})
    with pytest.raises(SystemExit) as stop:
        rf.fix_modules(PAGE, measured, [])
    assert "no description on the page for tools/new.py" in str(stop.value)


def test_fix_modules_flags_a_module_coverage_stopped_reporting():
    changes = []
    page = rf.fix_modules(PAGE, {"app.py": MEASURED["app.py"]}, changes)
    assert ("core/db.py is on the page but coverage no longer reports it"
            in changes[0])
    assert "core/db.py" not in page.split("var OMITTED")[0]


def test_fix_omitted_measures_line_counts(project):
    changes = []
    page = rf.fix_omitted(PAGE, changes)
    assert "  { name: 'tools/x.py', lines:   3, why: 'a script' }\n];" in page
    assert [line.split() for line in changes] == [
        ["tools/x.py", "1", "->", "3", "lines"]]
    assert rf.fix_omitted(page, []) == page


def test_fix_omitted_without_a_block_leaves_the_page_alone(project):
    page = PAGE.replace("var OMITTED = [\n  { name: 'tools/x.py', lines: 1, "
                        "why: 'a script' }\n];", "var OMITTED = [];")
    assert rf.fix_omitted(page, []) == page


def test_fix_omitted_refuses_a_file_that_is_gone(project):
    os.remove(str(project / "tools" / "x.py"))
    with pytest.raises(SystemExit) as stop:
        rf.fix_omitted(PAGE, [])
    assert "not in the project: tools/x.py" in str(stop.value)


def test_fix_tests_rewrites_counts_and_flags_the_missing():
    changes = []
    page = rf.fix_tests(PAGE, {"test_a.py": 2}, changes)
    assert "  { file: 'test_a.py', n:  2, of: 'the first' }\n];" in page
    assert changes[0] == "test_b.py is on the page but collects nothing"
    assert changes[1].split() == ["test_a.py", "1", "->", "2"]
    with pytest.raises(SystemExit) as stop:
        rf.fix_tests(PAGE, {"test_c.py": 1}, [])
    assert "no description on the page for test_c.py" in str(stop.value)


def test_fix_measured_with_records_this_interpreter():
    changes = []
    page = rf.fix_measured_with(PAGE, changes)
    assert "var MEASURED_WITH = 'Python %s';" % VERSION in page
    assert changes == ["MEASURED_WITH Python 2.7 -> %s" % VERSION]
    again = []
    assert rf.fix_measured_with(page, again) == page and again == []
    with pytest.raises(SystemExit) as stop:
        rf.fix_measured_with("var MODULES = [];", [])
    assert "no MEASURED_WITH line" in str(stop.value)


def test_fix_test_lines_counts_python_under_tests_only(project):
    """test_a.py's 3 and golden/g.py's 2: not the JSON, not __pycache__."""
    changes = []
    page = rf.fix_test_lines(PAGE, changes)
    assert "var TEST_LINES = 5;" in page and changes == ["TEST_LINES 1 -> 5"]
    again = []
    assert rf.fix_test_lines(page, again) == page and again == []
    assert rf.fix_test_lines("no figure here", []) == "no figure here"


# ------------------------------------------------------- the markdown

def test_fix_readme_rewrites_its_one_sentence(project):
    changes = []
    rf.fix_readme(MEASURED, COUNTS, changes)
    assert changes == ["README  1 tests, 50% of 2 statements -> "
                       "3 tests, 93% of 14 statements"]
    text = (project / "README.md").read_text(encoding="utf-8")
    assert "3 tests, 93% of 14 statements, needing nothing." in text
    again = []
    rf.fix_readme(MEASURED, COUNTS, again)
    assert again == []


def test_fix_readme_leaves_a_readme_that_states_no_figures(project):
    (project / "README.md").write_text("# nothing\n", encoding="utf-8")
    rf.fix_readme(MEASURED, COUNTS, [])
    assert (project / "README.md").read_text(encoding="utf-8") == "# nothing\n"
    os.remove(str(project / "README.md"))
    rf.fix_readme(MEASURED, COUNTS, [])
    assert not (project / "README.md").exists()


def test_fix_contributing_uses_the_plain_runs_skips(project, monkeypatch):
    monkeypatch.setattr(rf, "_OUTCOME", {"skipped": 1})
    changes = []
    rf.fix_contributing(COUNTS, changes)
    text = (project / ".github" / "CONTRIBUTING.md").read_text("utf-8")
    assert ("3 tests locally; 2 pass and 1 skip here (`-rs` says why)."
            in text)
    assert len(changes) == 1 and changes[0].startswith("CONTRIBUTING  1 tests")
    again = []
    rf.fix_contributing(COUNTS, again)
    assert again == []


def test_fix_contributing_leaves_a_file_without_the_sentence(project):
    path = project / ".github" / "CONTRIBUTING.md"
    path.write_text("Be kind.\n", encoding="utf-8")
    rf.fix_contributing(COUNTS, [])
    assert path.read_text(encoding="utf-8") == "Be kind.\n"
    os.remove(str(path))
    rf.fix_contributing(COUNTS, [])
    assert not path.exists()


# ---------------------------------------------------------------- main

@pytest.fixture
def measured_run(project, monkeypatch):
    monkeypatch.setattr(rf, "outcome",
                        lambda: rf._OUTCOME.update(skipped=0) or rf._OUTCOME)
    monkeypatch.setattr(rf, "measure", lambda: MEASURED)
    monkeypatch.setattr(rf, "collect", lambda: COUNTS)
    return project


def test_main_rewrites_every_stale_figure_and_prints_what_changed(
        measured_run, capsys):
    rf.main()
    out = capsys.readouterr().out
    page = (measured_run / "docs" / "index.html").read_text("utf-8")
    assert "lines:  20, stmts:   9, missed:  1" in page
    assert "var TEST_LINES = 5;" in page
    assert "var MEASURED_WITH = 'Python %s';" % VERSION in page
    assert "3 tests, 93% of 14 statements" in (
        measured_run / "README.md").read_text("utf-8")
    assert "  README  1 tests" in out and "  TEST_LINES 1 -> 5" in out
    assert out.splitlines()[-1] == "  3 tests, 14 statements, 93% covered"


def test_main_on_a_page_that_is_already_right_changes_nothing(
        measured_run, monkeypatch, capsys):
    rf.main()
    capsys.readouterr()
    writes = []
    monkeypatch.setattr(rf, "write", lambda path, text: writes.append(path))
    rf.main()
    assert writes == []
    assert capsys.readouterr().out == "  3 tests, 14 statements, 93% covered\n"


def test_main_refuses_to_write_from_a_rewrite_that_is_not_idempotent(
        measured_run, monkeypatch):
    monkeypatch.setattr(rf, "rewrite",
                        lambda page, measured, counts, changes: page + "x")
    with pytest.raises(SystemExit) as stop:
        rf.main()
    assert "not idempotent" in str(stop.value)
    assert (measured_run / "docs" / "index.html").read_text("utf-8") == PAGE
