"""The published page says its figures were measured. This checks that.

`docs/index.html` is served to the public from GitHub Pages, and every number
on it — the tiles, the charts, the module table — is read from one DATA block
near the bottom of the file. The block says the figures came from
`pytest --cov`, `wc -l` and `pytest --collect-only`, and were not estimated.

A page like that goes stale the moment a test is added, and this repository
has already published a coverage figure measured over the wrong set of
files: with no `source` setting, coverage reported only what something
imported, so `attendance.py` and `register_user.py` were missing from the
report entirely rather than showing as uncovered. So the figures are
measured again here and compared.

What is checked statically, so it works in CI with no coverage run:

  * every module the page lists exists, with the line count it claims;
  * its statement count is what `coverage` counts, using this repo's own
    `.coveragerc` — which is what keeps the corpus tools and the model
    downloader out, and everything else in;
  * no module in the project is missing from the page;
  * every test file and its collected count.

`missed` needs an actual run, so it is verified only when a `.coverage` data
file is present and skipped when it is not, rather than asserted against an
empty measurement — which would "pass" by calling everything uncovered.

`python tools/refresh_figures.py` rewrites whatever this finds wrong.
"""
import fnmatch
import io
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PAGE = os.path.join(ROOT, "docs", "index.html")

# Directories that hold no shipped module. tests/ is excluded from coverage by
# the config too; the rest are never source.
SKIP_DIRS = {".git", "__pycache__", "tests", "htmlcov", ".venv", "venv",
             "node_modules", "docs", "tools", "data", "static", "templates"}


@pytest.fixture(scope="module")
def page():
    with io.open(PAGE, encoding="utf-8") as handle:
        return handle.read()


def listed_modules(page):
    """name -> (lines, stmts, missed), as the page states them."""
    found = re.findall(
        r"\{ name: '([\w./]+)',\s*lines:\s*(\d+),\s*stmts:\s*(\d+),"
        r"\s*missed:\s*(\d+),", page)
    return {name: (int(lines), int(stmts), int(missed))
            for name, lines, stmts, missed in found}


def listed_omissions(page):
    """The modules the page says are left out of coverage on purpose."""
    return dict((name, int(lines)) for name, lines in re.findall(
        r"\{ name: '([\w./]+)',\s*lines:\s*(\d+),\s*why:", page))


def source_files():
    """Every shipped module, relative to the root, with forward slashes."""
    out = []
    for here, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if name.endswith(".py"):
                out.append(os.path.relpath(os.path.join(here, name),
                                           ROOT).replace("\\", "/"))
    return set(out)


def omit_patterns():
    """The omit patterns from .coveragerc, forward-slashed.

    Matched with fnmatch rather than compared as strings: this project omits
    `tools_*.py`, and an equality check called that "not omitted" for all
    five files the pattern covers.
    """
    import coverage
    config = os.path.join(ROOT, ".coveragerc")
    cov = coverage.Coverage(config_file=config if os.path.exists(config)
                            else True)
    return [pattern.replace("\\", "/")
            for pattern in cov.config.run_omit or ()]


def is_omitted(name):
    return any(fnmatch.fnmatch(name, pattern) for pattern in omit_patterns())


def measure(name):
    import coverage
    config = os.path.join(ROOT, ".coveragerc")
    cov = coverage.Coverage(config_file=config if os.path.exists(config)
                            else True)
    cov.load()
    path = os.path.join(ROOT, name.replace("/", os.sep))
    _, statements, _, missing, _ = cov.analysis2(path)
    with io.open(path, encoding="utf-8") as handle:
        lines = len(handle.read().splitlines())
    return lines, len(statements), missing


def has_coverage_data():
    """Whether a real run's data is on disk to check `missed` against."""
    import coverage
    cov = coverage.Coverage(config_file=True)
    try:
        cov.load()
        return bool(cov.get_data().measured_files())
    except Exception:                                  # pragma: no cover
        return False


def test_the_page_has_a_data_block_at_all(page):
    """A guard on the guard: every check below reads this block, and a loop
    over nothing passes. Whether the block is *complete* is the next test's
    job -- this one only establishes that it was read at all, which is the
    failure that would make the rest of the file vacuous."""
    assert listed_modules(page), (
        "the MODULES block could not be read, so none of these checks mean "
        "anything")
    assert "var TESTS = [" in page, (
        "the TESTS block could not be read either")


def test_every_module_is_on_the_page(page):
    """A module the page omits is worse than a wrong number: the chart of
    module sizes silently stops including it and the total under it is short
    by however much it holds."""
    listed = set(listed_modules(page)) | set(listed_omissions(page))
    present = source_files()
    assert present - listed == set(), (
        f"these modules are in the project but not on the page: "
        f"{sorted(present - listed)}")
    assert listed - present == set(), (
        f"the page lists modules that no longer exist: "
        f"{sorted(listed - present)}")


def test_the_omitted_modules_really_are_omitted(page):
    """If a page claims a module is left out of coverage on purpose and the
    config does not omit it, the omission has become an accident."""
    claimed = listed_omissions(page)
    if not claimed:
        pytest.skip("this page claims no omissions")
    for name in claimed:
        assert is_omitted(name), (
            f"the page says {name} is omitted from coverage, but nothing in "
            f".coveragerc matches it: {omit_patterns()}")


def test_every_figure_on_the_page_is_the_measured_one(page):
    wrong = []
    for name, (lines, stmts, missed) in sorted(listed_modules(page).items()):
        real_lines, real_stmts, missing = measure(name)
        if lines != real_lines:
            wrong.append(f"{name}: page says {lines} lines, measured "
                         f"{real_lines}")
        if stmts != real_stmts:
            wrong.append(f"{name}: page says {stmts} statements, measured "
                         f"{real_stmts}")
    for name, lines in sorted(listed_omissions(page).items()):
        real_lines, _, _ = measure(name)
        if lines != real_lines:
            wrong.append(f"{name}: page says {lines} lines, measured "
                         f"{real_lines}")
    assert not wrong, "the published figures are stale:\n  " + "\n  ".join(wrong)


@pytest.mark.skipif(not has_coverage_data(),
                    reason="no .coverage data file; run pytest --cov first")
def test_the_uncovered_counts_are_the_measured_ones(page):
    """Only when a run's data is on disk. Without it every statement looks
    uncovered, and a check against that would pass by being wrong twice."""
    wrong = []
    for name, (_, _, missed) in sorted(listed_modules(page).items()):
        _, _, missing = measure(name)
        if missed != len(missing):
            wrong.append(f"{name}: page says {missed} uncovered, measured "
                         f"{len(missing)}")
    assert not wrong, (
        "the coverage figures on the page are stale:\n  " + "\n  ".join(wrong))


def test_every_test_file_and_its_count_are_the_real_ones(page):
    """A `--collect-only` run rather than counting `def test_` by hand,
    because a parameterised test is one definition and many tests."""
    listed = dict((name, int(count)) for name, count in re.findall(
        r"\{ file: '([\w.]+)',\s*n:\s*(\d+),", page))
    assert listed, "the TESTS block could not be read at all"

    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    real = {}
    for line in done.stdout.splitlines():
        if "::" in line:
            name = os.path.basename(line.split("::")[0])
            real[name] = real.get(name, 0) + 1

    assert real, ("collection produced nothing, so this test would pass "
                  "whatever the page said:\n" + done.stdout[-500:])
    assert listed == real, (
        f"the page's test table does not match the suite.\n"
        f"  page:      {sorted(listed.items())}\n"
        f"  collected: {sorted(real.items())}")


def test_the_page_says_when_it_was_measured(page):
    """`missed` cannot be checked in CI, so the page has to date it. A
    coverage percentage with no date attached reads as current forever."""
    assert re.search(r"measured on \d{1,2} \w+ 20\d\d", page), (
        "the page no longer says when its figures were measured")


def test_no_count_is_typed_outside_the_data_block(page):
    """A number typed into a sentence beside a number computed from the data
    is how the two come to disagree. The data block may hold counts;
    everything else must derive them."""
    body = re.sub(r"var MODULES = \[.*?\n\];", "", page, flags=re.DOTALL)
    body = re.sub(r"var TESTS = \[.*?\n\];", "", body, flags=re.DOTALL)
    body = re.sub(r"var OMITTED = \[.*?\n\];", "", body, flags=re.DOTALL)
    # The suite's own history, which is about being wrong and has to stay.
    body = body.replace("38 tests", "").replace("Six red CI runs", "")
    typed = re.findall(r"[\d,]{2,}\s*(?:executable\s+)?(?:statements|tests)\b",
                       body)
    assert not typed, (
        f"these counts are typed into the page rather than computed from the "
        f"data block: {typed}")
