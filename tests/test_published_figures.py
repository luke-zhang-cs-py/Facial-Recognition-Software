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
import glob
import io
import os
import re
import string
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PAGE = os.path.join(ROOT, "docs", "index.html")

# Directories that hold no shipped module. tests/ is excluded from coverage by
# the config too; the rest are never source.
#
# tools/ is deliberately *not* skipped. The five developer scripts used to sit
# in the root as `tools_*.py`, so this walk found them and the page had to
# list them -- under OMITTED, with a reason, which is the whole point of that
# block. Skipping the directory they moved into would have quietly dropped
# five files off a page whose first rule is that nothing is missing from it,
# and the only visible sign would have been the page getting shorter.
SKIP_DIRS = {".git", "__pycache__", "tests", "htmlcov", ".venv", "venv",
             "node_modules", "docs", "data", "static", "templates"}

# The page's own history: counts that describe what a figure *used to* say.
# These are the only numbers on the page allowed to be wrong, because being
# wrong is what they are about. Each one is asserted to still be present, so
# an entry cannot quietly outlive the sentence it exempts.
HISTORY = ()


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


# --------------------------------------------- does the page's script run?
# Every check below this point reads the page as text, and for a while that
# was the whole story. The result: this page shipped an unterminated string
# literal, its script threw SyntaxError on load, and *none* of the
# JavaScript ran -- empty tiles, empty tables, empty charts, and a full set
# of green tests, because every figure in the data block was perfectly
# accurate and nothing had ever asked whether the page could read it. Five
# pages in this family had it, from the same copied four lines.
#
# So: no string literal left open at the end of a line, and every element
# the script writes into exists in the markup. Neither needs a JavaScript
# engine, which these projects do not depend on and should not start
# depending on for a static page.


def published_pages():
    """Every HTML file served from docs/, not just the main one.

    A glob rather than the PAGE constant because more than one project here
    publishes a second page, and a check that names one file stops checking
    the moment a second appears.

    Recursive, because `docs/*.html` is not "every page in docs/" -- it is
    every page in the top level of it, and the generated demo lives in
    `docs/app/`. It was not covered, and it shipped with exactly the defect
    these two checks exist to catch: `renderCalibration` assigned into an id
    that was not in the markup, threw, and left the two charts and the risk
    table empty on a page that otherwise looked finished.
    """
    return sorted(glob.glob(os.path.join(ROOT, "docs", "**", "*.html"),
                            recursive=True))


def script_body(text):
    """Every <script> block in a page, concatenated."""
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", text, re.DOTALL)
    return "\n".join(blocks)


# A `/` after one of these is division; anywhere else it opens a regex
# literal. The distinction matters because a page carrying `/[&<>"]/g` would
# otherwise have that `"` read as the start of a string, and the rest of the
# line reported as unterminated.
_ENDS_A_VALUE = set(")]}") | set(string.ascii_letters + string.digits + "_$")

_CLOSERS = {"single": "'", "double": '"', "template": "`", "regex": "/"}
_OPENERS = {"'": "single", '"': "double", "`": "template"}


def _step_in_code(text, index, previous):
    """(state, next index, last significant char) for one character of code.

    The state "comment" means the rest of the line is one, which the caller
    takes as its cue to stop.
    """
    char = text[index]
    pair = text[index:index + 2]
    if pair == "//":
        return "comment", len(text), previous
    if pair == "/*":
        return "block", index + 2, previous
    if char == "/" and previous not in _ENDS_A_VALUE:
        return "regex", index + 1, previous
    if char in _OPENERS:
        return _OPENERS[char], index + 1, previous
    return "code", index + 1, previous if char.isspace() else char


def _step_in_quotes(text, index, state):
    """(state, next index, last significant char) inside a string or regex."""
    char = text[index]
    if char == "\\":
        return state, index + 2, ""
    if char == _CLOSERS[state]:
        # A closing quote or slash ends a value, so a `/` after it is
        # division rather than the start of another regex.
        return "code", index + 1, "x"
    return state, index + 1, ""


def _scan_line(text, state, previous):
    """Run the scanner to the end of one line; return where it ended up."""
    index = 0
    while index < len(text):
        if state == "code":
            state, index, previous = _step_in_code(text, index, previous)
            if state == "comment":
                return "code", previous
        elif state == "block":
            if text[index:index + 2] == "*/":
                state, index = "code", index + 2
            else:
                index += 1
        else:
            state, index, seen = _step_in_quotes(text, index, state)
            previous = seen or previous
    return state, previous


def open_string_lines(source):
    """Lines where a quoted string is still open at the newline.

    A hand-written scanner, because the alternatives are a regex (which
    cannot do this) or a JavaScript engine. It tracks quotes, both comment
    forms, regex literals and backslash escapes, and flags only single- and
    double-quoted strings -- a template literal spanning lines is legal. It
    reports the offending line's text rather than its number, which is the
    more useful half of the answer.
    """
    flagged = []
    state, previous = "code", ""
    for text in source.splitlines():
        state, previous = _scan_line(text, state, previous)
        if state in ("single", "double"):
            flagged.append(text.strip())
            state = "code"                       # do not cascade
    return flagged


def test_no_published_page_leaves_a_string_open_at_a_line_end():
    """The failure this whole file did not catch."""
    pages = published_pages()
    assert pages, "no page was found, so this check is vacuous"

    broken = {}
    for path in pages:
        with io.open(path, encoding="utf-8") as handle:
            source = script_body(handle.read())
        if not source.strip():
            continue
        flagged = open_string_lines(source)
        if flagged:
            broken[os.path.basename(path)] = flagged

    assert not broken, (
        "these pages leave a string literal open, so the browser throws "
        "SyntaxError and none of their script runs: %r" % broken)


def test_every_element_a_published_script_writes_into_exists():
    """A renamed id fails silently: `el(...)` returns null and the
    assignment throws, taking the rest of the script with it."""
    pages = published_pages()
    assert pages, "no page was found, so this check is vacuous"

    missing = {}
    checked = 0
    for path in pages:
        with io.open(path, encoding="utf-8") as handle:
            text = handle.read()
        source = script_body(text)
        wanted = set(re.findall(r"el\(\s*'([\w-]+)'\s*\)", source))
        wanted |= set(re.findall(
            r"getElementById\(\s*['\"]([\w-]+)['\"]\s*\)", source))
        if not wanted:
            continue
        checked += 1
        present = set(re.findall(r'id="([\w-]+)"', text))
        present |= set(re.findall(r"id='([\w-]+)'", text))
        absent = sorted(wanted - present)
        if absent:
            missing[os.path.basename(path)] = absent

    assert checked, "no el() calls found on any page, so this check is vacuous"
    assert not missing, (
        "these scripts write into elements that are not in the markup: %r"
        % missing)


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


def measured_with(page):
    """The interpreter the page says produced its figures, as "3.12"."""
    found = re.search(r"var MEASURED_WITH = 'Python (\d+\.\d+)'", page)
    return found.group(1) if found else None


def running():
    return "%d.%d" % sys.version_info[:2]


def test_every_line_count_on_the_page_is_the_measured_one(page):
    """Line counts, which do not depend on the interpreter."""
    wrong = []
    listed = dict(listed_modules(page))
    everything = [(name, body[0]) for name, body in listed.items()]
    everything += sorted(listed_omissions(page).items())
    for name, lines in sorted(everything):
        real_lines, _, _ = measure(name)
        if lines != real_lines:
            wrong.append(f"{name}: page says {lines} lines, measured "
                         f"{real_lines}")
    assert not wrong, "the published figures are stale:\n  " + "\n  ".join(wrong)


def test_every_statement_count_on_the_page_is_the_measured_one(page):
    """Statement counts, which do.

    coverage counts 190 statements in one of this family's modules under
    Python 3.14 and 191 under 3.12; another differs by four. A count is a
    property of a file *and* an interpreter, so this compares only when the
    running one is what the page names -- otherwise it would fail on CI for a
    page that is perfectly accurate.
    """
    stated = measured_with(page)
    assert stated, ("the page does not say which Python measured it, so its "
                    "statement counts cannot be checked against anything")
    if stated != running():
        pytest.skip(f"page measured with Python {stated}, running "
                    f"{running()} -- statement counts differ by version")

    wrong = []
    for name, (_, stmts, _) in sorted(listed_modules(page).items()):
        _, real_stmts, _ = measure(name)
        if stmts != real_stmts:
            wrong.append(f"{name}: page says {stmts} statements, measured "
                         f"{real_stmts}")
    assert not wrong, "the published figures are stale:\n  " + "\n  ".join(wrong)


@pytest.mark.skipif(not has_coverage_data(),
                    reason="no .coverage data file; run pytest --cov first")
def test_the_uncovered_counts_are_the_measured_ones(page):
    """Only when a run's data is on disk. Without it every statement looks
    uncovered, and a check against that would pass by being wrong twice."""
    stated = measured_with(page)
    if stated and stated != running():
        pytest.skip(f"page measured with Python {stated}, running "
                    f"{running()}")
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
    for sentence in HISTORY:
        assert sentence in page, (
            f"the history this check exempts is no longer on the page: "
            f"{sentence!r} -- take it out of HISTORY too")
        body = body.replace(sentence, "")
    typed = re.findall(r"[\d,]{2,}\s*(?:executable\s+)?(?:statements|tests)\b",
                       body)
    assert not typed, (
        f"these counts are typed into the page rather than computed from the "
        f"data block: {typed}")
