"""The numbers more than one module depends on.

`vision.py` exists because these were written out separately in up to six
places each. That is a smell on its own, but one of them was a live bug:
`analytics.py` returns `currentThreshold` in its report, and four things
consume it -- the sweep table's "current" marker, the CLI's
"attendance.py has CONFIDENCE_THRESHOLD = N" advice line, and two places in
the web UI. All four read a typed literal `70`. Change the threshold the
recogniser actually uses and every one of them keeps reporting 70, so the
tool whose job is to tell you your configuration misreports it, and its
recommendation is computed against a baseline that no longer exists.

These tests are the price of having one value in one place: they fail if a
module goes back to declaring its own.
"""
import ast
import io
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import layout                                             # noqa: E402

from core import vision                                   # noqa: E402

# The module the shared numbers live in, root-relative. Named once because
# three checks below exempt it: it is the one file allowed to write them out.
VISION = "core/vision.py"

# Modules that read the shared geometry. tools/ holds developer scripts and
# is deliberately not in scope; tests/ is where literals belong.
#
# This was `os.listdir(ROOT)` filtered by `not name.startswith("tools")`,
# which said "every shipped module" only for as long as every shipped module
# sat in the root. Once they moved into core/, pipeline/, analysis/ and cli/
# it still returned a list -- `["app.py"]` -- and every loop below still ran,
# still passed, and checked one file out of twenty-six. layout.shipped_modules()
# walks the packages and asserts it found something, so the scan cannot
# quietly shrink again.
SHIPPED = layout.shipped_modules()


def code_of(name):
    """A module's source with docstrings and comments removed.

    Every guard in this family that grepped raw source got a false pass or a
    false failure from prose describing the very thing being searched for --
    `core/vision.py`'s own docstring quotes `CONFIDENCE_THRESHOLD = 70`, and
    three other modules explain the history in comments. Parse, then
    unparse, and only executable code is left.
    """
    tree = ast.parse(layout.source_of(name))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_only_vision_puts_a_number_on_the_confidence_threshold():
    """Everyone else has to get it from there.

    The assertion is about the *literal*: `CONFIDENCE_THRESHOLD = 70` may
    appear once, in vision.py. `CONFIDENCE_THRESHOLD = vision.<...>` is a
    re-export and is fine anywhere.
    """
    offenders = []
    for name in SHIPPED:
        for line in code_of(name).splitlines():
            match = re.match(r"\s*CONFIDENCE_THRESHOLD\s*=\s*(.+)", line)
            if match and "vision." not in match.group(1):
                offenders.append("%s: %s" % (name, line.strip()))

    assert offenders == ["%s: CONFIDENCE_THRESHOLD = 70" % VISION], (
        "the threshold is defined outside %s: %s" % (VISION, offenders))


def test_the_analysis_reports_the_threshold_it_reads_rather_than_a_literal():
    """The bug itself, asserted by behaviour.

    Monkeypatching is not enough here -- `analytics` re-exports nothing and
    reads `vision.CONFIDENCE_THRESHOLD` at call time, which is the property
    under test. If someone puts the literal back, this sees the old value
    while vision says the new one.
    """
    from analysis import analytics

    source = code_of("analysis/analytics.py")
    assert '"currentThreshold": 70' not in source.replace("'", '"'), (
        "analysis/analytics.py reports a typed threshold again")
    assert "vision.CONFIDENCE_THRESHOLD" in source, (
        "analysis/analytics.py no longer reads the shared threshold")

    # And the value that reaches the report is the shared one, whatever it is.
    assert analytics.vision.CONFIDENCE_THRESHOLD is vision.CONFIDENCE_THRESHOLD


def test_the_advice_line_and_the_recogniser_cannot_disagree():
    """`analyze_faces.py` prints "attendance.py has CONFIDENCE_THRESHOLD = N".
    That sentence names a module, so it had better be that module's value."""
    from cli import attendance

    assert attendance.CONFIDENCE_THRESHOLD == vision.CONFIDENCE_THRESHOLD


def test_every_module_that_matches_a_face_uses_the_same_threshold():
    from cli import attendance
    from pipeline import camera

    assert camera.CONFIDENCE_THRESHOLD == attendance.CONFIDENCE_THRESHOLD
    # Identity, not equality: two separately declared numbers that happen to
    # be equal is exactly the state this replaced.
    assert camera.CONFIDENCE_THRESHOLD is vision.CONFIDENCE_THRESHOLD
    assert attendance.CONFIDENCE_THRESHOLD is vision.CONFIDENCE_THRESHOLD


def test_no_shipped_module_writes_out_the_crop_size():
    """A gallery captured at one size and a query resized to another are not
    comparable -- LBPH histograms are computed over a fixed grid. Six call
    sites agreed on (200, 200) by everyone copying the same line, and a
    seventh that did not would have shown up only as worse accuracy."""
    offenders = []
    for name in SHIPPED:
        if name == VISION:
            continue
        source = code_of(name)
        if re.search(r"\(\s*200\s*,\s*200\s*\)", source):
            offenders.append(name)
    assert not offenders, (
        "these write out the LBPH crop size instead of importing it: %s"
        % offenders)


def test_no_shipped_module_writes_out_the_detector_tuning():
    """scaleFactor, minNeighbors and minSize only mean anything as a set: a
    smaller scale factor with the same neighbour count is a different
    detector, not a slightly slower one."""
    offenders = []
    for name in SHIPPED:
        if name == VISION:
            continue
        source = code_of(name)
        if re.search(r"minSize\s*=\s*\(\s*80\s*,\s*80\s*\)", source):
            offenders.append("%s: minSize" % name)
        if re.search(r"scaleFactor\s*=\s*1\.1", source):
            offenders.append("%s: scaleFactor" % name)
    assert not offenders, (
        "these write out detector tuning instead of importing it: %s"
        % offenders)


@pytest.mark.parametrize("name,value,why", [
    ("CONFIDENCE_THRESHOLD", 70, "an LBPH distance, so lower is a better match"),
    ("LBPH_INPUT_SIZE", (200, 200), "the fixed grid LBPH is trained on"),
    ("MIN_FACE_SIZE", (80, 80), "the smallest face worth recognising"),
    ("DETECT_SCALE_FACTOR", 1.1, "10% steps between detector scales"),
    ("DETECT_MIN_NEIGHBOURS", 5, "overlapping detections before it counts"),
    ("STREAM_JPEG_QUALITY", 80, "preview only; stored frames do not use it"),
])
def test_the_shared_values_are_what_the_pipeline_was_built_around(name, value,
                                                                  why):
    """A regression guard on the values themselves.

    Moving six numbers into one module is the kind of refactor that is
    supposed to change nothing, and the way to say so is to write down what
    they were. Each is documented in `vision.py`; `why` is here so a failure
    says what the number means rather than only that it moved.
    """
    assert getattr(vision, name) == value, why


def test_the_shared_module_holds_nothing_but_constants():
    """`paths.py` has a `use()` because paths are redirected by tests. These
    are not: a function here would be a second place for behaviour to live,
    and the module is imported by six others precisely because it is inert.
    """
    tree = ast.parse(layout.source_of(VISION))
    kinds = {type(node).__name__ for node in tree.body}
    assert kinds <= {"Expr", "Assign", "AnnAssign", "ImportFrom", "Import"}, (
        "%s has grown something other than constants: %s" % (VISION, kinds))
