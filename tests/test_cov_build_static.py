"""tools/build_static.py, with a stand-in for the browser.

The real build starts Chromium through Playwright and runs the ported
JavaScript against the Python. Neither Playwright nor Node is installed on
the CI runner, and a test has no business launching a browser, so
`playwright.sync_api` is replaced in sys.modules by FakeChromium below.

What that stand-in can and cannot show. It does not run JavaScript, so it
says nothing about whether js/guidance.js is a faithful port -- that is the
build's own job, run by hand. It answers each evaluate() from the Python
reference, and answers *wrongly* when the page it was handed carries one of
the SABOTAGE edits, in the way that edit would. That is enough to test
everything around the browser: that the comparison reports a difference
with the path to it, that every refusal stops the build before a file is
written, that --prove notices a breakage the checks let through, and that
a clean run writes exactly the demo that is published in docs/.
"""
import copy
import io
import os
import sys
import types

import pytest

from analysis import calibration
from pipeline import guidance, landmarks
from tools import build_static as bs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def constants():
    return bs.collect_constants()


@pytest.fixture(scope="module")
def written(constants):
    return bs.collect(constants)


@pytest.fixture(scope="module")
def bundle(constants):
    cases, spec = bs.guidance_cases(), bs.calibration_cases()
    return (cases, spec, bs.python_guidance(cases),
            bs.python_calibration(spec), constants)


# ------------------------------------------------------------ the browser

# How each sabotage shows up in what the page answers.
KIND = {"reorder-adjacent": "order", "reorder-distant": "order",
        "shift-threshold": "guidance", "shift-measurement": "calibration",
        "severity": "short"}
for _name, _target, _edits, _why in bs.SABOTAGE:
    KIND.setdefault(_name, "calibration" if _target == "js/calibration.js"
                    else "guidance")


class Message(object):
    def __init__(self, kind, text):
        self.type, self.text = kind, text


class FakePage(object):
    def __init__(self, browser):
        self.browser = browser
        self.handlers = {}
        self.html = None
        self.closed = False

    def on(self, event, handler):
        self.handlers[event] = handler

    def set_content(self, html):
        self.html = html
        self.handlers["console"](Message("log", "loaded"))
        if self.browser.throw_on_load:
            self.handlers["pageerror"]("SyntaxError: Unexpected token")

    def evaluate(self, script, arg=None):
        browser = self.browser
        if browser.log_error:
            self.handlers["console"](Message("error", "TypeError: x is null"))
        kind = browser.kind_of(self.html)
        if script == bs.ORDER_JS:
            order = list(browser.order)
            if kind == "order":
                order[0], order[1] = order[1], order[0]
            return order
        if script == bs.GUIDANCE_JS:
            out = copy.deepcopy(bs.python_guidance(arg))
            if kind == "guidance":
                out[0]["checklist"] = out[0]["checklist"][1:]
            elif kind == "short":
                out = out[:-1]
            return out
        assert script == bs.CALIBRATION_JS, "a script this fake does not know"
        out = copy.deepcopy(browser.calibration(arg))
        if kind == "calibration":
            out["fmr"][0] += 1e-9
        return out

    def close(self):
        self.closed = True


class FakeChromium(object):
    """Playwright's sync API, as much of it as build_static touches."""

    def __init__(self, clean, survive=(), throw_on_load=False,
                 log_error=False):
        self.order = bs.collect_constants()["ruleOrder"]
        self.survive = set(survive)
        self.throw_on_load = throw_on_load
        self.log_error = log_error
        self.launched = []
        self.pages = []
        self.closed = False
        self._calibration = {}
        # Each sabotaged file, as text, -> the sabotage that produced it.
        self.sabotages = {}
        for entry in bs.SABOTAGE:
            broken = bs.sabotaged(clean, entry)[entry[1]]
            self.sabotages[broken] = entry[0]

    def kind_of(self, html):
        for text, name in self.sabotages.items():
            if text in html and name not in self.survive:
                return KIND[name]
        return None

    def calibration(self, spec):
        key = id(spec)
        if key not in self._calibration:
            self._calibration[key] = bs.python_calibration(spec)
        return self._calibration[key]

    # sync_playwright() -> pw; pw.chromium.launch() -> this
    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def chromium(self):
        return self

    def launch(self, **options):
        self.launched.append(options)
        return self

    def new_page(self):
        page = FakePage(self)
        self.pages.append(page)
        return page

    def close(self):
        self.closed = True


@pytest.fixture
def browser(monkeypatch, written, tmp_path):
    """Install a FakeChromium as playwright.sync_api; no Chrome on disk."""
    fake = FakeChromium(written)
    install(monkeypatch, fake)
    monkeypatch.setattr(bs, "CHROME", str(tmp_path / "no-chrome.exe"))
    return fake


def install(monkeypatch, fake):
    api = types.ModuleType("playwright.sync_api")
    api.sync_playwright = fake
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", api)


# ------------------------------------------------------- small helpers

def test_replace_once_refuses_an_anchor_that_moved():
    assert bs.replace_once("a-b", "-", "+", "join") == "a+b"
    for text in ("ab", "a-b-c"):
        with pytest.raises(SystemExit) as stop:
            bs.replace_once(text, "-", "+", "join the halves")
        assert "could not join the halves" in str(stop.value)
        assert str(stop.value).startswith("build_static: ")


def test_write_makes_its_folder_and_writes_lf(tmp_path):
    path = str(tmp_path / "deep" / "er" / "x.js")
    bs.write(path, "one\ntwo\n")
    with io.open(path, "rb") as handle:
        assert handle.read() == b"one\ntwo\n"
    assert bs.read(path) == "one\ntwo\n"


def test_the_constants_are_read_off_the_modules(constants):
    assert constants["guidance"]["MAX_YAW"] == guidance.MAX_YAW
    assert set(constants["guidance"]) == set(bs.GUIDANCE_NAMES)
    assert constants["ruleOrder"][0] == "no_face"
    assert len(constants["ruleOrder"]) == len(guidance.RULES) == 14
    assert constants["calibration"]["fmr"] == [
        [t, f] for t, f in calibration.SFACE_FMR]


def test_a_constant_missing_from_guidance_stops_the_build(monkeypatch):
    monkeypatch.delattr(guidance, "MAX_ROLL")
    with pytest.raises(SystemExit) as stop:
        bs.collect_constants()
    assert "no longer defines MAX_ROLL" in str(stop.value)


def test_a_landmark_constant_missing_stops_the_build(monkeypatch):
    assert bs.collect_landmark_constants()["EAR_CLOSED"] == landmarks.EAR_CLOSED
    monkeypatch.delattr(landmarks, "MAR_OPEN")
    with pytest.raises(SystemExit) as stop:
        bs.collect_landmark_constants()
    assert "pipeline/landmarks.py no longer defines MAR_OPEN" in str(stop.value)


def test_the_constants_files_carry_the_exact_numbers(constants):
    import json
    text = bs.constants_file(constants)
    line = [x for x in text.splitlines()
            if x.startswith("const CALIBRATION_DATA = ")][0]
    data = json.loads(line[len("const CALIBRATION_DATA = "):-1])
    assert data["fmr"] == constants["calibration"]["fmr"], "nothing rounded"
    assert constants["calibration"]["corpus"] in text
    lm = bs.landmark_constants_file({"EAR_CLOSED": 0.2})
    assert 'const LANDMARK_CONSTANTS = {"EAR_CLOSED":0.2};' in lm


# ---------------------------------------------------------- the matrix

def test_the_guidance_matrix_is_the_size_the_docstring_says():
    cases = bs.guidance_cases()
    assert [c["index"] for c in cases] == list(range(len(cases)))
    labels = [c["label"] for c in cases]
    pairs = [x for x in labels if x.count("+") == 1]
    assert len(pairs) == 91, "every pair of the fourteen rules"
    assert "armed(nothing)" in labels
    assert any(c["traits"] == {} for c in cases)
    absent = [c for c in cases if c["label"] == "yaw absent"][0]
    assert "yaw" not in absent["traits"]
    null = [c for c in cases if c["label"] == "parts=None"][0]
    assert null["traits"]["parts"] is None


def test_case_and_the_markers():
    assert bs.case("x", {}, "base")["frame"] == bs.BASE_FRAME
    assert bs.case("x", {}, [1, 2])["frame"] == [1, 2]
    assert repr(bs._ABSENT) == "<absent>" and repr(bs._NULL) == "<null>"


def test_too_close_composes_with_too_far():
    both = bs.armed(["too_far", "too_close"])
    assert both["traits"]["facePx"] == 100 and both["frame"] == [100, 100]
    chain = bs.python_chain(both["traits"], tuple(both["frame"]))
    fired = [row["name"] for row in chain if row["fired"]]
    assert fired == ["too_far", "too_close"]
    assert [row["name"] for row in chain if row["chosen"]] == ["too_far"]


def test_the_chain_is_empty_before_any_measurement_is_read():
    assert bs.python_chain({}, None) == []
    assert bs.python_chain({"error": "busy"}, None) == []


def test_crossover_finds_the_published_1646():
    assert bs.crossover(0.01) == 1646
    assert calibration.recommend_threshold(1645)[2]
    assert not calibration.recommend_threshold(1646)[2]
    assert bs.crossover(0.0) == 2, "nothing is safe enough at zero risk"
    assert bs.crossover(1.0) is None, "everything is safe enough at 100%"


def test_gallery_sizes_are_dense_around_every_crossover():
    sizes = bs.gallery_sizes()
    assert sizes == sorted(set(sizes)) and sizes[0] == 0
    for edge in (1646, 165):
        assert all(edge + d in sizes for d in range(-5, 6))
    assert 1000000 in sizes


def test_threshold_probes_hit_every_knot_and_both_ends():
    probes = bs.threshold_probes()
    assert probes == sorted(set(probes))
    assert all(t in probes for t, _ in calibration.SFACE_FMR)
    assert probes[0] == 0.0 and probes[-1] == 1.0
    spec = bs.calibration_cases()
    assert spec["sizes"] == spec["tableSizes"]


def test_the_python_side_answers_every_case(bundle):
    cases, spec, guide, calib, _ = bundle
    assert len(guide) == len(cases)
    assert len(guide[0]["checklist"]) == 11
    assert guide[0]["instruction"]["ready"] is True, "armed(nothing) passes"
    assert len(calib["fmr"]) == len(spec["thresholds"])
    assert len(calib["recommend"]) == len(spec["sizes"]) * len(spec["targets"])
    assert len(calib["table"]) == len(spec["sizes"])


# ----------------------------------------------------------- comparing

def test_differences_names_the_path_to_each_disagreement():
    want = {"a": [1, 2, {"b": 0.5}], "c": True, "gone": 1, "n": 1}
    got = {"a": [1, 2, {"b": 0.25}], "c": 1, "new": 2, "n": 1.0}
    assert bs.differences(want, got) == [
        ".a[2].b: python 0.5, browser 0.25",
        ".c: python True, browser 1",
        ".gone: only python has it (1)",
        ".new: only the browser has it (2)",
    ], "1 == 1.0 across int and float, but True is not 1"
    assert bs.differences([1, 2], [1]) == [
        ".: python has 2 entries, browser has 1"]
    assert bs.differences(3, 4) == [".: python 3, browser 4"]
    assert bs.differences({"k": [1]}, {"k": [1]}) == []


def test_report_is_silent_when_nothing_is_wrong_and_stops_otherwise():
    assert bs.report("x", []) is None
    with pytest.raises(SystemExit) as one:
        bs.report("guidance", ["first"])
    assert "in 1 place.\n" in str(one.value)
    many = ["line %d" % i for i in range(30)]
    with pytest.raises(SystemExit) as stop:
        bs.report("calibration", many, "Context.\n")
    text = str(stop.value)
    assert "in 30 places" in text and "Context." in text
    assert "line 24" in text and "line 25" not in text
    assert "... and 5 more" in text


# -------------------------------------------------------- the browser

def test_without_playwright_the_build_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(SystemExit) as stop:
        with bs.chromium():
            pass
    assert "pip install playwright" in str(stop.value)
    assert "no --skip-checks flag" in str(stop.value)


def test_the_system_chrome_is_used_when_it_is_there(browser, monkeypatch,
                                                     tmp_path):
    chrome = tmp_path / "chrome.exe"
    chrome.write_bytes(b"")
    monkeypatch.setattr(bs, "CHROME", str(chrome))
    with bs.chromium() as instance:
        assert instance is browser and not browser.closed
    assert browser.launched == [{"executable_path": str(chrome)}]
    assert browser.closed

    monkeypatch.setattr(bs, "CHROME", str(tmp_path / "absent.exe"))
    with bs.chromium():
        pass
    assert browser.launched[-1] == {}, "Playwright's own Chromium"


def test_a_bundle_that_throws_on_load_stops_the_build(browser, written):
    browser.throw_on_load = True
    with pytest.raises(SystemExit) as stop:
        with bs.loaded(browser, bs.harness(written)):
            pytest.fail("the page should not have been handed over")
    assert "does not load" in str(stop.value)
    assert "SyntaxError: Unexpected token" in str(stop.value)
    assert browser.pages[-1].closed


def test_the_harness_loads_exactly_the_checked_files(browser, written):
    with bs.loaded(browser, bs.harness(written)) as (page, problems):
        assert problems == [], "a console.log is not an error"
        for name in bs.CHECKED_JS:
            assert "<script>\n%s\n</script>" % written[name] in page.html
        assert written["js/demo.js"] not in page.html
    assert page.closed


def test_a_faithful_port_passes_every_check(browser, written, bundle, capsys):
    bs.run_checks(browser, written, bundle)
    out = capsys.readouterr().out
    assert "14 rules, compared position by position" in out
    assert "%d readings" % len(bundle[0]) in out
    assert "%d gallery sizes" % len(bundle[1]["sizes"]) in out

    bs.run_checks(browser, written, bundle, loud=False)
    assert capsys.readouterr().out == ""


def test_an_error_logged_while_checking_stops_the_build(browser, written,
                                                        bundle):
    browser.log_error = True
    with pytest.raises(SystemExit) as stop:
        bs.run_checks(browser, written, bundle, loud=False)
    assert "logged errors while being checked" in str(stop.value)
    assert "TypeError: x is null" in str(stop.value)


@pytest.mark.parametrize("name, said", [
    ("reorder-adjacent", "the priority order in js/guidance.js"),
    ("invert-strict", "guidance.py vs js/guidance.js"),
    ("loosen-recommend", "calibration.py vs js/calibration.js"),
    ("severity", "the browser answered 248 guidance cases and python "
                 "answered 249"),
])
def test_each_kind_of_breakage_is_refused(browser, written, bundle, name,
                                          said):
    entry = [e for e in bs.SABOTAGE if e[0] == name][0]
    with pytest.raises(SystemExit) as stop:
        bs.run_checks(browser, bs.sabotaged(written, entry), bundle,
                      loud=False)
    assert said in str(stop.value)


def test_a_sabotage_whose_anchor_moved_is_itself_a_failure(written):
    entry = ("missing", "js/guidance.js", [("no such text", "x")], "why")
    with pytest.raises(SystemExit) as stop:
        bs.sabotaged(written, entry)
    assert "expected exactly one match in js/guidance.js and found 0" in str(
        stop.value)
    appended = bs.sabotaged(written, ("tail", "js/constants.js",
                                      [(bs.APPEND, "\n// end\n")], "why"))
    assert appended["js/constants.js"] == written["js/constants.js"] + "\n// end\n"
    assert appended["js/guidance.js"] is written["js/guidance.js"]


def test_prove_requires_every_breakage_to_be_caught(browser, written, bundle,
                                                    capsys):
    bs.prove(browser, written, bundle)
    out = capsys.readouterr().out
    assert "all %d caught" % len(bs.SABOTAGE) in out
    assert out.count(" caught\n") == len(bs.SABOTAGE)
    # The line that explains each catch: the order's python/browser lines,
    # a case or a path for a value, nothing extra for a length mismatch.
    assert "python:  ['no_face'" in out
    assert "armed(nothing) .checklist: python has 11 entries" in out
    assert ".fmr[0]: python" in out


def test_prove_fails_on_a_breakage_the_checks_let_through(browser, written,
                                                          bundle):
    browser.survive = {"interpolate", "off-by-one"}
    with pytest.raises(SystemExit) as stop:
        bs.prove(browser, written, bundle)
    text = str(stop.value)
    assert "went undetected" in text
    assert "interpolate (the false-match interpolation" in text
    assert "off-by-one" in text and "severity" not in text


# ----------------------------------------------------------- the pages

def test_the_demo_page_is_marked_generated_and_loads_in_order(written):
    page = written["index.html"]
    assert "Generated by tools/build_static.py" in page
    assert '<a id="repoLink" href="%s">' % bs.REPO in page
    tags = [page.index('<script src="%s">' % s) for s in bs.SCRIPT_ORDER]
    assert tags == sorted(tags)


def test_the_camera_page_shares_the_checked_port(written):
    camera = bs.collect_camera(written)
    for name in bs.SHARED_WITH_CAMERA:
        assert camera[name] is written[name]
    assert bs.check_camera_shares_the_checked_port(written, camera) == 2
    assert "'%s'" % bs.VISION_MODULE in camera["js/camera.js"]
    assert "@MODEL_URL@" not in camera["js/camera.js"]
    assert '<script type="module" src="js/camera.js">' in camera["index.html"]

    drifted = dict(camera, **{"js/guidance.js": "// a different port"})
    with pytest.raises(SystemExit) as stop:
        bs.check_camera_shares_the_checked_port(written, drifted)
    assert "docs/camera/js/guidance.js is not the same file" in str(stop.value)


def test_the_camera_page_cannot_lose_its_four_sentences():
    page = " ".join(sentence for _, sentence in bs.CAMERA_CLAIMS)
    bs.check_camera_says_what_it_is(page)
    with pytest.raises(SystemExit) as stop:
        bs.check_camera_says_what_it_is(
            page.replace("Nothing is uploaded.", ""))
    assert "no longer states: that nothing is sent" in str(stop.value)


def test_the_camera_bundle_needs_the_shared_files(written):
    partial = dict(written)
    del partial["js/guidance.js"]
    with pytest.raises(SystemExit) as stop:
        bs.collect_camera(partial)
    assert "js/guidance.js is no longer part of the docs/app bundle" in str(
        stop.value)


def test_prune_removes_only_what_the_build_did_not_write(tmp_path, capsys):
    bs.prune(str(tmp_path / "never-built"), set())
    out = tmp_path / "app"
    (out / "js").mkdir(parents=True)
    for name in ("index.html", "js/keep.js", "js/old.js"):
        (out / name).write_text("x", encoding="utf-8")
    bs.prune(str(out), {"index.html", "js/keep.js"})
    assert sorted(p.name for p in out.rglob("*") if p.is_file()) == [
        "index.html", "keep.js"]
    assert "removed stale %s" % os.path.join("js", "old.js") in (
        capsys.readouterr().out)


def test_sizes_are_reported_raw_and_gzipped(capsys):
    rows = bs.sizes({"b.js": "x" * 2048, "a.css": ""})
    assert [r[0] for r in rows] == ["a.css", "b.js"]
    assert rows[1][1] == 2048 and rows[1][2] < 100
    bs.report_sizes("docs/app", rows)
    out = capsys.readouterr().out
    assert "wrote 2 files to docs/app" in out
    assert "2.0 KB" in out and out.splitlines()[-1].split()[0] == "total"


# ---------------------------------------------------------------- main

def test_an_unknown_option_is_refused():
    with pytest.raises(SystemExit) as stop:
        bs.main(["build_static.py", "--skip-checks"])
    assert "unknown option '--skip-checks'" in str(stop.value)


def published(folder):
    """Every file under docs/<folder>, as text with LF endings."""
    base = os.path.join(ROOT, "docs", folder)
    out = {}
    for here, _dirs, names in os.walk(base):
        for name in names:
            path = os.path.join(here, name)
            with io.open(path, encoding="utf-8", newline="") as handle:
                out[os.path.relpath(path, base).replace(os.sep, "/")] = (
                    handle.read().replace("\r\n", "\n"))
    return out


def built(folder):
    out = {}
    for path in folder.rglob("*"):
        if path.is_file():
            with io.open(str(path), encoding="utf-8", newline="") as handle:
                out[path.relative_to(folder).as_posix()] = handle.read()
    return out


def test_a_proven_build_writes_exactly_the_published_demo(browser, tmp_path,
                                                          monkeypatch,
                                                          capsys):
    """Both bundles go to tmp_path, never over docs/. What lands there is
    the demo GitHub Pages serves, byte for byte, and a file an older build
    left behind is removed."""
    app, camera = tmp_path / "app", tmp_path / "camera"
    monkeypatch.setattr(bs, "OUT", str(app))
    monkeypatch.setattr(bs, "CAMERA_OUT", str(camera))
    (app / "js").mkdir(parents=True)
    (app / "js" / "renamed.js").write_text("stale", encoding="utf-8")

    bs.main(["build_static.py", "--prove"])
    out = capsys.readouterr().out

    assert built(app) == published("app")
    assert built(camera) == published("camera")
    assert "the 1% recommendation stops being reachable at 1,646 people" in out
    assert "all %d caught" % len(bs.SABOTAGE) in out
    assert "removed stale %s" % os.path.join("js", "renamed.js") in out
    assert "2 files shared byte-for-byte with docs/app" in out
    assert "tasks-vision@%s" % bs.VISION_VERSION in out
    assert browser.closed


def test_a_refused_build_writes_nothing(browser, tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "OUT", str(tmp_path / "app"))
    monkeypatch.setattr(bs, "CAMERA_OUT", str(tmp_path / "camera"))
    browser.log_error = True
    with pytest.raises(SystemExit):
        bs.main(["build_static.py"])
    assert not (tmp_path / "app").exists()
    assert not (tmp_path / "camera").exists()


def test_a_build_without_prove_checks_once_and_writes(browser, tmp_path,
                                                      monkeypatch, capsys):
    monkeypatch.setattr(bs, "OUT", str(tmp_path / "app"))
    monkeypatch.setattr(bs, "CAMERA_OUT", str(tmp_path / "camera"))
    bs.main(["build_static.py"])
    out = capsys.readouterr().out
    assert "proving" not in out
    assert len(browser.pages) == 1, "one page: the checks, and no proofs"
    assert built(tmp_path / "app") == published("app")
