"""Build the browser demo of the two pure-arithmetic modules into docs/app/.

    python tools/build_static.py
    python tools/build_static.py --prove     (see "Proving it bites", below)

--------------------------------------------------------------------------
What this publishes, and what it refuses to pretend
--------------------------------------------------------------------------
This project recognises faces with a Haar cascade, an LBPH model held in a
328 MB `trainer.yml`, and a webcam the server owns. None of that runs in a
browser tab. A page that mimed it -- a canvas with a rectangle drawn on a
stock photograph, a name picked from a list and a confidence number invented
to sit beside it -- would be a worse thing to publish than nothing, because
the reader has no way to tell a real verdict from a plausible one, and the
one sentence admitting it is the sentence nobody reads.

So this does not ship the recogniser. There is no detector here, no model,
no camera, and no identification. What it ships is the two modules that
import neither cv2 nor numpy, because they are arithmetic over numbers
something else measured -- and they happen to be the two worth reading:

    pipeline/guidance.py    fourteen rules over one frame's measurements,
                            tried in a fixed order, of which the first to
                            fire is the only thing the person is told. The
                            order is the design, and a static reading of the
                            source is a poor way to see it.

    analysis/calibration.py the false-match table counted over FairFace's
                            4,767,224,190 impostor pairs, and the
                            gallery-size arithmetic that turns it into a
                            threshold -- which scales with how many people
                            are enrolled, and stops being sufficient at a
                            gallery size this page makes you look at.

  tools/static_src/js/guidance.js     guidance.py's rules, ported
  tools/static_src/js/calibration.js  the gallery-size arithmetic, ported
  tools/static_src/js/demo.js         the page: reads controls, draws
  tools/static_src/index.html         the markup, which the Flask app has no
                                      equivalent of -- its template is a
                                      camera console
  tools/static_src/css/demo.css       the layout and the chart palette

The *numbers* are not ported. Every threshold guidance.py compares against,
and every row of the measured false-match table, is exported from the Python
modules into `js/constants.js` by this script. A retyped MIN_FACE_PX is a
second place for it to live; a retyped measurement is a published figure
that has stopped being the measured one. Only the control flow is
transcribed, and the control flow is what gets checked.

They are inlined as a `.js` file rather than fetched as `.json`,
deliberately: a `fetch()` of a relative URL is blocked by the file:// origin
rules, so a JSON bundle would work on GitHub Pages and fail the moment
somebody double-clicked index.html. A `<script src>` has no such
restriction, so the same directory works both ways.

--------------------------------------------------------------------------
What this refuses to ship
--------------------------------------------------------------------------
A guidance engine that is *nearly* the Python one is the worst possible
output of this program. Every failure mode is quiet: a page that picked the
second-priority instruction instead of the first still shows a sensible
sentence, and the reader -- who came here to see the priority order -- would
be reading a different one and could not tell. A calibration curve off by
one threshold step still looks like a calibration curve.

So the port is not trusted. Before a byte is written, this script starts a
real browser, loads the JavaScript it is about to publish, and compares it
against the real Python over a matrix built to be awkward:

  * every rule armed on its own, so each one is known to fire at all;
  * **every pair of rules armed together**, all 91 of them, because a
    priority order is only observable when more than one thing is wrong.
    This is not a precaution, it is the load-bearing part of the matrix, and
    it was measured: with the chain comparison switched off so that only the
    *chosen instruction* is compared, swapping the two neighbouring rules
    `eyes_closed` and `one_eye_closed` changes the answer on five readings
    out of 249 -- one of them the pair case, four of them the flag
    combinations that happen to set both flags -- and on **none** of the
    fourteen single-fault readings. Moving the last rule to the front
    changes 27 readings, and all 27 of them are two-rule cases. A matrix of
    one-thing-wrong-at-a-time would have published either reordering;
  * every rule armed but one, and all fourteen at once;
  * each threshold's boundary from both sides and exactly on it --
    `facePx` at 109/110/111, yaw at +-21.9/22.0/22.1, the face fraction
    landing exactly on 0.85, and so on, because `<` and `<=` differ on
    precisely one input and nowhere else;
  * all sixteen combinations of the four `parts.flags`;
  * the missing ones: absent keys, explicit nulls, no `parts` at all, no
    frame size, an empty frame tuple, an empty trait dict (which is falsy
    in Python and truthy in JavaScript, so a literal port of `if not
    traits` is wrong), and a read carrying an error;
  * gallery sizes across the whole curve -- 0 through 20 one at a time,
    then decades out to a million, and every size within five of the point
    where the recommendation stops being reachable -- crossed with six risk
    targets including 0.0 and 1.0.

Compared for each case: the chosen instruction, its severity, its detail
text and its ready flag; the full fourteen-row priority chain with which
rule won and which were also firing; every one of the eleven checklist rows
with its pass/fail and its fix text; and on the calibration side the
interpolated false-match rate, the gallery risk, the recommended threshold
with its achieved risk and its reachable flag, the whole nineteen-row risk
table, and the rendered sentence, compared as a string.

Floats are compared bit for bit. There is no tolerance: both sides are IEEE
doubles, the constants cross as Python's shortest round-tripping repr, and
the arithmetic is the same operations in the same order. A tolerance here
would be a place for a real difference to hide.

Any disagreement stops the build. Nothing is written.

--------------------------------------------------------------------------
Proving it bites
--------------------------------------------------------------------------
A comparison that cannot fail is worse than no comparison, because it reads
like one that passed. `--prove` is the standing answer: it edits the
JavaScript that is about to be published, twelve ways, and requires the
checks to catch every one.

    reorder-adjacent   swap two neighbours in the priority chain
    reorder-distant    move the last rule to the front (a real move -- the
                       list stays fourteen long, so only the order is wrong)
    shift-threshold    move MAX_YAW from 22.0 to 22.5
    shift-measurement  move one measured false-match rate by 1e-9
    invert-strict      `facePx < MIN_FACE_PX` becomes `<=`
    invert-abs         `|yaw| > MAX_YAW` becomes `>=`
    invert-fraction    `fraction > 0.85` becomes `>=`
    invert-checklist   the checklist's yaw row alone becomes `<`
    drop-eyemismatch   the checklist row no rule reads always passes
    empty-dict         `if not traits` ported as `if (!traits)`
    severity           one rule's 'warn' becomes 'block'
    interpolate        the false-match interpolation runs from the wrong end
    off-by-one         gallery risk over N pairs instead of N-1
    loosen-recommend   `risk <= target` becomes `risk < target`

Each is a string edit against the *published* file, so it is the real bundle
being broken rather than a mock of it. If any survives, `--prove` fails and
names it: that is a hole in the matrix, and it is reported as a build
failure rather than as a passing run.

Three mutations were tried and are deliberately **not** in that list,
because they are provably equivalent rather than uncaught, and shipping
them would mean shipping a check that can never fail:

    fmr_at's `threshold <= xs[0]` -> `<`, its `>= xs[-1]` -> `>`, and
    bisect_left -> bisect_right.

All three differ from the original on one input each: exactly a knot of the
interpolation. At a knot the general branch computes `frac` of 1.0 (or 0.0)
and the linear interpolation returns exactly the endpoint the clamp would
have returned -- linear interpolation is continuous, so the two paths meet
there by construction. Verified over all nineteen measured thresholds plus
the eighteen midpoints between them: zero differences. They are not holes
in the matrix; there is nothing to catch.

The browser is the system Chrome if it is there and Playwright's Chromium
otherwise. That is a build-time dependency, not a runtime one: `pip install
-r requirements.txt` and `pytest` do not need it, and neither does the
published page. Running the JavaScript rather than transcribing it a third
time is the point -- a transcription of a port is just a third thing to get
wrong.
"""

import contextlib
import gzip
import io
import json
import os
import sys

# The project root -- one level up, because this file lives in tools/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "docs", "app")
SRC = os.path.join(ROOT, "tools", "static_src")

REPO = "https://github.com/luke-zhang-cs-py/Facial-Recognition-Software"

# Where Chrome lives on the machine this is developed on. Optional:
# Playwright ships its own Chromium and that is used when this is not here,
# so the build works on a runner too.
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# Load order. constants.js first because both ports read it as they define
# themselves; demo.js last because it calls into both.
SCRIPT_ORDER = ["js/constants.js", "js/guidance.js", "js/calibration.js",
                "js/demo.js"]

# The three files the comparison is about. demo.js is not among them: it
# decides nothing, and loading it outside a page would only fail on a
# missing document.
CHECKED_JS = ["js/constants.js", "js/guidance.js", "js/calibration.js"]


def read(path):
    with io.open(path, encoding="utf-8") as handle:
        return handle.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def stop(message):
    raise SystemExit("build_static: " + message)


def replace_once(text, old, new, what):
    """A substitution that fails loudly if the source moved.

    A miss is silent by default -- the page builds, looks nearly right, and
    is missing the one paragraph that says it is not the real thing. So each
    one is counted.
    """
    count = text.count(old)
    if count != 1:
        stop("could not %s -- expected exactly one match, found %d. The "
             "source has moved; fix the anchor in tools/build_static.py.\n"
             "  looking for: %r" % (what, count, old[:120]))
    return text.replace(old, new)


# ---------------------------------------------------------------------------
# The constants, exported rather than retyped
# ---------------------------------------------------------------------------
# Read off the modules by name. A name that disappears from guidance.py stops
# the build here rather than leaving the page comparing against a stale
# number that nothing in Python reads any more.
GUIDANCE_NAMES = ("MIN_FACE_PX", "IDEAL_FACE_PX", "MAX_FACE_FRACTION",
                  "MAX_YAW", "MAX_ROLL", "MIN_QUALITY", "MIN_SHARPNESS",
                  "MAX_SHADOW_CLIP", "MAX_HIGHLIGHT_CLIP",
                  "EYE_MISMATCH_LIMIT", "READY")


def collect_constants():
    from analysis import calibration
    from pipeline import guidance

    values = {}
    for name in GUIDANCE_NAMES:
        if not hasattr(guidance, name):
            stop("pipeline/guidance.py no longer defines %s, which "
                 "js/guidance.js reads. Port the change before rebuilding."
                 % name)
        values[name] = getattr(guidance, name)

    rules = [fn.__name__[len("_rule_"):] for fn in guidance.RULES]
    return {
        "guidance": values,
        "ruleOrder": rules,
        "calibration": {
            "fmr": [[t, f] for t, f in calibration.SFACE_FMR],
            "corpus": calibration.CORPUS,
            "reference": calibration.SFACE_REFERENCE,
            "disparity": calibration.FALSE_MATCH_DISPARITY,
        },
    }


def constants_file(constants):
    """The exported numbers as two `const`s, with their provenance written
    down. Nothing is rounded: every float is Python's shortest
    round-tripping repr, so JSON.parse recovers the same IEEE double the
    module holds -- which is what lets the checks compare exactly rather
    than approximately."""
    return (
        "/* Generated by tools/build_static.py from pipeline/guidance.py and\n"
        " * analysis/calibration.py -- do not edit. Rebuild with:\n"
        " *   python tools/build_static.py\n"
        " *\n"
        " * These are the measurements and the tuned thresholds, exported\n"
        " * rather than retyped. js/guidance.js and js/calibration.js port\n"
        " * the arithmetic and read every number from here, so a threshold\n"
        " * can only be wrong on this page if it is also wrong in Python.\n"
        " *\n"
        " * Inlined as a script rather than fetched as JSON so that opening\n"
        " * index.html from the filesystem works as well as serving it: a\n"
        " * relative fetch() is blocked by the file:// origin rules and a\n"
        " * <script src> is not.\n"
        " *\n"
        " *   fmr        [threshold, false-match rate per impostor pair],\n"
        " *              counted over %s\n"
        " */\n"
        "const GUIDANCE_CONSTANTS = %s;\n"
        "const CALIBRATION_DATA = %s;\n"
        % (constants["calibration"]["corpus"],
           json.dumps(constants["guidance"], separators=(",", ":"),
                      ensure_ascii=True),
           json.dumps(constants["calibration"], separators=(",", ":"),
                      ensure_ascii=True)))


# ---------------------------------------------------------------------------
# The guidance matrix
# ---------------------------------------------------------------------------
# A frame read with nothing wrong with it. Every case below is this, with
# something broken in it on purpose.
def base_traits():
    return {"detected": True, "faces": 1, "facePx": 190, "yaw": 4.0,
            "roll": 2.0, "sharpness": 90.0, "qualityScore": 0.62,
            "shadowClip": 0.05, "highlightClip": 0.04,
            "parts": {"flags": [], "eyeMismatch": 0.05}}


BASE_FRAME = [480, 640]


# One "arm" per rule: the smallest change to a passing read that makes that
# one rule fire. Applied in priority order so a pair of arms composes --
# too_close sizes the frame to whatever facePx is by then, which is what lets
# it fire alongside too_far (a small face in a smaller frame) as well as on
# its own (a big face in a normal frame).
def _arm_no_face(t, f):
    t["detected"] = False
    return t, f


def _arm_crowd(t, f):
    t["faces"] = 2
    return t, f


def _arm_too_far(t, f):
    t["facePx"] = 100
    return t, f


def _arm_too_close(t, f):
    px = t["facePx"]
    return t, [px, px]


def _arm_yaw(t, f):
    t["yaw"] = 35.0
    return t, f


def _arm_roll(t, f):
    t["roll"] = 25.0
    return t, f


def _flag(name):
    def arm(t, f):
        t["parts"]["flags"] = t["parts"]["flags"] + [name]
        return t, f
    return arm


def _set(key, value):
    def arm(t, f):
        t[key] = value
        return t, f
    return arm


ARMS = [
    ("no_face", _arm_no_face),
    ("crowd", _arm_crowd),
    ("too_far", _arm_too_far),
    ("too_close", _arm_too_close),
    ("yaw", _arm_yaw),
    ("roll", _arm_roll),
    ("eyes_closed", _flag("eyes closed")),
    ("one_eye_closed", _flag("one eye closed")),
    ("obscured", _flag("face partly obscured")),
    ("mouth_open", _flag("mouth open")),
    ("shadow", _set("shadowClip", 0.80)),
    ("highlight", _set("highlightClip", 0.70)),
    ("blur", _set("sharpness", 8.0)),
    ("quality", _set("qualityScore", 0.05)),
]


def armed(names, mode="idle"):
    """A case with exactly these rules made to fire."""
    traits, frame = base_traits(), list(BASE_FRAME)
    for name, arm in ARMS:
        if name in names:
            traits, frame = arm(traits, frame)
    return {"label": "armed(%s)" % ("+".join(names) if names else "nothing"),
            "traits": traits, "frame": frame, "mode": mode}


def case(label, traits, frame=None, mode="idle"):
    return {"label": label, "traits": traits,
            "frame": BASE_FRAME if frame == "base" else frame, "mode": mode}


class _Marker(object):
    """`None` is a value these cases have to be able to *send*, so "leave
    this key out entirely" and "send an explicit null" need to be two
    different things that are both distinct from it."""

    def __init__(self, what):
        self.what = what

    def __repr__(self):
        return self.what


_ABSENT = _Marker("<absent>")
_NULL = _Marker("<null>")


def tweak(label, **changes):
    """The passing read with individual fields overridden."""
    traits = base_traits()
    frame = changes.pop("frame", BASE_FRAME)
    mode = changes.pop("mode", "idle")
    for key, value in changes.items():
        if value is _ABSENT:
            traits.pop(key, None)
        elif value is _NULL:
            traits[key] = None
        else:
            traits[key] = value
    return {"label": label, "traits": traits, "frame": frame, "mode": mode}


def guidance_cases():
    """The matrix. Built, not written out -- 91 pairs is not a list anybody
    maintains by hand, and the ones that matter are the ones nobody would
    think to type."""
    cases = [armed([], "idle"), armed([], "register")]

    names = [name for name, _ in ARMS]

    # Each rule on its own: does it fire at all?
    for name in names:
        cases.append(armed([name]))
        cases.append(armed([name], "register"))

    # Every pair. This is the part that makes the priority order observable:
    # with one fault there is nothing to order.
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            cases.append(armed([first, second]))

    # Everything at once, and everything-but-one -- a rule that can never
    # win is a rule the page would never show.
    cases.append(armed(names))
    for name in names:
        cases.append(armed([n for n in names if n != name]))

    # ---- boundaries. `<` and `<=` differ on exactly one input.
    for px in (0, 1, 109, 110, 111, 200):
        cases.append(tweak("facePx=%d" % px, facePx=px))
    cases.append(tweak("facePx=None", facePx=None))
    cases.append(tweak("facePx absent", facePx=_ABSENT))

    for yaw in (0.0, 21.9, 22.0, 22.1, -21.9, -22.0, -22.1, 90.0, -90.0):
        cases.append(tweak("yaw=%s" % yaw, yaw=yaw))
    cases.append(tweak("yaw=None", yaw=None))
    cases.append(tweak("yaw absent", yaw=_ABSENT))

    for roll in (0.0, 14.9, 15.0, 15.1, -14.9, -15.0, -15.1):
        cases.append(tweak("roll=%s" % roll, roll=roll))
    cases.append(tweak("roll=None", roll=None))
    cases.append(tweak("roll absent", roll=_ABSENT))

    for sharp in (0.0, 24.9, 25.0, 25.1, 300.0):
        cases.append(tweak("sharpness=%s" % sharp, sharpness=sharp))
    cases.append(tweak("sharpness=None", sharpness=None))
    cases.append(tweak("sharpness absent", sharpness=_ABSENT))

    for q in (0.0, 0.219, 0.22, 0.221, 1.0):
        cases.append(tweak("quality=%s" % q, qualityScore=q))
    cases.append(tweak("quality=None", qualityScore=None))
    cases.append(tweak("quality absent", qualityScore=_ABSENT))

    for clip in (0.0, 0.449, 0.45, 0.451, 1.0):
        cases.append(tweak("shadowClip=%s" % clip, shadowClip=clip))
    cases.append(tweak("shadowClip=None", shadowClip=None))
    cases.append(tweak("shadowClip absent", shadowClip=_ABSENT))

    for clip in (0.0, 0.299, 0.30, 0.301, 1.0):
        cases.append(tweak("highlightClip=%s" % clip, highlightClip=clip))
    cases.append(tweak("highlightClip=None", highlightClip=None))
    cases.append(tweak("highlightClip absent", highlightClip=_ABSENT))

    # eyeMismatch is read by the checklist and by no rule, so it is only
    # observable if the checklist is compared. That is the point of it.
    for mismatch in (0.0, 0.279, 0.28, 0.281, 1.0, None):
        cases.append(tweak("eyeMismatch=%s" % mismatch,
                           parts={"flags": [], "eyeMismatch": mismatch}))
    cases.append(tweak("eyeMismatch absent", parts={"flags": []}))
    cases.append(tweak("parts=None", parts=_NULL))
    cases.append(tweak("parts absent", parts=_ABSENT))

    # The face fraction, which is the one threshold that is not a field but
    # a ratio: px*px/(h*w) against 0.85. Straddled from both sides and hit
    # exactly -- 170*170 / (100*340) is 0.85 on the nose, and the rule is a
    # strict `>`, so exactly-on-it must not fire.
    for label, px, frame in (("fraction just under", 100, [100, 118]),
                             ("fraction just over", 100, [100, 117]),
                             ("fraction exactly 0.85", 170, [100, 340]),
                             ("fraction 1.0", 200, [200, 200]),
                             ("fraction tiny", 100, [1080, 1920])):
        cases.append(tweak(label, facePx=px, frame=frame))
    cases.append(tweak("no frame", frame=None))
    cases.append(tweak("empty frame", frame=[]))
    cases.append(tweak("zero frame", frame=[0, 0]))
    cases.append(tweak("big face, no frame", facePx=600, frame=None))

    for faces in (0, 1, 2, 3, 5):
        cases.append(tweak("faces=%d" % faces, faces=faces))
    cases.append(tweak("faces absent", faces=_ABSENT))
    for detected in (True, False, None):
        for faces in (0, 1, 2):
            cases.append(tweak("detected=%s faces=%d" % (detected, faces),
                               detected=detected, faces=faces))
    cases.append(tweak("detected absent", detected=_ABSENT))

    # All sixteen flag combinations.
    flags = ["eyes closed", "one eye closed", "mouth open",
             "face partly obscured"]
    for mask in range(16):
        chosen = [flags[i] for i in range(4) if mask & (1 << i)]
        cases.append(tweak("flags=%s" % (chosen or "none"),
                           parts={"flags": chosen, "eyeMismatch": 0.05}))

    # Read before any measurement is looked at.
    cases.append(case("no traits", {}))
    cases.append(case("error", {"error": "device busy"}))
    cases.append(case("error, and everything else wrong",
                      dict(base_traits(), error="camera not found",
                           detected=False, faces=4)))
    cases.append(case("error is the empty string",
                      dict(base_traits(), error="")))
    cases.append(case("only detected", {"detected": True}))
    cases.append(case("only faces", {"faces": 1}))
    cases.append(case("detected with nothing else", {"detected": False}))

    for index, entry in enumerate(cases):
        entry["index"] = index
    return cases


# ---------------------------------------------------------------------------
# The calibration matrix
# ---------------------------------------------------------------------------
RISK_TARGETS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]


def gallery_sizes():
    """Across the whole curve, and dense where it turns.

    The interesting sizes are not evenly spaced: everything below about
    twenty moves a step at a time, and the point where the recommendation
    stops being reachable is a single integer, so the sizes within five of
    it are enumerated rather than sampled.
    """
    sizes = list(range(0, 21))
    sizes += [25, 30, 40, 50, 75, 100, 150, 200, 250, 350, 500, 750]
    sizes += [1000, 1250, 1500, 2000, 3000, 5000, 10000, 50000, 100000,
              1000000]
    for risk in RISK_TARGETS:
        edge = crossover(risk)
        if edge is not None:
            sizes += [edge + d for d in range(-5, 6) if edge + d >= 0]
    return sorted(set(sizes))


def crossover(max_risk):
    """The first gallery size at which no measured threshold is enough.

    Found by asking the module, not by writing the answer down -- it moves
    with the target, and at the 1% default it is 1,646, which is not the
    round number anybody would have guessed.
    """
    from analysis import calibration

    if not calibration.recommend_threshold(2, max_risk)[2]:
        return 2
    lo, hi = 2, 4
    while calibration.recommend_threshold(hi, max_risk)[2]:
        lo, hi = hi, hi * 2
        if hi > 10 ** 9:
            return None
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if calibration.recommend_threshold(mid, max_risk)[2]:
            lo = mid
        else:
            hi = mid
    return hi


def threshold_probes():
    """Where fmr_at is asked. Every measured knot exactly -- which is where
    a bisect that leaned the wrong way would show, if it could -- plus the
    midpoints between them, which is where the interpolation itself is the
    only thing being tested, plus outside both ends."""
    from analysis import calibration

    knots = [t for t, _ in calibration.SFACE_FMR]
    probes = list(knots)
    probes += [(knots[i] + knots[i + 1]) / 2 for i in range(len(knots) - 1)]
    probes += [knots[i] + 0.001 for i in range(len(knots) - 1)]
    probes += [0.0, 0.1, 0.25, 0.29, 0.2999, 0.7501, 0.8, 0.95, 1.0]
    return sorted(set(probes))


def calibration_cases():
    sizes = gallery_sizes()
    return {
        "thresholds": threshold_probes(),
        "sizes": sizes,
        "targets": RISK_TARGETS,
        # Kept apart from the cross product: the table is nineteen rows of
        # three floats per size, and it is compared for every size.
        "tableSizes": sizes,
    }


# ---------------------------------------------------------------------------
# The Python side, measured once
# ---------------------------------------------------------------------------
def python_chain(traits, frame):
    """Which rules fire, in priority order -- the same walk `instruction`
    makes, with nothing discarded. Built from `guidance.RULES` itself, so
    the order compared against the browser is the order the module declares
    rather than one written down twice."""
    from pipeline import guidance

    if not traits or traits.get("error"):
        return []
    reading = guidance.Reading(traits, frame)
    rows = []
    chosen = False
    for rule in guidance.RULES:
        verdict = rule(reading)
        rows.append({
            "name": rule.__name__[len("_rule_"):],
            "fired": verdict is not None,
            "chosen": verdict is not None and not chosen,
            "severity": None if verdict is None else verdict["severity"],
            "message": None if verdict is None else verdict["message"],
        })
        if verdict is not None:
            chosen = True
    return rows


def python_guidance(cases):
    from pipeline import guidance

    out = []
    for entry in cases:
        frame = tuple(entry["frame"]) if entry["frame"] is not None else None
        out.append({
            "instruction": guidance.instruction(entry["traits"], frame,
                                                entry["mode"]),
            "chain": python_chain(entry["traits"], frame),
            "checklist": guidance.checklist(entry["traits"], frame),
        })
    return out


def python_calibration(spec):
    from analysis import calibration

    fmr = [calibration.fmr_at(t) for t in spec["thresholds"]]
    risk = [[calibration.gallery_risk(t, n) for t in spec["thresholds"]]
            for n in spec["sizes"]]
    recommend = []
    described = []
    for size in spec["sizes"]:
        for target in spec["targets"]:
            threshold, achieved, ok = calibration.recommend_threshold(
                size, target)
            recommend.append({"threshold": threshold, "risk": achieved,
                              "reachable": ok})
            described.append(calibration.describe(size, target))
    table = [calibration.risk_table(n) for n in spec["tableSizes"]]
    return {"fmr": fmr, "risk": risk, "recommend": recommend,
            "describe": described, "table": table}


# ---------------------------------------------------------------------------
# The browser
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def chromium():
    """One browser for the whole build, checks and proofs alike."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        stop("this build runs the JavaScript it is about to publish against "
             "the Python it was ported from, and that needs a browser.\n"
             "  pip install playwright && python -m playwright install "
             "chromium\n"
             "There is no --skip-checks flag on purpose: an unchecked port "
             "of these two modules is the one thing this script exists to "
             "prevent.")

    with sync_playwright() as pw:
        options = {}
        if os.path.exists(CHROME):
            options["executable_path"] = CHROME
        instance = pw.chromium.launch(**options)
        try:
            yield instance
        finally:
            instance.close()


@contextlib.contextmanager
def loaded(instance, sources):
    """A page holding the bundle's JavaScript, and no tolerance for errors.

    about:blank, so there is no origin and no server -- the same situation a
    file:// double-click puts the page in.
    """
    html = ('<!doctype html><html><head><meta charset="utf-8">'
            "<title>build check</title></head><body>"
            + "".join("<script>\n%s\n</script>" % text for text in sources)
            + "</body></html>")
    page = instance.new_page()
    problems = []
    page.on("pageerror", lambda bad: problems.append(str(bad)))
    page.on("console",
            lambda msg: problems.append(msg.text) if msg.type == "error"
            else None)
    try:
        page.set_content(html)
        if problems:
            stop("the bundle's JavaScript does not load:\n  "
                 + "\n  ".join(problems))
        yield page, problems
    finally:
        page.close()


GUIDANCE_JS = """(cases) => cases.map((c) => ({
  instruction: Guidance.instruction(c.traits, c.frame, c.mode),
  chain: Guidance.chain(c.traits, c.frame),
  checklist: Guidance.checklist(c.traits, c.frame)
}))"""

CALIBRATION_JS = """(spec) => {
  const fmr = spec.thresholds.map((t) => Calibration.fmrAt(t));
  const risk = spec.sizes.map(
    (n) => spec.thresholds.map((t) => Calibration.galleryRisk(t, n)));
  const recommend = [];
  const describe = [];
  for (const n of spec.sizes) {
    for (const target of spec.targets) {
      const found = Calibration.recommendThreshold(n, target);
      recommend.push({ threshold: found.threshold, risk: found.risk,
                       reachable: found.reachable });
      describe.push(Calibration.describe(n, target));
    }
  }
  const table = spec.tableSizes.map((n) => Calibration.riskTable(n));
  return { fmr, risk, recommend, describe, table };
}"""

ORDER_JS = "() => Guidance.RULES.map((r) => r.name)"


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------
def _same_leaf(want, got):
    """Whether two values at the bottom of a payload agree.

    Numbers compare across int and float, because JSON has one number type
    and the two sides hand back whichever fits. A bool never compares equal
    to a number, though: `True == 1` in Python, and a `reachable` flag
    coming back as `1` is precisely the kind of thing this is for.

    Floats are compared exactly. Both sides are IEEE doubles, the constants
    cross as shortest round-tripping reprs, and the arithmetic is the same
    operations in the same order -- so a tolerance would only ever be
    somewhere for a real difference to hide.
    """
    if isinstance(want, bool) != isinstance(got, bool):
        return False
    return want == got


def differences(want, got, path="", found=None):
    """Every place two payloads disagree, with the path to each."""
    if found is None:
        found = []
    if isinstance(want, dict) and isinstance(got, dict):
        for key in sorted(set(want) | set(got)):
            where = "%s.%s" % (path, key)
            if key not in want:
                found.append("%s: only the browser has it (%r)"
                             % (where, got[key]))
            elif key not in got:
                found.append("%s: only python has it (%r)"
                             % (where, want[key]))
            else:
                differences(want[key], got[key], where, found)
    elif isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            found.append("%s: python has %d entries, browser has %d"
                         % (path or ".", len(want), len(got)))
        else:
            for index, (mine, yours) in enumerate(zip(want, got)):
                differences(mine, yours, "%s[%d]" % (path, index), found)
    elif not _same_leaf(want, got):
        found.append("%s: python %r, browser %r" % (path or ".", want, got))
    return found


def report(what, wrong, extra=""):
    if not wrong:
        return
    shown = wrong[:25]
    stop("%s: the browser build disagrees with the Python in %d place%s.\n"
         "Nothing has been written to docs/app.\n%s  %s%s"
         % (what, len(wrong), "" if len(wrong) == 1 else "s", extra,
            "\n  ".join(shown),
            "\n  ... and %d more" % (len(wrong) - len(shown))
            if len(wrong) > len(shown) else ""))


# ---------------------------------------------------------------------------
# The passes
# ---------------------------------------------------------------------------
def check_rule_order(page, constants):
    """The priority order, position by position.

    Compared on its own as well as through the cases, because it is the one
    thing this page exists to show. A reordering that happened to change no
    verdict in the matrix would still be a different design, and the page
    draws the list.
    """
    theirs = page.evaluate(ORDER_JS)
    mine = constants["ruleOrder"]
    if theirs != mine:
        stop("the priority order in js/guidance.js is not the one "
             "pipeline/guidance.py declares. Nothing has been written.\n"
             "  python:  %s\n  browser: %s" % (mine, theirs))
    return len(mine)


def check_guidance(page, cases, reference):
    theirs = page.evaluate(GUIDANCE_JS, cases)
    if len(theirs) != len(reference):
        stop("the browser answered %d guidance cases and python answered %d"
             % (len(theirs), len(reference)))
    wrong = []
    for entry, mine, yours in zip(cases, reference, theirs):
        wrong += ["%s %s" % (entry["label"], line)
                  for line in differences(mine, yours)]
    report("guidance.py vs js/guidance.js", wrong,
           "The instruction, its severity and detail, the whole priority "
           "chain and every checklist row are compared.\n")
    return len(cases)


def check_calibration(page, spec, reference):
    theirs = page.evaluate(CALIBRATION_JS, spec)
    wrong = differences(reference, theirs)
    report("calibration.py vs js/calibration.js", wrong,
           "Interpolated rates, gallery risk, the recommendation with its "
           "reachable flag, the whole table and the rendered sentence.\n")
    return (len(spec["thresholds"]), len(spec["sizes"]),
            len(spec["sizes"]) * len(spec["targets"]))


def harness(written):
    return [written[name] for name in CHECKED_JS]


def run_checks(instance, written, bundle, loud=True):
    cases, spec, guidance_ref, calib_ref, constants = bundle
    with loaded(instance, harness(written)) as (page, problems):
        rules = check_rule_order(page, constants)
        if loud:
            print("  RULES      == RULES                 %d rules, compared "
                  "position by position" % rules)
        checked = check_guidance(page, cases, guidance_ref)
        if loud:
            print("  instruction== instruction          %d readings: the "
                  "chosen instruction," % checked)
            print("                                     its severity, detail "
                  "and ready flag, the 14-row")
            print("                                     priority chain and "
                  "all 11 checklist rows")
        probes, sizes, pairs = check_calibration(page, spec, calib_ref)
        if loud:
            print("  calibration== calibration          %d thresholds, %d "
                  "gallery sizes, %d" % (probes, sizes, pairs))
            print("                                     (size, target) pairs "
                  "-- rate, risk, recommendation,")
            print("                                     19-row table and the "
                  "rendered sentence")
        if problems:
            stop("the bundle's JavaScript logged errors while being "
                 "checked:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# Proving the checks bite
# ---------------------------------------------------------------------------
# Each entry is (name, file, [(find, replace), ...], what it does). A miss is
# a failure in itself: an anchor that no longer matches would mean a sabotage
# that was never injected and therefore trivially "caught".
#
# `APPEND` as the find-string means "add this to the end of the file", which
# is how the two data breakages work -- the constants arrive as one JSON blob
# and there is no line inside it to edit.
APPEND = None

SABOTAGE = [
    ("reorder-adjacent", "js/guidance.js", [(
        "    { name: 'eyes_closed', fn: ruleEyesClosed },\n"
        "    { name: 'one_eye_closed', fn: ruleOneEyeClosed },",
        "    { name: 'one_eye_closed', fn: ruleOneEyeClosed },\n"
        "    { name: 'eyes_closed', fn: ruleEyesClosed },")],
     "swap two neighbours in the priority chain"),

    # A genuine move, not a duplicate: the rule is taken out of the tail and
    # put at the head, so the list is still fourteen long and the only thing
    # wrong with it is the order. A duplicated entry would be caught on
    # length alone, which would prove much less.
    ("reorder-distant", "js/guidance.js", [
        ("    { name: 'quality', fn: ruleQuality }\n  ];",
         "  ];"),
        ("  var RULES = [\n    { name: 'no_face', fn: ruleNoFace },",
         "  var RULES = [\n    { name: 'quality', fn: ruleQuality },\n"
         "    { name: 'no_face', fn: ruleNoFace },"),
        ("    { name: 'blur', fn: ruleBlur },\n",
         "    { name: 'blur', fn: ruleBlur }\n")],
     "move the lighting rule from last to first"),

    ("shift-threshold", "js/constants.js", [(
        APPEND, "\n/* --prove */ GUIDANCE_CONSTANTS.MAX_YAW = 22.5;\n")],
     "move MAX_YAW from 22.0 to 22.5"),

    ("shift-measurement", "js/constants.js", [(
        APPEND, "\n/* --prove */ CALIBRATION_DATA.fmr[8][1] += 1e-9;\n")],
     "move one measured false-match rate by 1e-9"),

    ("invert-strict", "js/guidance.js", [(
        "if (r.facePx && r.facePx < K.MIN_FACE_PX) {",
        "if (r.facePx && r.facePx <= K.MIN_FACE_PX) {")],
     "`facePx < MIN_FACE_PX` becomes `<=`"),

    ("invert-abs", "js/guidance.js", [(
        "if (r.yaw !== null && Math.abs(r.yaw) > K.MAX_YAW) {",
        "if (r.yaw !== null && Math.abs(r.yaw) >= K.MAX_YAW) {")],
     "`|yaw| > MAX_YAW` becomes `>=`"),

    ("invert-fraction", "js/guidance.js", [(
        "if (r.frameArea && r.faceFraction() > K.MAX_FACE_FRACTION) {",
        "if (r.frameArea && r.faceFraction() >= K.MAX_FACE_FRACTION) {")],
     "`fraction > 0.85` becomes `>=`"),

    ("invert-checklist", "js/guidance.js", [(
        "['Facing forward', yaw === null || Math.abs(yaw) <= K.MAX_YAW,",
        "['Facing forward', yaw === null || Math.abs(yaw) < K.MAX_YAW,")],
     "the checklist's yaw row alone becomes `<`"),

    ("drop-eyemismatch", "js/guidance.js", [(
        "       eyeMismatch === null || eyeMismatch <= K.EYE_MISMATCH_LIMIT,",
        "       true,")],
     "the checklist row that no rule reads always passes"),

    ("empty-dict", "js/guidance.js", [(
        "    if (traits === null || traits === undefined) { return true; }",
        "    return !traits;")],
     "`if not traits` ported as JavaScript's `if (!traits)`"),

    ("severity", "js/guidance.js", [(
        "      return out('warn', 'Head upright', 'Your head is tilted.');",
        "      return out('block', 'Head upright', 'Your head is tilted.');")],
     "one rule's 'warn' becomes 'block'"),

    ("interpolate", "js/calibration.js", [(
        "    return y0 + frac * (y1 - y0);",
        "    return y1 + frac * (y1 - y0);")],
     "the false-match interpolation runs from the wrong end"),

    ("off-by-one", "js/calibration.js", [(
        "    return 1.0 - Math.pow(1.0 - fmrAt(threshold), gallerySize - 1);",
        "    return 1.0 - Math.pow(1.0 - fmrAt(threshold), gallerySize);")],
     "gallery risk over N pairs instead of N-1"),

    ("loosen-recommend", "js/calibration.js", [(
        "      if (risk <= maxRisk) {",
        "      if (risk < maxRisk) {")],
     "`risk <= target` becomes `risk < target`"),
]


def sabotaged(written, entry):
    name, target, edits, _why = entry
    text = written[target]
    for old, new in edits:
        if old is APPEND:
            text += new
            continue
        count = text.count(old)
        if count != 1:
            stop("the --prove edit %r expected exactly one match in %s and "
                 "found %d. The source moved; fix the anchor in "
                 "tools/build_static.py -- a sabotage that is never injected "
                 "would be reported as caught.\n  looking for: %r"
                 % (name, target, count, old[:110]))
        text = text.replace(old, new)
    return dict(written, **{target: text})


def prove(instance, written, bundle):
    """Break the published bundle every way in SABOTAGE, and require the
    checks to catch each one."""
    print("proving the comparison is not vacuous...")
    survived = []
    for entry in SABOTAGE:
        name, _target, _edits, why = entry
        broken = sabotaged(written, entry)
        try:
            run_checks(instance, broken, bundle, loud=False)
        except SystemExit as refusal:
            lines = [line.strip() for line in str(refusal).splitlines()
                     if line.strip()]
            print("  %-18s %-48s caught" % (name, why))
            print("      %s" % lines[0][:108])
            for line in lines[1:]:
                if line.startswith(("guidance", "calibration", "armed",
                                    "the priority", "python:", "browser:"))\
                        or ": python " in line:
                    print("      %s" % line[:108])
                    break
            continue
        survived.append("%s (%s)" % (name, why))

    if survived:
        stop("these deliberate breakages went undetected, so the checks "
             "above are not covering what they claim to:\n  "
             + "\n  ".join(survived))
    print("  all %d caught -- the checks fail when the port is wrong\n"
          % len(SABOTAGE))


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def page_html(template):
    text = replace_once(
        template,
        '<link rel="stylesheet" href="css/demo.css">',
        '<link rel="stylesheet" href="css/demo.css">\n'
        '<!-- Generated by tools/build_static.py from tools/static_src/ --\n'
        '     do not edit this copy; edit the source and rebuild. -->',
        "mark the page as generated")

    text = replace_once(
        text, '<a id="repoLink" href="#">',
        '<a id="repoLink" href="%s">' % REPO,
        "point the banner link at the repository")

    scripts = "\n".join('<script src="%s"></script>' % name
                        for name in SCRIPT_ORDER)
    text = replace_once(
        text, "</body>",
        "<!-- Order matters. constants.js carries the exported numbers and\n"
        "     is read by the two ports as they define themselves; demo.js\n"
        "     calls into both and goes last. -->\n"
        + scripts + "\n\n</body>",
        "add the script tags")
    return text


def collect(constants):
    """Everything the bundle is made of, as {path in docs/app: text}."""
    written = {"js/constants.js": constants_file(constants)}
    for name in ("guidance.js", "calibration.js", "demo.js"):
        written["js/" + name] = read(os.path.join(SRC, "js", name))
    written["css/demo.css"] = read(os.path.join(SRC, "css", "demo.css"))
    written["index.html"] = page_html(read(os.path.join(SRC, "index.html")))
    return written


def prune(kept):
    """Delete anything left in docs/app/ from an older build.

    Without this a renamed file lives on and is still loaded, which is the
    one failure mode a generated directory has that a hand-written one does
    not.
    """
    if not os.path.isdir(OUT):
        return
    for here, _dirs, names in os.walk(OUT):
        for name in names:
            path = os.path.join(here, name)
            if os.path.relpath(path, OUT).replace("\\", "/") not in kept:
                os.remove(path)
                print("  removed stale %s" % os.path.relpath(path, OUT))


def sizes(written):
    rows = []
    for name, body in sorted(written.items()):
        raw = body.encode("utf-8")
        rows.append((name, len(raw), len(gzip.compress(raw, 9))))
    return rows


def main(argv):
    proving = "--prove" in argv[1:]
    for flag in argv[1:]:
        if flag != "--prove":
            stop("unknown option %r. The only one is --prove." % flag)

    constants = collect_constants()
    cases = guidance_cases()
    spec = calibration_cases()
    edge = crossover(0.01)

    print("exporting %d guidance constants and %d measured false-match rates"
          % (len(constants["guidance"]), len(constants["calibration"]["fmr"])))
    print("  the 1%% recommendation stops being reachable at %s people"
          % format(edge, ","))

    written = collect(constants)

    print("checking the port against the Python it came from, over %d "
          "readings and %d gallery sizes..."
          % (len(cases), len(spec["sizes"])))
    bundle = (cases, spec, python_guidance(cases), python_calibration(spec),
              constants)

    with chromium() as instance:
        run_checks(instance, written, bundle)
        if proving:
            print()
            prove(instance, written, bundle)

    for name, body in sorted(written.items()):
        write(os.path.join(OUT, name.replace("/", os.sep)), body)
    prune(set(written))

    rows = sizes(written)
    print("wrote %d files to docs/app" % len(rows))
    for name, raw, packed in rows:
        print("  %-22s %7.1f KB  %6.1f KB gzipped"
              % (name, raw / 1024.0, packed / 1024.0))
    print("  %-22s %7.1f KB  %6.1f KB gzipped"
          % ("total", sum(r[1] for r in rows) / 1024.0,
             sum(r[2] for r in rows) / 1024.0))
    print("  serve it:  python -m http.server -d docs 8000  ->  "
          "http://127.0.0.1:8000/app/")


if __name__ == "__main__":
    main(sys.argv)
