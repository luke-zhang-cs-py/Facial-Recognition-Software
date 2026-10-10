"""Rules stated as properties, with hypothesis hunting for the counterexample.

The example tests elsewhere check the cases somebody thought of. These state
what must hold for *every* input and let hypothesis generate the inputs and
shrink any failure to the smallest one:

* the calibrated threshold never loosens as more people enroll;
* guidance gives exactly one instruction, and it is the first rule to fire;
* json_safe never lets NaN or Infinity through, however deeply it is nested;
* the liveness vote never passes a new person on the previous person's frames.

No faces, no weights, no network: every input is generated here.
"""
import json
import math
import os

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings        # noqa: E402
from hypothesis import strategies as st                    # noqa: E402

from analysis import calibration                           # noqa: E402
from pipeline import decision, guidance, liveness          # noqa: E402

# Windows runners and coverage both slow the first example down; a deadline
# would fail on the machine, not on the property.
settings.register_profile("project", deadline=None,
                          suppress_health_check=[HealthCheck.too_slow])
# More examples on request: HYPOTHESIS_PROFILE=thorough pytest tests/test_properties.py
settings.register_profile("thorough", parent=settings.get_profile("project"),
                          max_examples=3000)
# The same examples every run, for mutmut: a mutant is killed or survives on
# the tests, not on which inputs this run happened to draw.
settings.register_profile("mutmut", parent=settings.get_profile("project"),
                          derandomize=True, database=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "project"))

# float32 of a huge float is infinity, which is one of the inputs wanted here;
# numpy warns about the cast every time.
pytestmark = pytest.mark.filterwarnings(
    "ignore:overflow encountered in cast:RuntimeWarning")


# ---------------------------------------------------------- calibration

risks = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
sizes = st.integers(min_value=0, max_value=200_000)


@given(sizes, sizes, risks)
def test_the_threshold_never_loosens_as_the_gallery_grows(a, b, max_risk):
    """More people means more ways to collide, so the threshold can only stay
    or tighten -- a looser one at a bigger gallery would be the small-sweep
    defect calibration.py exists to prevent."""
    small, large = sorted((a, b))
    t_small, _, ok_small = calibration.recommend_threshold(small, max_risk)
    t_large, _, ok_large = calibration.recommend_threshold(large, max_risk)
    assert t_large >= t_small
    assert ok_small or not ok_large, \
        "a target unreachable for a small gallery cannot become reachable"


@given(sizes, risks)
def test_a_reachable_threshold_meets_its_target_and_is_the_loosest(n, max_risk):
    threshold, risk, reachable = calibration.recommend_threshold(n, max_risk)
    assert threshold in [t for t, _ in calibration.SFACE_FMR]
    if reachable:
        assert risk <= max_risk
        looser = [t for t, _ in calibration.SFACE_FMR if t < threshold]
        assert all(calibration.gallery_risk(t, n) > max_risk for t in looser)
    else:
        assert threshold == calibration.SFACE_FMR[-1][0] and risk > max_risk


@given(st.floats(min_value=0.0, max_value=1.0), sizes, sizes)
def test_gallery_risk_grows_with_the_gallery(threshold, a, b):
    small, large = sorted((a, b))
    lo = calibration.gallery_risk(threshold, small)
    hi = calibration.gallery_risk(threshold, large)
    assert 0.0 <= lo <= hi <= 1.0


# ------------------------------------------------------------- guidance

PART_FLAGS = ["eyes closed", "one eye closed", "face partly obscured",
              "mouth open", "something else"]
maybe_float = st.one_of(st.none(), st.floats(allow_nan=False,
                                             min_value=-1e4, max_value=1e4))

trait_reads = st.fixed_dictionaries({}, optional={
    "detected": st.booleans(),
    "faces": st.integers(min_value=0, max_value=5),
    "facePx": st.one_of(st.none(), st.integers(min_value=0, max_value=800)),
    "yaw": maybe_float,
    "roll": maybe_float,
    "parts": st.fixed_dictionaries({}, optional={
        "flags": st.lists(st.sampled_from(PART_FLAGS), unique=True)}),
    "shadowClip": st.one_of(st.none(), st.floats(0, 1)),
    "highlightClip": st.one_of(st.none(), st.floats(0, 1)),
    "sharpness": maybe_float,
    "qualityScore": st.one_of(st.none(), st.floats(0, 1)),
})
frame_shapes = st.one_of(st.none(), st.tuples(st.integers(1, 1080),
                                              st.integers(1, 1920)))
KEYS = {"severity", "message", "detail", "ready"}


def fired(traits, shape):
    reading = guidance.Reading(traits, shape)
    return [out for out in (rule(reading) for rule in guidance.RULES)
            if out is not None]


@given(trait_reads, frame_shapes, st.sampled_from(["idle", "register"]))
def test_guidance_gives_one_instruction_and_the_first_rule_wins(traits, shape,
                                                               mode):
    # A read with no keys at all is the "Starting camera" case, handled
    # before the rules; give it the one key that makes it a real read.
    traits = dict(traits, detected=traits.get("detected", False))
    out = guidance.instruction(traits, shape, mode)
    assert isinstance(out, dict) and set(out) == KEYS
    assert isinstance(out["message"], str) and out["message"]

    firing = fired(traits, shape)
    if firing:
        assert out == firing[0]
        assert out["ready"] is False and out["severity"] in ("block", "warn")
    else:
        assert out["ready"] is True and out["severity"] == "ok"


@given(trait_reads, frame_shapes)
def test_a_lower_priority_problem_never_displaces_the_instruction(traits,
                                                                  shape):
    """Worsening image quality -- the last rule -- must not change what a read
    that already has a problem is told."""
    traits = dict(traits, detected=traits.get("detected", False))
    before = guidance.instruction(traits, shape)
    worse = guidance.instruction(dict(traits, qualityScore=0.0), shape)
    if not before["ready"]:
        assert worse == before
    else:
        assert worse["message"] == "Improve lighting"


# ------------------------------------------------------------ json_safe

def finite_or_not():
    return st.floats(allow_nan=True, allow_infinity=True)


leaves = st.one_of(
    st.none(), st.booleans(), st.integers(), st.text(max_size=5),
    finite_or_not(),
    finite_or_not().map(np.float32),
    finite_or_not().map(np.float64),
    st.integers(-2**31, 2**31 - 1).map(np.int32),
    st.lists(finite_or_not(), max_size=4).map(
        lambda xs: np.array(xs, dtype=np.float32)),
    st.lists(st.lists(finite_or_not(), min_size=2, max_size=2),
             max_size=3).map(lambda xs: np.array(xs, dtype=np.float64)),
    finite_or_not().map(np.array),                 # a 0-d array
)
nested = st.recursive(
    leaves,
    lambda inner: st.one_of(st.lists(inner, max_size=4),
                            st.lists(inner, max_size=4).map(tuple),
                            st.dictionaries(st.text(max_size=4), inner,
                                            max_size=4)),
    max_leaves=25)


def _app():
    from app import json_safe
    return json_safe


@given(nested)
def test_json_safe_never_emits_nan_or_infinity(value):
    """allow_nan=False is the browser's JSON.parse: it raises on the bare
    NaN / Infinity tokens Python would otherwise write."""
    safe = _app()(value)
    json.dumps(safe, allow_nan=False)
    assert all(math.isfinite(x) for x in floats_in(safe))


def floats_in(value):
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from floats_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from floats_in(v)


@given(st.recursive(
    st.one_of(st.none(), st.integers(), st.text(max_size=4),
              st.floats(allow_nan=False, allow_infinity=False)),
    lambda inner: st.one_of(st.lists(inner, max_size=4),
                            st.dictionaries(st.text(max_size=4), inner,
                                            max_size=4)),
    max_leaves=25))
def test_json_safe_leaves_plain_finite_data_alone(value):
    assert _app()(value) == value


# --------------------------------------------------------- liveness vote

LIVE = liveness.LIVE_THRESHOLD
scores = st.one_of(st.none(), st.floats(0.0, 1.0))
people = st.one_of(st.none(), st.integers(1, 4))
steps = st.lists(st.one_of(
    st.tuples(st.just("follow"), people),
    st.tuples(st.just("push"), scores),
    st.tuples(st.just("reset"), st.none()),
), max_size=40)
windows = st.integers(1, 9).flatmap(
    lambda w: st.tuples(st.just(w), st.integers(1, w)))


@given(steps, windows)
def test_the_vote_never_passes_anyone_on_frames_from_before_they_appeared(
        ops, window):
    """A model of what the vote may count: only frames pushed since it last
    started over -- a reset, or a different person recognised. "live" with
    fewer passing frames than that would be a previous person's evidence
    vouching for this one."""
    size, required = window
    vote = liveness.LivenessVote(window=size, required=required)
    since, who = [], None
    for op, arg in ops:
        if op == "follow":
            vote.follow(arg)
            if arg is not None and arg != who:
                since, who = [], arg
        elif op == "push":
            vote.push(arg)
            if arg is not None:
                since = (since + [arg])[-size:]
        else:
            vote.reset()
            since, who = [], None
        if vote.verdict() == "live":
            assert sum(s >= LIVE for s in since) >= required


@given(st.lists(scores, max_size=12), st.integers(1, 4), st.integers(1, 4),
       st.lists(st.floats(0.0, LIVE, exclude_max=True), min_size=1,
                max_size=12))
def test_a_photograph_never_passes_on_the_last_persons_frames(
        before, first, second, photo):
    """The third audit's attack, as a property: anyone at all in front of the
    camera first, then somebody else whose every frame scores as a spoof."""
    if first == second:
        second = first + 1
    vote = liveness.LivenessVote()
    vote.follow(first)
    for s in before:
        vote.push(s)
    vote.follow(second)
    for s in photo:
        vote.push(s)
        assert vote.verdict() != "live"


class Recogniser:
    def __init__(self):
        self.user_id = None

    def predict(self, face):
        return self.user_id, 10.0          # always accepted


@given(st.lists(st.floats(LIVE, 1.0), max_size=10),
       st.lists(st.floats(0.0, LIVE, exclude_max=True), min_size=1,
                max_size=10))
def test_decide_never_marks_a_new_person_on_the_last_persons_frames(live,
                                                                   photo):
    """The same property through decision.decide, the one function both front
    ends call, so a front end cannot skip the vote's follow()."""
    frame = np.full((240, 320, 3), 128, np.uint8)
    box = (100, 60, 100, 100)
    vote = liveness.LivenessVote()
    recogniser = Recogniser()
    queue = []
    marked = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(liveness, "score", lambda bgr, b: queue.pop(0))
        mp.setattr(decision, "record",
                   lambda uid, conf, method=decision.LBPH:
                   marked.append(uid) or decision.LOGGED)
        for user_id, frames in ((1, live), (2, photo)):
            recogniser.user_id = user_id
            for s in frames:
                queue.append(s)
                decision.decide(frame, box, None, vote, recognizer=recogniser)
    assert 2 not in marked
