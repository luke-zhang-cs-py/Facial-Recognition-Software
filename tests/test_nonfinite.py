"""Numbers that are not numbers, and what they do to a JSON response.

`float('-inf')` and `float('nan')` serialise as the bare tokens `-Infinity`
and `NaN`. Python's json module writes them without complaint and reads them
back again, so nothing looks wrong from the server side. The browser's
JSON.parse rejects both, so the page fails with a parse error naming no
field -- the report simply does not appear, and the reason is invisible.

The analysis report is the endpoint with the most measured numbers on it,
and it was the one endpoint not passing them through json_safe.
"""

import json

import numpy as np
import pytest

import analytics
import app as web


def embedding(uid, vec):
    v = np.asarray(vec, dtype=np.float32)
    return {"userId": uid, "name": f"user{uid}", "path": f"{uid}.jpg",
            "embedding": v / np.linalg.norm(v)}


def strict_loads(raw):
    """json.loads the way a browser does it: no NaN, no Infinity."""
    def reject(token):
        raise ValueError(f"not valid JSON: {token}")
    return json.loads(raw, parse_constant=reject)


# ------------------------------------------------------------- the backstop

@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_json_safe_nulls_a_non_finite_float(value):
    assert web.json_safe(value) is None


@pytest.mark.parametrize("value", [np.float32("inf"), np.float64("nan"),
                                   np.float32("-inf")])
def test_json_safe_nulls_a_non_finite_numpy_scalar(value):
    """numpy scalars used to be unwrapped with .item() and returned as-is,
    which turns a numpy inf into a Python inf and changes nothing."""
    assert web.json_safe(value) is None


def test_json_safe_leaves_real_numbers_alone():
    assert web.json_safe(1.5) == 1.5
    assert web.json_safe(0.0) == 0.0
    assert web.json_safe(np.float32(2.5)) == pytest.approx(2.5)
    assert web.json_safe(-3) == -3


def test_json_safe_reaches_into_nested_structures():
    out = web.json_safe({"a": [{"b": float("nan")}], "c": (1.0, float("inf"))})
    assert out == {"a": [{"b": None}], "c": [1.0, None]}


def test_json_safe_output_always_parses_strictly():
    payload = {"nested": {"values": [float("nan"), 1.0, np.float32("-inf")]},
               "scalar": np.float64("inf")}
    strict_loads(json.dumps(web.json_safe(payload)))


# ------------------------------------------------------------- at the source

def test_a_person_with_one_image_does_not_poison_the_report():
    """Leave-one-out has nothing to match a lone sample against. That used to
    be recorded as a similarity of -inf and averaged into the genuine
    distribution, taking the mean and the minimum with it."""
    records = [embedding(1, [1, 0, 0]), embedding(1, [0.9, 0.1, 0]),
               embedding(1, [0.8, 0.2, 0]),
               embedding(2, [0, 1, 0])]          # user 2 has exactly one
    report = analytics.sface_analysis(records)

    assert report["available"]
    for key in ("mean", "min"):
        assert np.isfinite(report["genuine"][key]), report["genuine"]
    assert np.isfinite(report["margin"])
    strict_loads(json.dumps(web.json_safe(report)))


def test_the_unmatchable_sample_is_counted_not_hidden():
    """It is a fact about the dataset, not a score. Counting it as a failed
    identification would deflate the accuracy for a reason that has nothing
    to do with the model, and would not say so anywhere."""
    records = [embedding(1, [1, 0, 0]), embedding(1, [0.9, 0.1, 0]),
               embedding(2, [0, 1, 0])]
    report = analytics.sface_analysis(records)
    assert report["unmatchableSamples"] == 1
    assert report["samples"] == 2, "the two that could be evaluated"
    assert report["accuracy"] == 100.0


def test_everybody_having_one_image_is_reported_not_crashed():
    records = [embedding(1, [1, 0, 0]), embedding(2, [0, 1, 0])]
    report = analytics.sface_analysis(records)
    assert report["available"] is False
    assert "one usable image" in report["reason"]


def test_a_normal_dataset_still_measures_the_same_way():
    """The guard must not have changed the ordinary path."""
    records = [embedding(1, [1, 0, 0]), embedding(1, [0.95, 0.05, 0]),
               embedding(2, [0, 1, 0]), embedding(2, [0.05, 0.95, 0])]
    report = analytics.sface_analysis(records)
    assert report["available"]
    assert report["samples"] == 4 and report["unmatchableSamples"] == 0
    assert report["accuracy"] == 100.0
    assert report["genuine"]["mean"] > report["impostor"]["mean"]


def test_one_registered_person_is_still_refused():
    report = analytics.sface_analysis([embedding(1, [1, 0, 0]),
                                       embedding(1, [0.9, 0.1, 0])])
    assert report["available"] is False
    assert "2 registered people" in report["reason"]


# ------------------------------------------------------------- the endpoint

def test_the_analysis_endpoint_passes_its_report_through_json_safe():
    """It was the one endpoint returning measured numbers without it."""
    client = web.app.test_client()
    with web._analysis_lock:
        web._analysis.update(running=False, done=1, total=1, error=None,
                             report={"sface": {"genuine": {"mean": float("-inf")}}})
    try:
        raw = client.get("/api/analysis").get_data(as_text=True)
        assert "Infinity" not in raw and "NaN" not in raw
        body = strict_loads(raw)
        assert body["report"]["sface"]["genuine"]["mean"] is None
    finally:
        with web._analysis_lock:
            web._analysis.update(report=None, done=0, total=0)
