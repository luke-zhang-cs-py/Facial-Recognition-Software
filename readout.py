"""
readout.py
-----------
The live trait panel's data, assembled from a single frame.

What the sidebar shows while the camera is running: is there a face, is it
sharp enough, is it lit, is it pointed at the lens, are the eyes open, and
-- with the caveats `traits.py` attaches -- an age and gender guess.

Why it is not in camera.py
==========================
It was, inside `CameraManager._maybe_traits`, which reached complexity 14 by
doing five separate jobs in one method: rate-limiting against the shared
clock, running the models, reshaping their output into something JSON-safe,
augmenting it with 68-point landmark metrics, and augmenting it again with
demographics.

Only the first of those is the camera's. The rest are transformations of a
dictionary, and as methods on a class that owns a capture device they could
not be tested without one -- which is most of why this was the least covered
part of the project. They are plain functions here.

`camera.py` keeps the rate limit and the locking, because those are about the
shared device and the worker thread.

Everything returned is JSON-safe: no numpy arrays survive past `summarise`,
because the whole point of this data is that it gets serialised to a browser.
"""

import cv2

import guidance
import landmarks as facelandmarks
import traits as facetraits

# The keys copied straight across from a traits.analyze result. Listed rather
# than copied wholesale so that adding a field to traits.py cannot silently
# start shipping it -- and so that anything numpy-shaped stays out.
DIRECT_KEYS = ("detected", "faces", "sharpness", "brightness", "contrast",
               "qualityScore", "facePx", "flags", "usable")

# Present on some results and not others, depending on which checks ran.
OPTIONAL_KEYS = ("shadowClip", "highlightClip", "dynamicRange")

# The two demographic fields, and the sub-keys each carries. `traits.py`
# attaches an uncertainty and a runner-up to both, and the UI shows them:
# a guess without its confidence is the thing this project is careful not
# to present.
DEMOGRAPHIC_KEYS = ("age", "gender")
DEMOGRAPHIC_FIELDS = ("label", "confidence", "uncertain", "runnerUp")


def summarise(result):
    """A JSON-safe dictionary from one `traits.analyze` result.

    `geometry` is flattened to the two angles the panel uses. The rest of it
    -- landmark coordinates, the raw detection row -- is deliberately left
    behind: it is numpy, and nothing on the page reads it.
    """
    geometry = result.get("geometry") or {}
    summary = {key: result[key] for key in DIRECT_KEYS}
    summary.update({key: result.get(key) for key in OPTIONAL_KEYS})
    summary["yaw"] = geometry.get("yaw")
    summary["roll"] = geometry.get("roll")
    return summary


def add_part_metrics(summary, frame):
    """68-point part measurements, if the landmark model can place them.

    Eyes open, mouth neutral, both halves of the face equally visible. A face
    can pass every geometric check and still be unusable because the person
    blinked, which is the gap this closes.

    Swallows its exceptions on purpose, and this is the one place in the
    module where that is right: the parts panel is an enhancement on top of a
    readout that is already complete and useful, and the landmark model is
    the most likely of the five to be missing from a given install. Losing a
    section of the sidebar is the correct degradation; losing the whole
    readout because of it is not.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows = facetraits.detect(frame)
        if not rows:
            return summary
        box = facetraits.geometry(rows[0])["box"]
        points = facelandmarks.fit(gray, box)
        metrics = facelandmarks.metrics(points)
    except Exception:
        return summary

    if metrics:
        summary["parts"] = metrics
        summary["flags"] = list(summary["flags"]) + metrics["flags"]
    return summary


def add_guidance(summary, frame_shape, mode):
    """What to tell the person to do, and the checklist behind it."""
    summary["guidance"] = guidance.instruction(summary,
                                               frame_shape=frame_shape,
                                               mode=mode)
    summary["checklist"] = guidance.checklist(summary,
                                              frame_shape=frame_shape)
    return summary


def add_demographics(summary, result):
    """The age and gender guesses, each with its uncertainty.

    `demographicsSkipped` is carried through when present: "we did not look"
    and "we looked and are unsure" are different answers, and the panel says
    which.
    """
    if result.get("demographicsSkipped"):
        summary["demographicsSkipped"] = result["demographicsSkipped"]

    demographics = result.get("demographics") or {}
    if not demographics:
        return summary

    for key in DEMOGRAPHIC_KEYS:
        found = demographics.get(key)
        if found:
            summary[key] = {field: found[field]
                            for field in DEMOGRAPHIC_FIELDS}

    if demographics.get("age", {}).get("estimate"):
        summary["age"]["estimate"] = demographics["age"]["estimate"]
    return summary


def build(result, frame, mode):
    """The whole panel payload for one frame.

    Split into the four steps above rather than written straight through,
    because each is a transformation worth being able to test on a dictionary
    without a camera attached.
    """
    summary = summarise(result)
    add_part_metrics(summary, frame)
    add_guidance(summary, frame.shape[:2], mode)
    add_demographics(summary, result)
    return summary
