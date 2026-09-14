"""
enrollment.py
--------------
The report produced after a registration finishes.

It measures the *enrollment*, not the person: how well the staged capture
covered the poses it asked for, how good the images are, how separable this
face is from everyone already enrolled, and what that implies for the
recognition threshold.

Why it is not in camera.py
==========================
It was, as `CameraManager._build_report` -- 95 lines reaching complexity 14,
and the only method in the class that imported `analytics`, `calibration` and
`numpy` from inside its own body. Function-level imports in a module that
already imports fifteen things at the top are usually a message, and the
message here was that this work is not the camera's: nothing in it touches
the capture device, the frame buffer or the worker thread. It reads files off
disk and rows out of the database.

Moving it out makes it testable without a webcam, which is most of why the
camera path was the least covered part of the project.

`camera.py` keeps the part that is genuinely its own: taking the registration
state out from under its lock, and putting the finished report back.
"""

import os

import numpy as np

import analytics
import calibration
import db

# A gallery of one has no impostors to calibrate against, so the threshold
# recommendation needs at least a notional pair.
MIN_GALLERY_FOR_THRESHOLD = 2

# Similarity is reported to three places. The underlying cosine is not
# precise to anything like that, but the number is compared against
# calibration's published operating points, which are quoted the same way.
SIMILARITY_PLACES = 3
RISK_PLACES = 5


def samples_from_folder(user_id, folder):
    """Analyse every image in a user's sample folder, newest analysis first.

    `use_cache=False` on purpose: the samples were written seconds ago, and a
    cached analysis from a previous enrollment of the same user would
    describe the images that got replaced.
    """
    records = []
    if not folder or not os.path.isdir(folder):
        return records
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        record = analytics.analyze_sample(user_id, path, use_cache=False)
        if record:
            records.append(record)
    return records


def pose_coverage(poses):
    """How many samples each stage contributed, and the yaw spread.

    This is what the staged capture was for: thirty frames of somebody
    holding still teach the recogniser one angle, and a face turned ten
    degrees at the door then fails to match. The spread is the evidence that
    the staging worked.

    Returns (stages, yawSpread, yawRange). `yawSpread` is None for a single
    sample, where a standard deviation would be 0 and read as "no variation"
    rather than "not enough data to say".
    """
    stages = {}
    for pose in poses:
        stages[pose["stage"]] = stages.get(pose["stage"], 0) + 1

    yaws = [pose["yaw"] for pose in poses if pose.get("yaw") is not None]
    spread = round(float(np.std(yaws)), 1) if len(yaws) > 1 else None
    span = [round(min(yaws), 1), round(max(yaws), 1)] if yaws else None
    return stages, spread, span


def centroid_of(records):
    """The mean sample embedding, re-normalised, or None if there are none.

    A centroid is far more stable than any single shot, which is the whole
    point of capturing thirty of them across different poses.
    """
    vectors = [r["embedding"] for r in records if r["embedding"] is not None]
    if not vectors:
        return None
    centroid = np.mean(np.vstack(vectors), axis=0)
    norm = float(np.linalg.norm(centroid)) or 1.0
    return centroid / norm


def nearest_other(user_id, records):
    """The enrolled person this face is most similar to, or None.

    The number that matters most in the report and the least obvious one: a
    new enrollment that sits close to somebody already on file is the
    condition under which a threshold that looked fine starts producing
    confident wrong names. Better to say so at capture time than to discover
    it from an attendance record.
    """
    centroid = centroid_of(records)
    if centroid is None:
        return None

    best = None
    for other_id, other_name in db.get_all_users():
        if other_id == user_id:
            continue
        rows = db.get_traits_for_user(other_id)
        vectors = [np.frombuffer(row["embedding"], dtype=np.float32)
                   for row in rows if row["embedding"]]
        if not vectors:
            continue
        similarity = float(np.max(np.vstack(vectors) @ centroid))
        if best is None or similarity > best[1]:
            best = (other_name, similarity)

    if best is None:
        return None
    return {"name": best[0],
            "similarity": round(best[1], SIMILARITY_PLACES)}


def build(user_id, name, poses, folder):
    """The whole report, or None if the enrollment produced nothing usable.

    None rather than an empty report: a registration that captured no
    analysable samples has nothing to say about itself, and a report full of
    nulls reads as a measurement rather than an absence.
    """
    records = samples_from_folder(user_id, folder)
    if not records:
        return None

    summary = analytics.summarize_user(user_id, name, records)
    stages, yaw_spread, yaw_range = pose_coverage(poses)
    gallery = len(db.get_all_users())
    threshold, risk, reachable = calibration.recommend_threshold(
        max(gallery, MIN_GALLERY_FOR_THRESHOLD))

    return {
        "userId": user_id,
        "name": name,
        "samples": len(records),
        "usable": summary["usable"],
        "verdict": summary["verdict"],
        "flags": summary["flags"],
        "recommendations": summary["recommendations"],
        "sharpness": summary["sharpness"],
        "quality": summary["quality"],
        "facePx": summary["facePx"],
        "poseStages": stages,
        "yawSpread": yaw_spread,
        "yawRange": yaw_range,
        "nearestOther": nearest_other(user_id, records),
        "gallerySize": gallery,
        "threshold": threshold,
        "thresholdRisk": round(risk, RISK_PLACES),
        "thresholdReachable": reachable,
        "age": summary.get("age"),
        "fairness": calibration.FAIRNESS,
        "sampleAdvice": calibration.describe_samples(len(records)),
        "sampleAccuracy": round(
            calibration.accuracy_for_samples(len(records)), SIMILARITY_PLACES),
        "sampleSaturation": calibration.SAMPLE_SATURATION,
    }
