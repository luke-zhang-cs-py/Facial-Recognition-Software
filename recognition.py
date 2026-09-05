"""
recognition.py
---------------
Identify a face against everyone enrolled, using SFace embeddings.

Why not LBPH
------------
attendance.py matches with LBPH, which compares texture histograms. Its
"confidence" is a distance with no fixed meaning: it does not transfer
between datasets, cameras, or gallery sizes, which is why picking a threshold
for it needed a sweep over local data and still did not generalise.

SFace embeddings live in a metric space. Cosine similarity between two of
them means the same thing everywhere, so the thresholds measured over 4.77
billion impostor pairs in calibration.py apply directly, and the recommended
value scales with how many people are enrolled.

Both are kept. LBPH still drives the live camera path; this is what the
image-identification endpoint uses, and it is the better of the two.

A person is represented by the mean of their sample embeddings, re-normalised
-- a centroid is far more stable than any single shot, which is the whole
point of capturing 30 of them across different poses.
"""

import os

import numpy as np

import calibration
import db
import traits

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# Clearing the threshold is not enough on its own. If the best match beats the
# runner-up by only a hair, the pair is being told apart by noise, and naming
# the winner presents a coin toss as an identification. Below this margin the
# answer is "unknown" with the tie reported, which is the honest output and
# the safe one -- a refusal is a person swiping a badge instead, a confident
# wrong name is somebody else's attendance record.
MIN_MARGIN = 0.10


def _centroid(vectors):
    if not vectors:
        return None
    mat = np.vstack(vectors)
    mean = mat.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm else None


def gallery():
    """{user_id: {'name', 'centroid', 'samples'}} for everyone with embeddings."""
    out = {}
    for user_id, name in db.get_all_users():
        rows = db.get_traits_for_user(user_id)
        vecs = [np.frombuffer(r["embedding"], dtype=np.float32)
                for r in rows if r["embedding"]]
        centroid = _centroid(vecs)
        if centroid is not None:
            out[user_id] = {"name": name, "centroid": centroid,
                            "samples": len(vecs)}
    return out


def embed_image(bgr):
    """Embedding for the most prominent face in an image, or None."""
    rows = traits.detect(bgr)
    if not rows:
        return None, None
    row = rows[0]
    vec = traits.embed(bgr, row)
    return vec, traits.geometry(row)


def identify(bgr, gal=None, max_risk=0.01):
    """Match a face against the gallery.

    The threshold comes from calibration.py and depends on how many people are
    enrolled, because the chance of colliding with *somebody* grows with the
    gallery -- the same reasoning as everywhere else in this project. Returning
    the runner-up as well matters: a top match of 0.62 means very different
    things when the next best is 0.20 versus 0.61.
    """
    gal = gallery() if gal is None else gal
    if not gal:
        return {"ok": False, "error": "Nobody is enrolled yet."}

    vec, geom = embed_image(bgr)
    if vec is None:
        return {"ok": False, "error": "No face found in that image."}

    scored = sorted(
        ({"userId": uid, "name": e["name"], "samples": e["samples"],
          "similarity": round(float(np.dot(vec, e["centroid"])), 4)}
         for uid, e in gal.items()),
        key=lambda m: -m["similarity"])

    threshold, risk, reachable = calibration.recommend_threshold(len(gal), max_risk)
    best = scored[0]
    runner = scored[1] if len(scored) > 1 else None
    margin = (round(best["similarity"] - runner["similarity"], 4)
              if runner else None)

    above = best["similarity"] >= threshold
    ambiguous = above and margin is not None and margin < MIN_MARGIN
    if not above:
        reason = "below threshold"
    elif ambiguous:
        reason = "ambiguous — too close to call"
    else:
        reason = None

    return {
        "ok": True,
        "match": None if (not above or ambiguous) else best,
        "verdict": "unknown" if (not above or ambiguous) else "identified",
        "reason": reason,
        "ambiguous": bool(ambiguous),
        "minMargin": MIN_MARGIN,
        "best": best,
        "runnerUp": runner,
        "margin": margin,
        "threshold": threshold,
        "galleryRisk": round(risk, 5),
        "thresholdReachable": reachable,
        "gallerySize": len(gal),
        "candidates": scored[:5],
        "geometry": geom,
    }
