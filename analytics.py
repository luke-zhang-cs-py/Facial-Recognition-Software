"""
analytics.py
-------------
Dataset-level analysis: walk everything in dataset/, extract per-image traits
(cached in the sample_traits table), and roll them up into two reports.

1. Enrollment quality — per person, how good are their 30 samples? Which are
   blurry, dark, or turned away, and is there enough pose variety for the
   recogniser to generalise? Bad enrollment is the most common cause of
   "it keeps saying Unknown", and it is invisible until you measure it.

2. Recognition analytics — how separable are the enrolled people actually,
   and what confidence threshold does *your* data justify? attendance.py
   ships CONFIDENCE_THRESHOLD = 70 as a guess. This replaces the guess with
   a sweep over measured accept and false-match rates.

Two evaluation protocols, because the two recognisers cost very different
amounts to retrain:

    SFace   leave-one-out. Embeddings are computed once, so LOO is just
            nearest-neighbour arithmetic and effectively free.
    LBPH    5-fold cross-validation. LBPH has to be retrained per split, so
            true LOO would mean one retrain per image. 5 folds gives an
            honest held-out estimate for 5 trainings.

Both are held-out: no image is ever scored by a model that had already seen
it. Numbers from scoring the training set would look far better and mean
nothing.
"""

import os

import paths
from collections import Counter, defaultdict

import cv2
import numpy as np

import calibration
import db
import facemodels
import traits


LBPH_SWEEP = list(range(30, 131, 10))
SFACE_SWEEP = [round(x, 2) for x in np.arange(0.20, 0.71, 0.05)]
KFOLDS = 5

# LBPH is trained and queried at a fixed size; every sample is resized to it.
# It was written out as a bare (200, 200) at the two call sites, where a
# mismatch between the two would train on one geometry and predict on
# another, and the only symptom would be worse numbers.
LBPH_INPUT_SIZE = (200, 200)

# SFace's own documented operating point for "same person" on cosine.
SFACE_REFERENCE = calibration.SFACE_REFERENCE

# Below this many enrolled people, a locally-swept threshold means nothing:
# there are too few ways to be wrong for a 0% false-match reading to be
# evidence of anything. The recommendation falls back to calibration.py,
# which was measured on ~98k identities.
MIN_USERS_FOR_LOCAL_SWEEP = 15

# Thresholds that turn per-sample flags into advice. Named because "0.75" in
# the middle of a conditional says nothing about what it is a fraction of.
USABLE_FRACTION = 0.75      # below this, recommend recapturing outright
SOFT_FRACTION = 0.15        # share of soft samples worth mentioning
DARK_FRACTION = 0.25        # share flagged dark before lighting advice fires
UNDETECTED_FRACTION = 0.30  # share that will not re-detect before warning
FLAT_POSE_DEGREES = 5.0     # yaw spread below this is one angle repeated
EXPECTED_SAMPLES = 20       # fewer than this is a thin enrollment

# What a completed enrollment actually produces: the sum of camera.py's
# CAPTURE_PLAN counts. Not imported from there -- analytics is used by
# scripts that never touch a camera -- so a test asserts the two agree.
# The advice below used to say "30" as a bare number in a sentence, three
# hundred lines from the constant that decides when to show it, and nothing
# connected the two.
INTENDED_SAMPLES = 30


# --------------------------------------------------------------- collecting

def iter_sample_paths():
    """Yield (user_id, folder_label, path) for every image under dataset/."""
    if not os.path.isdir(paths.dataset_dir()):
        return
    for folder in sorted(os.listdir(paths.dataset_dir())):
        full = os.path.join(paths.dataset_dir(), folder)
        if not os.path.isdir(full):
            continue
        try:
            user_id = int(folder.split("_")[0])
        except ValueError:
            continue
        for name in sorted(os.listdir(full)):
            path = os.path.join(full, name)
            if os.path.isfile(path):
                yield user_id, folder, path


def _row_to_record(row):
    """Turn a cached DB row back into the in-memory shape analytics uses."""
    emb = row["embedding"]
    return {
        "path": row["path"],
        "userId": row["user_id"],
        "sharpness": row["sharpness"],
        "brightness": row["brightness"],
        "contrast": row["contrast"],
        "quality": row["quality"],
        "facePx": row["face_px"],
        "yaw": row["yaw"],
        "roll": row["roll"],
        "detected": bool(row["detected"]),
        "flags": [f for f in (row["flags"] or "").split("|") if f],
        "embedding": np.frombuffer(emb, dtype=np.float32) if emb else None,
        "age": row["age_label"],
        "ageConf": row["age_conf"],
        "gender": row["gender_label"],
        "genderConf": row["gender_conf"],
    }


def analyze_sample(user_id, path, use_cache=True):
    """Trait-analyse one image, using the DB cache when the file is unchanged."""
    mtime = os.path.getmtime(path)
    if use_cache:
        cached = db.get_cached_traits(path, mtime)
        if cached:
            return _row_to_record(cached)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    t = traits.analyze(img)
    if t is None:
        return None

    geom = t.get("geometry") or {}
    demo = t.get("demographics") or {}
    age = (demo.get("age") or {})
    gender = (demo.get("gender") or {})
    emb = t.get("embedding")

    # Returns False when the folder's user id has no row in `users`; the
    # analysis is still valid, it just is not cacheable. scan() reports those
    # folders under "orphanFolders".
    db.save_traits({
        "path": path, "user_id": user_id, "mtime": mtime,
        "sharpness": t["sharpness"], "brightness": t["brightness"],
        "contrast": t["contrast"], "quality": t["qualityScore"],
        "face_px": t["facePx"], "yaw": geom.get("yaw"), "roll": geom.get("roll"),
        "detected": int(t["detected"]), "flags": "|".join(t["flags"]),
        "embedding": emb.astype(np.float32).tobytes() if emb is not None else None,
        "age_label": age.get("label"), "age_conf": age.get("confidence"),
        "gender_label": gender.get("label"), "gender_conf": gender.get("confidence"),
    })

    return {
        "path": path, "userId": user_id,
        "sharpness": t["sharpness"], "brightness": t["brightness"],
        "contrast": t["contrast"], "quality": t["qualityScore"],
        "facePx": t["facePx"], "yaw": geom.get("yaw"), "roll": geom.get("roll"),
        "detected": t["detected"], "flags": t["flags"], "embedding": emb,
        "age": age.get("label"), "ageConf": age.get("confidence"),
        "gender": gender.get("label"), "genderConf": gender.get("confidence"),
    }


# ------------------------------------------------------------ summarising

def _stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    return {
        "mean": round(float(arr.mean()), 1),
        "min": round(float(arr.min()), 1),
        "max": round(float(arr.max()), 1),
        "std": round(float(arr.std()), 1),
    }


def _modal(values, confidences):
    """Most common label across a person's samples, plus how much they agree.

    Agreement matters more than the label: if 30 photos of one person give
    three different age buckets, the estimate is noise regardless of what
    the plurality says.
    """
    pairs = [(v, c) for v, c in zip(values, confidences) if v]
    if not pairs:
        return None
    labels = [v for v, _ in pairs]
    counts = Counter(labels)
    label, n = counts.most_common(1)[0]
    confs = [c for v, c in pairs if v == label and c is not None]
    return {
        "label": label,
        "agreement": round(n / len(labels), 2),
        "meanConfidence": round(float(np.mean(confs)), 3) if confs else None,
        "distinctLabels": len(counts),
        "samples": len(labels),
    }


def _mark_relative_blur(records):
    """Flag samples much softer than the rest of this person's set.

    Laplacian variance is not comparable between people or cameras, so an
    absolute cutoff either misses real blur or condemns a whole enrollment.
    Comparing each sample against its own cohort's median is scale-free: it
    asks "is this one soft *for this person*", which is the actual question.
    """
    sharps = [r["sharpness"] for r in records if r["sharpness"] is not None]
    if len(sharps) < 4:
        return
    median = float(np.median(sharps))
    if median <= 0:
        return
    for r in records:
        s = r["sharpness"]
        if s is not None and s < traits.SOFT_RATIO * median and "blurry" not in r["flags"]:
            r["flags"] = r["flags"] + ["soft focus"]


def _worst_samples(records, limit=5):
    """The specific files to delete and recapture, worst first."""
    flagged = [r for r in records if r["flags"]]
    flagged.sort(key=lambda r: (-len(r["flags"]), r["sharpness"] or 0))
    return [{"file": os.path.basename(r["path"]),
             "reasons": r["flags"],
             "sharpness": r["sharpness"],
             "brightness": r["brightness"]}
            for r in flagged[:limit]]


def _recommendations(records, flags, usable, yaw_spread):
    """What to tell somebody about their captured samples, worst first.

    Each rule is here because it names something the person can actually do
    differently next time. A summary that only reports numbers leaves the
    reader to work out what a yaw spread of 1.2 degrees is supposed to mean
    to them.
    """
    total = len(records)
    if not total:
        return []

    advice = []
    if usable / total < USABLE_FRACTION:
        advice.append("More than a quarter of samples are flagged — recapture.")

    soft = flags.get("blurry", 0) + flags.get("soft focus", 0)
    if soft > total * SOFT_FRACTION:
        advice.append(
            f"{soft} samples are noticeably softer than the rest — hold still, or add "
            f"light so the camera picks a shorter exposure.")

    if flags.get("too dark", 0) > total * DARK_FRACTION or flags.get("flat contrast", 0):
        advice.append("Add light in front of the face, not behind it.")

    if yaw_spread is not None and yaw_spread < FLAT_POSE_DEGREES:
        advice.append("Every sample is the same angle — turn your head "
                      "a little while capturing.")

    if total < EXPECTED_SAMPLES:
        # Two different numbers doing two different jobs: EXPECTED_SAMPLES is
        # when to complain, INTENDED_SAMPLES is what to aim for. The sentence
        # used to carry the second one as a bare literal.
        advice.append(f"Only {total} samples; {INTENDED_SAMPLES} is the intended count.")

    undetected = sum(1 for r in records if not r["detected"])
    if undetected > total * UNDETECTED_FRACTION:
        advice.append(
            f"{undetected} samples did not re-detect as faces — the crops may be too tight.")

    return advice


def summarize_user(user_id, name, records):
    _mark_relative_blur(records)

    flags = Counter()
    for r in records:
        for f in r["flags"]:
            flags[f] += 1

    usable = sum(1 for r in records if not r["flags"])
    yaws = [r["yaw"] for r in records if r["yaw"] is not None]
    yaw_spread = round(float(np.std(yaws)), 1) if len(yaws) > 1 else None
    yaw_range = (round(float(min(yaws)), 1), round(float(max(yaws)), 1)) if yaws else None

    recommendations = _recommendations(records, flags, usable, yaw_spread)

    return {
        "userId": user_id,
        "name": name,
        "samples": len(records),
        "usable": usable,
        "flags": dict(flags.most_common()),
        "sharpness": _stats([r["sharpness"] for r in records]),
        "brightness": _stats([r["brightness"] for r in records]),
        "contrast": _stats([r["contrast"] for r in records]),
        "quality": _stats([r["quality"] for r in records]),
        "facePx": _stats([r["facePx"] for r in records]),
        "yawSpread": yaw_spread,
        "yawRange": yaw_range,
        "worstSamples": _worst_samples(records),
        "age": _modal([r["age"] for r in records], [r["ageConf"] for r in records]),
        "gender": _modal([r["gender"] for r in records], [r["genderConf"] for r in records]),
        "verdict": "good" if not recommendations else "needs work",
        "recommendations": recommendations,
    }


# ------------------------------------------------------- recognition (SFace)

def sface_analysis(records):
    """Leave-one-out nearest-neighbour over SFace embeddings."""
    usable = [r for r in records if r["embedding"] is not None]
    by_user = defaultdict(list)
    for r in usable:
        by_user[r["userId"]].append(r)

    if len(by_user) < 2:
        return {"available": False,
                "reason": f"Needs at least 2 registered people to measure separability "
                          f"(found {len(by_user)})."}

    mat = np.vstack([r["embedding"] for r in usable])
    labels = np.array([r["userId"] for r in usable])
    sims = mat @ mat.T
    np.fill_diagonal(sims, -np.inf)  # leave-one-out: never match against yourself

    # A sample whose owner has no *other* usable image cannot be matched to
    # anything under leave-one-out: there is nothing left to match it to. That
    # used to be recorded as a similarity of -inf and averaged in with the
    # rest, which dragged the whole genuine distribution to -inf -- and -inf
    # is not valid JSON, so the report reached the browser as a parse error
    # rather than as a number. It is a property of the dataset, not a score,
    # so it is counted and reported separately.
    genuine_best, impostor_best, correct, unmatchable = [], [], 0, 0
    for i in range(len(usable)):
        same = labels == labels[i]
        same[i] = False
        other = labels != labels[i]

        g = float(sims[i][same].max()) if same.any() else None
        m = float(sims[i][other].max()) if other.any() else None

        if g is None:
            unmatchable += 1
        else:
            genuine_best.append(g)
        if m is not None:
            impostor_best.append(m)
        if g is not None and m is not None and g > m:
            correct += 1

    if not genuine_best:
        return {"available": False,
                "reason": f"Every person has only one usable image, so there is "
                          f"nothing to match against ({len(usable)} samples)."}

    genuine = np.array(genuine_best)
    impostor = np.array(impostor_best)
    evaluated = len(genuine_best)

    sweep = []
    for t in SFACE_SWEEP:
        sweep.append({
            "threshold": t,
            "accept": round(100.0 * float((genuine >= t).mean()), 1),
            "falseMatch": round(100.0 * float((impostor >= t).mean()), 1),
        })

    # The local sweep can only see the impostors that exist in this dataset.
    # With a handful of enrolled people that is a handful of chances to be
    # wrong, so "0% false matches" is close to guaranteed and means nothing.
    # Anything below MIN_USERS_FOR_LOCAL_SWEEP defers to the large-corpus
    # calibration instead of trusting its own numbers.
    n_users = len(by_user)
    trust_local = n_users >= MIN_USERS_FOR_LOCAL_SWEEP

    best = best_threshold(sweep)

    cal_threshold, cal_risk, cal_ok = calibration.recommend_threshold(n_users)
    recommended = best["threshold"] if trust_local else cal_threshold

    # Which two people are most confusable?
    pairs = []
    user_ids = sorted(by_user)
    for a in range(len(user_ids)):
        for b in range(a + 1, len(user_ids)):
            ua, ub = user_ids[a], user_ids[b]
            block = sims[np.ix_(labels == ua, labels == ub)]
            pairs.append({"a": ua, "b": ub,
                          "maxSimilarity": round(float(block.max()), 3),
                          "meanSimilarity": round(float(block.mean()), 3)})
    pairs.sort(key=lambda p: -p["maxSimilarity"])

    return {
        "available": True,
        "protocol": "leave-one-out 1-NN over SFace cosine similarity",
        # Over the samples that could be evaluated. Counting the unmatchable
        # ones as failures would deflate the figure for a reason that has
        # nothing to do with the model, and would not say so.
        "samples": evaluated,
        "unmatchableSamples": unmatchable,
        "users": len(by_user),
        "accuracy": round(100.0 * correct / evaluated, 1),
        "genuine": {"mean": round(float(genuine.mean()), 3),
                    "min": round(float(genuine.min()), 3)},
        "impostor": {"mean": round(float(impostor.mean()), 3),
                     "max": round(float(impostor.max()), 3)},
        "margin": round(float(genuine.mean() - impostor.mean()), 3),
        "sweep": sweep,
        "recommendedThreshold": recommended,
        "recommendedAccept": best["accept"],
        "recommendedFalseMatch": best["falseMatch"],
        "referenceThreshold": SFACE_REFERENCE,
        "weakestPairs": pairs[:5],
        # Gallery-size-aware guidance from the large-corpus calibration.
        "localSweepTrusted": trust_local,
        "localSweepThreshold": best["threshold"],
        "calibration": {
            "corpus": calibration.CORPUS,
            "threshold": cal_threshold,
            "galleryRisk": round(cal_risk, 5),
            "reachable": cal_ok,
            "gallerySize": n_users,
            "summary": calibration.describe(n_users),
            "riskAtLocalChoice": round(
                calibration.gallery_risk(best["threshold"], n_users), 5),
            "disparity": calibration.FALSE_MATCH_DISPARITY,
        },
        "warning": None if trust_local else (
            f"Only {n_users} people enrolled. A threshold swept on this few "
            f"identities cannot measure false matches meaningfully, so the "
            f"recommendation comes from {calibration.CORPUS} instead."),
    }


# -------------------------------------------------------- recognition (LBPH)

def best_threshold(sweep):
    """The threshold to recommend from a sweep.

    Prefer the most permissive setting that lets nothing wrong through; if
    every setting admits a false match, fall back to the best trade-off. A
    false match is somebody being marked present as somebody else, which is
    worse than a rejection the person can retry.

    Written out twice -- once in sface_analysis and once in lbph_analysis,
    in two different spellings of the same conditional. Same rule, so one
    copy of it.
    """
    clean = [s for s in sweep if s["falseMatch"] == 0.0]
    if clean:
        return max(clean, key=lambda s: s["accept"])
    return max(sweep, key=lambda s: s["accept"] - s["falseMatch"])


def _lbph_sweep(results):
    """Accept and false-match rates across the candidate thresholds.

    LBPH confidence is a DISTANCE, not a similarity: a match is accepted when
    it is *below* the threshold. Getting that backwards is the kind of
    inversion that produces a plausible-looking curve pointing the wrong way.
    """
    total = len(results)
    sweep = []
    for threshold in LBPH_SWEEP:
        accepted = [(true, pred) for true, pred, conf in results if conf < threshold]
        n_correct = sum(1 for true, pred in accepted if true == pred)
        n_wrong = len(accepted) - n_correct
        sweep.append({
            "threshold": threshold,
            "accept": round(100.0 * n_correct / total, 1),
            "falseMatch": round(100.0 * n_wrong / total, 1),
        })
    return sweep


def _stratify(by_user, folds):
    """Assign every sample to a fold, spreading each person evenly.

    Round-robin per person rather than a global shuffle: with a handful of
    samples each, a random split can leave a fold with nobody from a given
    person in training, and that fold then measures nothing useful.
    """
    assignment = []
    # Not `paths`: that is the name of an imported module in this file, and a
    # loop variable shadowing it means the next person to reach for
    # paths.DATASET inside this function gets an AttributeError on a list.
    for user_id, sample_paths in by_user.items():
        for i, path in enumerate(sorted(sample_paths)):
            assignment.append((i % folds, user_id, path))
    return assignment


def _load_square(path):
    """A greyscale sample at the size LBPH is trained on, or None."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if img is None else cv2.resize(img, LBPH_INPUT_SIZE)


def _train_lbph(train_pairs):
    """An LBPH model over these (user, path) pairs, or None if there is not
    enough to train on. LBPH needs at least two identities to separate."""
    images, labels = [], []
    for user_id, path in train_pairs:
        img = _load_square(path)
        if img is not None:
            images.append(img)
            labels.append(user_id)
    if len(set(labels)) < 2:
        return None
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, np.array(labels))
    return model


def _score_fold(model, test_pairs):
    """(true, predicted, confidence) for every readable sample in the fold."""
    scored = []
    for user_id, path in test_pairs:
        img = _load_square(path)
        if img is None:
            continue
        pred, conf = model.predict(img)
        scored.append((user_id, int(pred), float(conf)))
    return scored


def lbph_analysis(records, folds=KFOLDS):
    """K-fold cross-validation of the LBPH recogniser attendance.py uses."""
    by_user = defaultdict(list)
    for r in records:
        by_user[r["userId"]].append(r["path"])

    if len(by_user) < 2:
        return {"available": False,
                "reason": f"Needs at least 2 registered people (found {len(by_user)})."}

    assignment = _stratify(by_user, folds)
    usable_folds = sorted({f for f, _, _ in assignment})

    results = []
    for fold in usable_folds:
        train = [(u, p) for f, u, p in assignment if f != fold]
        test = [(u, p) for f, u, p in assignment if f == fold]
        if not train or not test:
            continue
        model = _train_lbph(train)
        if model is not None:
            results += _score_fold(model, test)

    if not results:
        return {"available": False, "reason": "Not enough samples to cross-validate."}

    correct = sum(1 for true, pred, _ in results if true == pred)
    sweep = _lbph_sweep(results)
    best = best_threshold(sweep)

    confs = np.array([c for _, _, c in results])
    return {
        "available": True,
        "protocol": f"{len(usable_folds)}-fold cross-validation of LBPH",
        "samples": len(results),
        "users": len(by_user),
        "accuracy": round(100.0 * correct / len(results), 1),
        "confidence": {"mean": round(float(confs.mean()), 1),
                       "min": round(float(confs.min()), 1),
                       "max": round(float(confs.max()), 1)},
        "sweep": sweep,
        "recommendedThreshold": best["threshold"],
        "recommendedAccept": best["accept"],
        "recommendedFalseMatch": best["falseMatch"],
        "currentThreshold": 70,
    }


# ------------------------------------------------------------------ top API

def scan(progress=None, use_cache=True):
    """Analyse the whole dataset and return the full report."""
    db.init_db()
    names = dict(db.get_all_users())

    paths = list(iter_sample_paths())
    records, folders = [], {}
    for i, (user_id, folder, path) in enumerate(paths):
        folders[user_id] = folder
        rec = analyze_sample(user_id, path, use_cache=use_cache)
        if rec:
            records.append(rec)
        if progress:
            progress(i + 1, len(paths))

    by_user = defaultdict(list)
    for r in records:
        by_user[r["userId"]].append(r)

    users = [
        summarize_user(uid, names.get(uid) or folders.get(uid, f"id {uid}"), recs)
        for uid, recs in sorted(by_user.items())
    ]

    # Folders whose id has no matching row in `users`. Their samples are still
    # analysed, but nothing links them to a person, so attendance can never be
    # logged for them -- worth saying out loud rather than silently ignoring.
    orphans = [{"userId": uid, "folder": folders[uid],
                "samples": len(by_user[uid])}
               for uid in sorted(by_user) if uid not in names]

    return {
        "models": facemodels.available(),
        "totalSamples": len(records),
        "totalUsers": len(by_user),
        "users": users,
        "orphanFolders": orphans,
        "sface": sface_analysis(records),
        "lbph": lbph_analysis(records),
        "notes": [
            traits.AGE_CAVEAT,
            traits.GENDER_CAVEAT,
            "dataset/ stores greyscale crops; the age, gender and quality nets "
            "expect colour, so live-camera reads are more reliable than these.",
        ],
    }
