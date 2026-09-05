"""
tools_benchmark_loop.py
------------------------
Keep testing the recogniser until told to stop, and write down everything.

One benchmark run is a measurement. A benchmark that keeps running is a
distribution: every round draws fresh held-out images and a fresh point in
the degradation space, so the intervals tighten as the sample grows rather
than the same number being recomputed. That is the whole reason to leave it
running -- 100 trials tells you roughly; 100,000 tells you where the edges
are.

What each round does
--------------------
1. **Identity trials (LFW).** For every enrolled person, held-out photographs
   they were *not* enrolled from, each pushed through a random point in the
   operating space: distance, motion blur, rotation, gamma, sensor noise and
   JPEG quality, all varied together. That composite is the closest thing to
   a video frame available without a camera -- a still off a moving webcam is
   exactly a downscaled, motion-blurred, badly-lit, recompressed photograph.

2. **Detection sweep (FairFace).** Whether a face is found at all, broken
   down by the group labels the corpus carries. Recognition accuracy on faces
   the detector never returned is a number about the wrong thing, so this is
   measured separately and reported separately.

3. **Age error (faceage).** Absolute error against labelled ages, which is
   what the +/-2 year display band is answerable to.

Three outcomes for an identity trial, and the distinction is the point:

    CORRECT  named the right person
    UNKNOWN  declined -- under threshold, or too close between two people
    WRONG    named somebody else. The only genuinely dangerous one.

A system that says "I don't know" is inconvenient. A system that confidently
says the wrong name is worse than one that says nothing, so WRONG is tracked
on its own and never folded into an accuracy figure.

Stopping
--------
Create a file called STOP_BENCHMARK in the project directory, or pass
--rounds N. It checks between rounds, so it finishes the round it is in
rather than leaving a partial one in the log.

Output
------
    logs/benchmark.jsonl   one JSON object per round, append-only
    logs/benchmark.log     the same thing, readable
    logs/benchmark_state.json  cumulative totals, rewritten each round

Usage
-----
    python tools_benchmark_loop.py                    # until stopped
    python tools_benchmark_loop.py --rounds 5         # five rounds
    python tools_benchmark_loop.py --trials 200       # per identity per round
"""

import argparse
import collections
import datetime as dt
import io
import json
import math
import os
import sys
import time

import cv2
import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

import paths          # noqa: E402
import recognition    # noqa: E402
import traits         # noqa: E402

TEMP = os.environ.get("TEMP", os.path.join(os.path.expanduser("~"), "tmp"))
LFW_PARQUET = os.path.join(TEMP, "lfw", "lfw.parquet")
FAIRFACE_DIR = os.path.join(TEMP, "claude", "c--Users-justl-Downloads-toronto-transit",
                            "8a4bfe8e-abc0-4a7c-ab43-54af99de2068", "scratchpad", "fairface")
FACEAGE_PARQUET = os.path.join(TEMP, "faceage", "val.parquet")

LOG_DIR = os.path.join(PROJ, "logs")
STOP_FILE = os.path.join(PROJ, "STOP_BENCHMARK")

# How many FairFace / faceage images to draw per round. Smaller than the
# identity pass because those two answer narrower questions and the identity
# trials are what the operating threshold is set from.
DETECTION_SAMPLE = 300
AGE_SAMPLE = 200

# The ground truth and the model do not use the same buckets, so neither can
# be scored against the other directly.
#
#   FairFace labels : 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
#   Adience predicts: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60+
#
# Both go to the midpoint of their band in years and the error is the gap.
# That is generous to the model -- a band is not a point, and a prediction of
# "25-32" is not really a claim that somebody is 28 -- but it is the only
# comparison the two taxonomies allow, and it is the one the 12.4-year MAE in
# BENCHMARK.md was computed with, so the numbers stay comparable.
TRUTH_MIDPOINT = {0: 1, 1: 6, 2: 15, 3: 25, 4: 35, 5: 45, 6: 55, 7: 65, 8: 75}
PREDICTED_MIDPOINT = {"0-2": 1, "4-6": 5, "8-12": 10, "15-20": 17,
                      "25-32": 28, "38-43": 40, "48-53": 50, "60+": 70}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson(hits, n, z=1.96):
    """A 95% interval that behaves at the edges.

    The textbook p +/- z*sqrt(p(1-p)/n) gives [1.0, 1.0] for 40 out of 40,
    which claims certainty from forty samples. Wilson gives [0.91, 1.0] --
    the honest version, and the reason this runs for a long time.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def decode(cell):
    """A parquet image cell -> BGR, or None if it will not decode."""
    from PIL import Image
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    try:
        rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class LFW:
    """Labelled Faces in the Wild: news photographs of public figures.

    Which is why it suits this: thousands of identities, photographed by
    different people, in different years, in whatever light the room had --
    rather than a studio set where every image of a person was taken in one
    sitting and the model is really being asked to recognise the lighting.
    """

    def __init__(self, path):
        import pyarrow.parquet as pq
        handle = pq.ParquetFile(path)
        meta = handle.schema_arrow.metadata[b"huggingface"].decode()
        self.names = json.loads(meta)["info"]["features"]["label"]["names"]
        table = handle.read()
        self.labels = table.column("label").to_pylist()
        self.images = table.column("image").to_pylist()
        handle.close()

        self.by_label = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            self.by_label[label].append(i)
        self.label_of = {n.replace("_", " "): i for i, n in enumerate(self.names)}

    def image(self, index):
        return decode(self.images[index])

    def __len__(self):
        return len(self.labels)


def degrade(img, rng):
    """One random point in the operating space.

    Composite rather than one-factor-at-a-time: the failures worth finding
    are the combinations. A face is recognisable when slightly blurred, and
    recognisable when slightly dark, and neither of those tells you what
    happens when it is both, at an angle, after JPEG.
    """
    h, w = img.shape[:2]

    scale = rng.uniform(0.25, 1.0)                       # distance from camera
    small = cv2.resize(img, (max(16, int(w * scale)), max(16, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    k = int(rng.choice([1, 1, 3, 3, 5, 7, 9]))           # motion blur
    if k >= 3:
        kernel = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)

    degrees = rng.uniform(-18, 18)                        # head not level
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    gamma = rng.uniform(0.45, 2.0)                        # under/over exposure
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], np.uint8)
    img = cv2.LUT(img, lut)

    if rng.random() < 0.5:                                # sensor noise
        noise = rng.normal(0, rng.uniform(2, 10), img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    quality = int(rng.integers(18, 96))                   # stream compression
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else img


# ---------------------------------------------------------------------------
# The three passes
# ---------------------------------------------------------------------------
def identity_round(lfw, gallery, trials, rng):
    """Held-out photographs of every enrolled person, degraded."""
    enrolled = {v["name"].replace("[demo] ", ""): uid
                for uid, v in gallery.items() if v["name"].startswith("[demo] ")}
    sample_counts = {v["name"].replace("[demo] ", ""): v["samples"]
                     for v in gallery.values()}

    per_person = {}
    totals = collections.Counter()
    similarities = []

    for person in sorted(enrolled):
        label = lfw.label_of.get(person)
        if label is None:
            continue
        # Skip the images they were enrolled from: testing against training
        # data measures memory, not recognition.
        indices = lfw.by_label[label]
        held_out = indices[sample_counts.get(person, 3):] or indices[-1:]

        correct = unknown = wrong = 0
        for t in range(trials):
            source = lfw.image(held_out[int(rng.integers(len(held_out)))])
            if source is None:
                continue
            result = recognition.identify(degrade(source, rng), gallery)
            if not result.get("ok") or result.get("match") is None:
                unknown += 1
            elif result["match"]["name"].replace("[demo] ", "") == person:
                correct += 1
                similarities.append(result["best"]["similarity"])
            else:
                wrong += 1

        n = correct + unknown + wrong
        if not n:
            continue
        per_person[person] = {"n": n, "correct": correct,
                              "unknown": unknown, "wrong": wrong}
        totals["correct"] += correct
        totals["unknown"] += unknown
        totals["wrong"] += wrong
        totals["n"] += n

    return {
        "identities": len(per_person),
        "trials": totals["n"],
        "correct": totals["correct"],
        "unknown": totals["unknown"],
        "wrong": totals["wrong"],
        "meanSimilarity": round(float(np.mean(similarities)), 4) if similarities else None,
        "perPerson": per_person,
    }


def detection_round(rng, sample=DETECTION_SAMPLE):
    """Is a face found at all, by group. Skipped if FairFace is not cached."""
    import glob
    files = sorted(glob.glob(os.path.join(FAIRFACE_DIR, "*.parquet")))
    if not files:
        return None

    import pyarrow.parquet as pq
    handle = pq.ParquetFile(files[int(rng.integers(len(files)))])
    table = handle.read()
    handle.close()

    columns = table.column_names
    group_col = "race" if "race" in columns else None
    images = table.column("image").to_pylist()
    groups = table.column(group_col).to_pylist() if group_col else [0] * len(images)

    picks = rng.choice(len(images), size=min(sample, len(images)), replace=False)
    hits = collections.Counter()
    seen = collections.Counter()
    for i in picks:
        img = decode(images[int(i)])
        if img is None:
            continue
        group = str(groups[int(i)])
        seen[group] += 1
        found = traits.analyze(img)
        if found and found.get("detected"):
            hits[group] += 1

    by_group = {}
    for group, n in seen.items():
        low, high = wilson(hits[group], n)
        by_group[group] = {"n": n, "detected": hits[group],
                           "rate": round(hits[group] / n, 4),
                           "ci": [round(low, 4), round(high, 4)]}

    total_n = sum(seen.values())
    total_hits = sum(hits.values())
    rates = [g["rate"] for g in by_group.values() if g["n"] >= 20]
    return {
        "sampled": total_n,
        "detected": total_hits,
        "rate": round(total_hits / total_n, 4) if total_n else None,
        # The number that matters for fairness is the ratio between the best
        # and worst served group, not the average.
        "disparity": round(max(rates) / min(rates), 3) if rates and min(rates) > 0 else None,
        "byGroup": by_group,
    }


def age_round(rng, sample=AGE_SAMPLE):
    """Absolute age error. Skipped if the corpus is not cached."""
    if not os.path.exists(FACEAGE_PARQUET):
        return None
    import pyarrow.parquet as pq
    handle = pq.ParquetFile(FACEAGE_PARQUET)
    table = handle.read()
    handle.close()

    columns = table.column_names
    if "age" not in columns:
        return None
    images = table.column("image").to_pylist()
    ages = table.column("age").to_pylist()

    picks = rng.choice(len(images), size=min(sample, len(images)), replace=False)
    errors = []
    for i in picks:
        truth = TRUTH_MIDPOINT.get(int(ages[int(i)])) if isinstance(ages[int(i)], int) else None
        if truth is None:
            continue
        img = decode(images[int(i)])
        if img is None:
            continue
        found = traits.analyze(img)
        if not found or not found.get("detected"):
            continue
        band = (found.get("demographics") or {}).get("age") or {}
        predicted = PREDICTED_MIDPOINT.get(band.get("label"))
        if predicted is None:
            continue
        errors.append(abs(predicted - truth))

    if not errors:
        return None
    errors = np.array(errors)
    return {
        "sampled": len(errors),
        "mae": round(float(errors.mean()), 2),
        "median": round(float(np.median(errors)), 2),
        # What the displayed band actually delivers, measured rather than
        # assumed. A +/-2 year window is a display choice; this is its cost.
        "within2": round(float((errors <= 2).mean()), 4),
        "within5": round(float((errors <= 5).mean()), 4),
        "within10": round(float((errors <= 10).mean()), 4),
    }


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def append(path, text):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def summarise(state):
    """The running totals, as a line somebody can read."""
    ident = state["cumulative"]["identity"]
    n = ident["trials"]
    if not n:
        return "  no identity trials yet"
    low, high = wilson(ident["correct"], n)
    wrong_low, wrong_high = wilson(ident["wrong"], n)
    return (
        f"  cumulative over {n:,} trials / {state['rounds']} rounds\n"
        f"    correct  {100 * ident['correct'] / n:6.2f}%   "
        f"95% CI [{100 * low:.2f}, {100 * high:.2f}]\n"
        f"    unknown  {100 * ident['unknown'] / n:6.2f}%   (declined, not wrong)\n"
        f"    WRONG    {100 * ident['wrong'] / n:6.2f}%   "
        f"95% CI [{100 * wrong_low:.2f}, {100 * wrong_high:.2f}]"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--rounds", type=int, default=0,
                        help="run N rounds this session (default: until stopped). "
                             "Counted per session, not cumulatively -- the log "
                             "resumes, but --rounds 1 always means one more.")
    parser.add_argument("--trials", type=int, default=60,
                        help="identity trials per enrolled person per round")
    parser.add_argument("--detection-every", type=int, default=3,
                        help="run the FairFace detection sweep every N rounds")
    parser.add_argument("--age-every", type=int, default=5,
                        help="run the age pass every N rounds")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    jsonl = os.path.join(LOG_DIR, "benchmark.jsonl")
    readable = os.path.join(LOG_DIR, "benchmark.log")
    state_file = os.path.join(LOG_DIR, "benchmark_state.json")

    if not os.path.exists(LFW_PARQUET):
        print(f"LFW not cached at {LFW_PARQUET} -- nothing to test against.")
        return 1

    gallery = recognition.gallery()
    if not gallery:
        print("Nobody is enrolled. Run seed_demo.py first.")
        return 1

    print(f"loading LFW from {LFW_PARQUET} ...")
    lfw = LFW(LFW_PARQUET)
    print(f"  {len(lfw):,} photographs, {len(lfw.names):,} identities")
    print(f"  gallery: {len(gallery)} enrolled")
    print(f"  logging to {LOG_DIR}")
    print(f"  stop with: create {STOP_FILE}")
    print()

    state = {"rounds": 0, "startedAt": dt.datetime.now().isoformat(timespec="seconds"),
             "cumulative": {"identity": {"trials": 0, "correct": 0, "unknown": 0, "wrong": 0},
                            "detection": {"sampled": 0, "detected": 0},
                            "age": {"sampled": 0, "errorSum": 0.0}}}
    if os.path.exists(state_file):
        try:
            state = json.load(open(state_file, encoding="utf-8"))
            print(f"  resuming from {state['rounds']} previous rounds\n")
        except Exception:
            pass

    append(readable, f"\n=== benchmark loop started "
                     f"{dt.datetime.now().isoformat(timespec='seconds')} ===")

    this_session = 0
    while True:
        if os.path.exists(STOP_FILE):
            print("STOP_BENCHMARK found -- stopping.")
            append(readable, "  stopped: STOP_BENCHMARK present")
            break
        # Counted per session. The cumulative total in the state file is what
        # tightens the intervals; --rounds is how many more to add now, so
        # `--rounds 1` does something on the fiftieth run as well as the first.
        if args.rounds and this_session >= args.rounds:
            break
        this_session += 1

        round_no = state["rounds"] + 1
        # A fresh seed per round: the point of running repeatedly is to draw
        # different images and different degradations, not to recompute one
        # sample with better precision.
        rng = np.random.default_rng(int(time.time() * 1000) % (2 ** 32))
        started = time.time()

        record = {"round": round_no,
                  "at": dt.datetime.now().isoformat(timespec="seconds")}

        record["identity"] = identity_round(lfw, gallery, args.trials, rng)

        if args.detection_every and round_no % args.detection_every == 0:
            record["detection"] = detection_round(rng)
        if args.age_every and round_no % args.age_every == 0:
            record["age"] = age_round(rng)

        record["seconds"] = round(time.time() - started, 1)

        # accumulate
        ident = record["identity"]
        cume = state["cumulative"]["identity"]
        for key in ("trials", "correct", "unknown", "wrong"):
            cume[key] += ident[key]
        if record.get("detection"):
            det = state["cumulative"]["detection"]
            det["sampled"] += record["detection"]["sampled"]
            det["detected"] += record["detection"]["detected"]
        if record.get("age"):
            age = state["cumulative"]["age"]
            age["sampled"] += record["age"]["sampled"]
            age["errorSum"] += record["age"]["mae"] * record["age"]["sampled"]

        state["rounds"] = round_no
        state["updatedAt"] = record["at"]

        append(jsonl, json.dumps(record))
        line = (f"round {round_no:>4}  {ident['trials']:>5} trials  "
                f"correct {100 * ident['correct'] / max(1, ident['trials']):5.1f}%  "
                f"unknown {100 * ident['unknown'] / max(1, ident['trials']):5.1f}%  "
                f"WRONG {100 * ident['wrong'] / max(1, ident['trials']):5.1f}%  "
                f"({record['seconds']}s)")
        print(line)
        append(readable, line)
        if record.get("detection"):
            d = record["detection"]
            extra = (f"           detection {100 * d['rate']:.1f}% over {d['sampled']} "
                     f"FairFace images, disparity {d['disparity']}")
            print(extra)
            append(readable, extra)
        if record.get("age"):
            a = record["age"]
            extra = (f"           age MAE {a['mae']}y, within2 {100 * a['within2']:.0f}%, "
                     f"within10 {100 * a['within10']:.0f}% over {a['sampled']}")
            print(extra)
            append(readable, extra)

        with open(state_file, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

        if round_no % 5 == 0:
            print(summarise(state))
            append(readable, summarise(state))

    print()
    print(summarise(state))
    append(readable, summarise(state))
    append(readable, f"=== stopped {dt.datetime.now().isoformat(timespec='seconds')} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
