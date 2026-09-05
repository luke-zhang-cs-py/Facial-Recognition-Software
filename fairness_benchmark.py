"""
fairness_benchmark.py
----------------------
Measure whether the quality gate treats demographic groups equally, using
stratified sampling against a labelled external corpus.

    python fairness_benchmark.py --corpus <dir-of-parquet> --per-group 500
    python fairness_benchmark.py --corpus <dir> --per-group 500 --json out.json

Why stratified rather than random
---------------------------------
A random sample reproduces the corpus's own imbalance. If a corpus is 70%
one group, a random draw gives that group 70% of the statistical power and
leaves the others with wide error bars -- exactly the groups whose treatment
you are trying to measure. Drawing an equal number per group gives every
group the same precision, so a gap either clears the confidence intervals or
it does not.

The number that matters is the disparity ratio: the worst group's flag rate
divided by the best group's. A gate that measures photographs should land
near 1.0x. A gate that measures people will not.

Corpus format
-------------
Parquet files with an `image` column (struct with `bytes`) and one or more
integer label columns. Built for FairFace's layout (`race`, `gender`, `age`)
but nothing here is FairFace-specific -- point `--group-column` at whatever
label you want to slice by.
"""

import argparse
import glob
import io
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# FairFace's class orders, used when the corpus does not carry names.
DEFAULT_NAMES = {
    "race": ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern',
             'Latino_Hispanic', 'Southeast Asian'],
    "gender": ['Male', 'Female'],
    "age": ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59',
            '60-69', '70+'],
}

# A gate is called biased above this. 1.25x is a judgement call, not a law:
# it is roughly where a gap stops being explicable by sampling noise at a few
# hundred per group and starts being visible to the people affected.
DISPARITY_BUDGET = 1.25


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_labels(files, columns):
    import pyarrow.parquet as pq
    cols = {c: [] for c in columns}
    offsets, start = [], 0
    for path in files:
        t = pq.read_table(path, columns=columns)
        for c in columns:
            cols[c].append(np.array(t.column(c).to_pylist(), np.int16))
        offsets.append((path, start, start + t.num_rows))
        start += t.num_rows
    return {c: np.concatenate(v) for c, v in cols.items()}, offsets, start


def stratified_indices(groups, per_group, seed=0):
    """Equal draw per group, so every group gets the same statistical power."""
    rng = np.random.default_rng(seed)
    out = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        out.append(rng.choice(idx, min(per_group, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


def analyse_sample(indices, offsets):
    """Run traits.analyze over the sampled images; return per-image flags."""
    import cv2
    import pyarrow.parquet as pq
    from PIL import Image
    import traits

    results = [None] * len(indices)
    position = {gi: k for k, gi in enumerate(indices)}
    done = 0

    for path, f_start, f_end in offsets:
        want = indices[(indices >= f_start) & (indices < f_end)]
        if not len(want):
            continue
        col = pq.read_table(path, columns=["image"]).column("image")
        for gi in want:
            try:
                raw = col[gi - f_start].as_py()["bytes"]
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                t = traits.analyze(bgr, want_embedding=False,
                                   want_demographics=False,
                                   require_detection=True)
                results[position[gi]] = t
            except Exception:
                pass
            done += 1
            if done % 250 == 0:
                print(f"    {done}/{len(indices)}", flush=True)
    return results


def report_rate(name, hits, groups, names, budget=DISPARITY_BUDGET):
    print(f"\n  {name}")
    print(f"    {'group':<18}{'n':>7}{'rate':>9}{'95% CI':>18}")
    rows = []
    for g in sorted(np.unique(groups)):
        sel = groups == g
        n = int(sel.sum())
        if not n:
            continue
        k = int(hits[sel].sum())
        lo, hi = wilson(k, n)
        label = names[g] if g < len(names) else f"group {g}"
        rows.append({"group": label, "n": n, "rate": k / n, "ci": [lo, hi]})
        print(f"    {label:<18}{n:>7}{100*k/n:>8.2f}%"
              f"{f'[{100*lo:.2f}, {100*hi:.2f}]':>18}")

    if len(rows) < 2:
        return {"rows": rows, "disparity": None}

    worst = max(rows, key=lambda r: r["rate"])
    best = min(rows, key=lambda r: r["rate"])
    ratio = (worst["rate"] / best["rate"]) if best["rate"] > 0 else (
        float("inf") if worst["rate"] > 0 else 1.0)

    # A ratio alone is not evidence. On a rare flag, 0.2% vs 0.0% is an
    # infinite ratio and pure noise -- two events either side of a coin toss.
    # Calling bias requires both an effect big enough to matter (over budget)
    # and confidence intervals that do not overlap, so the gap survives the
    # sampling error. Otherwise the honest label is "inconclusive".
    separated = worst["ci"][0] > best["ci"][1]
    if ratio <= budget:
        verdict = "OK"
    elif not separated:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "BIASED"

    shown = "inf" if ratio == float("inf") else f"{ratio:.2f}x"
    note = "" if separated or verdict == "OK" else "  (CIs overlap)"
    print(f"    disparity {shown}   ({worst['group']} {100*worst['rate']:.2f}% "
          f"vs {best['group']} {100*best['rate']:.2f}%)   [{verdict}]{note}")
    return {"rows": rows, "disparity": ratio, "worst": worst,
            "best": best, "verdict": verdict, "ciSeparated": separated}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="directory containing .parquet files")
    ap.add_argument("--per-group", type=int, default=400)
    ap.add_argument("--group-column", default="race")
    ap.add_argument("--extra-columns", default="gender",
                    help="comma-separated additional label columns to slice by")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.corpus, "*.parquet")))
    if not files:
        print(f"no parquet files under {args.corpus}")
        return 1

    extra = [c for c in args.extra_columns.split(",") if c]
    columns = [args.group_column] + extra
    labels, offsets, total = load_labels(files, columns)
    groups = labels[args.group_column]

    idx = stratified_indices(groups, args.per_group, args.seed)
    print(f"corpus: {total:,} images in {len(files)} file(s)")
    print(f"stratified sample: {len(idx):,} images, up to {args.per_group} "
          f"per '{args.group_column}' group")

    results = analyse_sample(idx, offsets)
    got = [r for r in results if r is not None]
    print(f"analysed {len(got):,}/{len(idx):,}")

    sel_groups = groups[idx]
    detected = np.array([bool(r and r["detected"]) for r in results])
    any_flag = np.array([bool(r and r["flags"]) for r in results])

    names = DEFAULT_NAMES.get(args.group_column,
                              [f"group {i}" for i in range(int(groups.max()) + 1)])

    print("\n" + "=" * 70)
    print(f"FAIRNESS BY {args.group_column.upper()}  (disparity budget {DISPARITY_BUDGET}x)")
    print("=" * 70)

    report = {"corpus": args.corpus, "sample": len(idx),
              "perGroup": args.per_group, "sections": {}}

    report["sections"]["detection_failure"] = report_rate(
        "detection FAILURE rate (lower is better, must be even)",
        ~detected, sel_groups, names)
    report["sections"]["flagged"] = report_rate(
        "share of images the quality gate rejects", any_flag, sel_groups, names)

    # Per-flag breakdown: which specific check, if any, is uneven.
    all_flags = sorted({f for r in got for f in r["flags"]})
    for flag in all_flags:
        hits = np.array([bool(r and flag in r["flags"]) for r in results])
        if hits.sum() < 20:
            continue
        report["sections"][f"flag_{flag}"] = report_rate(
            f"flag '{flag}'", hits, sel_groups, names)

    for col in extra:
        vals = labels[col][idx]
        nm = DEFAULT_NAMES.get(col, [f"{col} {i}" for i in range(int(vals.max()) + 1)])
        print("\n" + "=" * 70)
        print(f"FAIRNESS BY {col.upper()}")
        print("=" * 70)
        report["sections"][f"{col}_flagged"] = report_rate(
            "share of images the quality gate rejects", any_flag, vals, nm)

    biased = [k for k, v in report["sections"].items()
              if isinstance(v, dict) and v.get("verdict") == "BIASED"]
    print("\n" + "=" * 70)
    if biased:
        print(f"RESULT: {len(biased)} check(s) over the {DISPARITY_BUDGET}x budget:")
        for k in biased:
            print(f"  - {k}: {report['sections'][k]['disparity']:.2f}x")
    else:
        print(f"RESULT: every check within the {DISPARITY_BUDGET}x disparity budget.")
    print("=" * 70)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"\nwrote {args.json}")
    return 1 if biased else 0


if __name__ == "__main__":
    sys.exit(main())
