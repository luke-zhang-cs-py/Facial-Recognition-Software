"""
analyze_faces.py
-----------------
Console version of the trait analysis — the CLI sibling of the Analysis
panel in the web UI.

    python analyze_faces.py              # full report
    python analyze_faces.py --refresh    # ignore the cache, re-read every image
    python analyze_faces.py --quality    # enrollment quality only
    python analyze_faces.py --thresholds # recogniser threshold sweeps only

Prints, per person, how usable their enrolled samples are and which specific
files to recapture; then how separable the enrolled people actually are, and
what confidence threshold your own data supports.
"""

import argparse
import sys

import analytics
import facemodels


def bar(value, width=22, lo=0.0, hi=100.0):
    span = hi - lo or 1.0
    filled = int(round(width * max(0.0, min(1.0, (value - lo) / span))))
    return "#" * filled + "." * (width - filled)


def print_quality(report):
    print("=" * 66)
    print("ENROLLMENT QUALITY")
    print("=" * 66)
    if not report["users"]:
        print("  No samples found under dataset/ — register someone first.")
        return

    for u in report["users"]:
        pct = 100.0 * u["usable"] / u["samples"] if u["samples"] else 0.0
        print(f"\n  [{u['userId']}] {u['name']}   {u['verdict'].upper()}")
        print(f"    usable      {u['usable']}/{u['samples']}  {bar(pct)} {pct:.0f}%")

        for key, label in (("sharpness", "sharpness"), ("brightness", "brightness"),
                           ("contrast", "contrast"), ("quality", "quality 0-1"),
                           ("facePx", "face px")):
            s = u[key]
            if s:
                print(f"    {label:<12}mean {s['mean']:<9} min {s['min']:<9} max {s['max']}")

        if u["yawSpread"] is not None:
            print(f"    pose        yaw spread {u['yawSpread']} deg, range {u['yawRange']}")
        else:
            print("    pose        unavailable (no landmarks — face did not re-detect)")

        if u["flags"]:
            print(f"    flags       {', '.join(f'{k} x{v}' for k, v in u['flags'].items())}")

        if u["age"]:
            a = u["age"]
            print(f"    age         {a['label']}  (agreement {a['agreement']:.0%} across "
                  f"{a['samples']} samples, {a['distinctLabels']} distinct labels)")
        if u["gender"]:
            g = u["gender"]
            print(f"    gender      {g['label']}  (agreement {g['agreement']:.0%})")

        if u["worstSamples"]:
            print("    recapture these first:")
            for w in u["worstSamples"]:
                print(f"      {w['file']:<10}{', '.join(w['reasons'])}")

        for rec in u["recommendations"]:
            print(f"    -> {rec}")


def print_sweep(title, block, unit=""):
    print(f"\n  {title}")
    if not block["available"]:
        print(f"    unavailable: {block['reason']}")
        return

    print(f"    protocol    {block['protocol']}")
    print(f"    accuracy    {block['accuracy']}%  over {block['samples']} held-out predictions")
    print(f"    {'thresh':>9}{'accept%':>10}{'falseMatch%':>13}")
    for row in block["sweep"]:
        marks = []
        if row["threshold"] == block["recommendedThreshold"]:
            marks.append("recommended")
        if row["threshold"] == block.get("currentThreshold"):
            marks.append("current default")
        suffix = ("   <- " + ", ".join(marks)) if marks else ""
        print(f"    {row['threshold']:>9}{row['accept']:>10}{row['falseMatch']:>13}{suffix}")

    print(f"    => use {block['recommendedThreshold']}{unit}: "
          f"{block['recommendedAccept']}% accepted, "
          f"{block['recommendedFalseMatch']}% false matches")


def print_thresholds(report):
    print()
    print("=" * 66)
    print("RECOGNITION ANALYTICS")
    print("=" * 66)

    lbph = report["lbph"]
    print_sweep("LBPH — the recogniser attendance.py uses today", lbph)
    if lbph["available"] and lbph["recommendedThreshold"] != lbph["currentThreshold"]:
        print(f"    NOTE: attendance.py has CONFIDENCE_THRESHOLD = "
              f"{lbph['currentThreshold']}; your data supports "
              f"{lbph['recommendedThreshold']}.")

    sface = report["sface"]
    print_sweep("SFace — 128-d embeddings (cosine similarity)", sface)
    if sface["available"]:
        print(f"    genuine mean {sface['genuine']['mean']} vs impostor mean "
              f"{sface['impostor']['mean']}  (margin {sface['margin']})")
        print(f"    SFace's own reference operating point is "
              f"{sface['referenceThreshold']}")
        if sface["weakestPairs"]:
            w = sface["weakestPairs"][0]
            print(f"    most confusable pair: user {w['a']} vs user {w['b']} "
                  f"(peak similarity {w['maxSimilarity']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--quality", action="store_true", help="enrollment quality only")
    ap.add_argument("--thresholds", action="store_true", help="threshold sweeps only")
    args = ap.parse_args()

    missing = facemodels.missing_summary()
    if missing:
        print(f"NOTE: {missing}\n")

    def progress(done, total):
        print(f"\r  analysing {done}/{total} ...", end="", flush=True)

    report = analytics.scan(progress=progress, use_cache=not args.refresh)
    print("\r" + " " * 40 + "\r", end="")

    if report["totalSamples"] == 0:
        print("No samples found under dataset/. Run register_user.py first.")
        return 1

    show_all = not (args.quality or args.thresholds)
    if args.quality or show_all:
        print_quality(report)
    if args.thresholds or show_all:
        print_thresholds(report)

    print()
    print("=" * 66)
    for note in report["notes"]:
        print(f"  * {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
