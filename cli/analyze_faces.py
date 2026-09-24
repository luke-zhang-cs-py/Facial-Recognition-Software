"""
cli/analyze_faces.py
--------------------
Console version of the trait analysis — the CLI sibling of the Analysis
panel in the web UI.

    python -m cli.analyze_faces              # full report
    python -m cli.analyze_faces --refresh    # ignore the cache, re-read every image
    python -m cli.analyze_faces --quality    # enrollment quality only
    python -m cli.analyze_faces --thresholds # recogniser threshold sweeps only

Prints, per person, how usable their enrolled samples are and which specific
files to recapture; then how separable the enrolled people actually are, and
what confidence threshold your own data supports.
"""

import argparse
import sys

from analysis import analytics
from analysis import calibration
from core import facemodels


# Console layout. RULE_WIDTH was written out as a bare 66 at five call
# sites; BAR_WIDTH is the usable-percentage bar.
RULE_WIDTH = 66
BAR_WIDTH = 22

# The per-image statistics, and the label each gets in the readout. A tuple
# rather than five near-identical print calls, which is what it replaced.
STAT_ROWS = (("sharpness", "sharpness"), ("brightness", "brightness"),
             ("contrast", "contrast"), ("quality", "quality 0-1"),
             ("facePx", "face px"))


def rule(char="="):
    print(char * RULE_WIDTH)


def bar(value, width=BAR_WIDTH, lo=0.0, hi=100.0):
    span = hi - lo or 1.0
    filled = int(round(width * max(0.0, min(1.0, (value - lo) / span))))
    return "#" * filled + "." * (width - filled)


def print_stats(user):
    """The numeric spread for each property this user has a reading for."""
    for key, label in STAT_ROWS:
        stats = user[key]
        if stats:
            print(f"    {label:<12}mean {stats['mean']:<9} "
                  f"min {stats['min']:<9} max {stats['max']}")


def print_pose(user):
    """Yaw spread, or why there isn't one.

    "unavailable" and "0 degrees" mean opposite things -- no landmarks at all
    against every sample taken from the same angle -- so the absent case says
    so rather than printing a zero that reads as a measurement.
    """
    if user["yawSpread"] is not None:
        print(f"    pose        yaw spread {user['yawSpread']} deg, "
              f"range {user['yawRange']}")
    else:
        print("    pose        unavailable "
              "(no landmarks — face did not re-detect)")


def print_demographics(user):
    """The age and gender guesses, each with the agreement behind it.

    The agreement figure is the point: a label that thirty samples disagreed
    about is a different thing from one they all produced, and printing the
    label alone would present the two identically.
    """
    if user["age"]:
        age = user["age"]
        print(f"    age         {age['label']}  "
              f"(agreement {age['agreement']:.0%} across {age['samples']} "
              f"samples, {age['distinctLabels']} distinct labels)")
    if user["gender"]:
        gender = user["gender"]
        print(f"    gender      {gender['label']}  "
              f"(agreement {gender['agreement']:.0%})")


def print_user(user):
    """One person's enrollment-quality block."""
    usable, samples = user["usable"], user["samples"]
    percent = 100.0 * usable / samples if samples else 0.0

    print(f"\n  [{user['userId']}] {user['name']}   {user['verdict'].upper()}")
    print(f"    usable      {usable}/{samples}  {bar(percent)} {percent:.0f}%")

    print_stats(user)
    print_pose(user)

    if user["flags"]:
        counted = ", ".join(f"{k} x{v}" for k, v in user["flags"].items())
        print(f"    flags       {counted}")

    print_demographics(user)

    if user["worstSamples"]:
        print("    recapture these first:")
        for worst in user["worstSamples"]:
            print(f"      {worst['file']:<10}{', '.join(worst['reasons'])}")

    for recommendation in user["recommendations"]:
        print(f"    -> {recommendation}")


def print_quality(report):
    rule()
    print("ENROLLMENT QUALITY")
    rule()
    if not report["users"]:
        print("  No samples found under dataset/ — register someone first.")
        return

    for user in report["users"]:
        print_user(user)


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
    rule()
    print("RECOGNITION ANALYTICS")
    rule()

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

        cal = sface.get("calibration")
        if cal:
            print(f"\n    GALLERY-SIZE CALIBRATION ({cal['corpus']})")
            if sface.get("warning"):
                print(f"    ! {sface['warning']}")
            print(f"    {cal['summary']}")
            if not cal["reachable"]:
                print("    At this scale embeddings alone are not sufficient — "
                      "add a second factor.")
            local = sface.get("localSweepThreshold")
            if local is not None and not sface.get("localSweepTrusted"):
                print(f"    For contrast, this dataset's own sweep would have said "
                      f"{local}, which carries a "
                      f"{100 * cal['riskAtLocalChoice']:.2f}% gallery-wide false-match "
                      f"risk at {cal['gallerySize']} enrolled.")
            print(f"\n    {'gallery':>9}{'threshold':>11}{'risk':>9}")
            for size in (10, 100, 1000, 10000):
                t, r, ok = calibration.recommend_threshold(size)
                mark = "" if ok else "   (unreachable)"
                print(f"    {size:>9}{t:>11.3f}{100 * r:>8.2f}%{mark}")
            d = cal["disparity"]
            print(f"\n    Risk is not evenly shared: at threshold {d['threshold']}, "
                  f"{d['worst'][0]} faces false-match at "
                  f"{100 * d['worst'][1]:.1f}% vs {d['best'][0]} at "
                  f"{100 * d['best'][1]:.1f}% ({d['ratio']}x).")


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
        print("No samples found under dataset/. "
              "Run `python -m cli.register_user \"Name\"` first.")
        return 1

    show_all = not (args.quality or args.thresholds)
    if args.quality or show_all:
        print_quality(report)
    if args.thresholds or show_all:
        print_thresholds(report)

    print()
    rule()
    for note in report["notes"]:
        print(f"  * {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
