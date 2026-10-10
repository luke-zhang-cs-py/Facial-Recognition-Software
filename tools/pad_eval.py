"""APCER and BPCER from a CSV of presentation-attack trial results.

    python tools/pad_eval.py attempts.csv
    python tools/pad_eval.py attempts.csv --json

The protocol is in notes/BENCHMARK.md ("Liveness against presentation
attacks"). Each row is one attempt in front of the camera, already decided:

    attack_type,ground_truth,verdict
    none,bona_fide,live
    print,attack,spoof
    replay,attack,live

* attack_type   the presentation attack instrument (PAI) species: `print`,
                `replay`, ... -- anything; `none` (or blank) for a real person.
* ground_truth  `bona_fide` or `attack`.
* verdict       what the liveness vote said: `live`, `spoof`, or `unknown`
                when it never reached a decision.

Other columns (a participant code, a lighting note) are allowed and ignored.
The CSV holds outcomes only. Never put a frame, a face crop, an embedding or a
name in it: the recordings stay with the people who consented to them, off
this repository (see .gitignore).

The metrics are ISO/IEC 30107-3's:

* APCER, per PAI species: the share of that species' attack presentations
  classified as bona fide. Reported per species and as the worst species,
  because the standard reports the attack the system handles worst, not an
  average that a strong species can hide a weak one behind.
* BPCER: the share of bona fide presentations classified as attacks.

`unknown` is neither accepted nor marked present, so it counts as "classified
as an attack" on both sides: harmless for APCER, an error for BPCER. That is
the conservative reading for an attendance system, where a real person the
vote never passes is a person not marked. The count is reported separately so
it is visible rather than folded in.

Each rate carries a Wilson 95% interval: these trials are tens of attempts,
not thousands, and a rate of 0/20 is not a rate of zero.
"""
import argparse
import csv
import io
import json
import math
import sys

BONA_FIDE, ATTACK = "bona_fide", "attack"
LIVE, SPOOF, UNKNOWN = "live", "spoof", "unknown"
NO_ATTACK = ("", "none", BONA_FIDE)

# z for a two-sided 95% interval.
Z95 = 1.959964


def wilson(errors, n, z=Z95):
    """(low, high) Wilson score interval for errors/n, or (None, None)."""
    if n == 0:
        return None, None
    p = errors / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def read_rows(handle):
    """Validated (attack_type, ground_truth, verdict) tuples from a CSV."""
    reader = csv.DictReader(handle)
    wanted = {"attack_type", "ground_truth", "verdict"}
    missing = wanted - set(reader.fieldnames or ())
    if missing:
        raise ValueError("the CSV has no %s column" % ", ".join(sorted(missing)))
    rows = []
    for number, row in enumerate(reader, start=2):      # line 1 is the header
        kind = (row["attack_type"] or "").strip().lower()
        truth = (row["ground_truth"] or "").strip().lower()
        verdict = (row["verdict"] or "").strip().lower()
        if truth not in (BONA_FIDE, ATTACK):
            raise ValueError("line %d: ground_truth must be bona_fide or "
                             "attack, not %r" % (number, row["ground_truth"]))
        if verdict not in (LIVE, SPOOF, UNKNOWN):
            raise ValueError("line %d: verdict must be live, spoof or unknown, "
                             "not %r" % (number, row["verdict"]))
        if truth == BONA_FIDE and kind not in NO_ATTACK:
            raise ValueError("line %d: a bona fide attempt has attack_type %r"
                             % (number, row["attack_type"]))
        if truth == ATTACK and kind in NO_ATTACK:
            raise ValueError("line %d: an attack needs its attack_type "
                             "(print, replay, ...)" % number)
        rows.append((kind if truth == ATTACK else "none", truth, verdict))
    return rows


def _rate(errors, n, unknown):
    low, high = wilson(errors, n)
    return {"n": n, "errors": errors, "unknown": unknown,
            "rate": errors / n if n else None, "ci95": [low, high]}


def evaluate(rows):
    """APCER per species and worst-case, and BPCER, from read_rows() output."""
    species = {}
    for kind, truth, verdict in rows:
        if truth == ATTACK:
            seen = species.setdefault(kind, [0, 0, 0])
            seen[0] += 1
            seen[1] += verdict == LIVE
            seen[2] += verdict == UNKNOWN
    apcer = {kind: _rate(errors, n, unknown)
             for kind, (n, errors, unknown) in sorted(species.items())}
    worst = max(apcer, key=lambda k: apcer[k]["rate"]) if apcer else None

    bona = [verdict for _, truth, verdict in rows if truth == BONA_FIDE]
    bpcer = _rate(sum(v != LIVE for v in bona), len(bona),
                  sum(v == UNKNOWN for v in bona))
    return {
        "apcer": apcer,
        "apcerWorst": ({"species": worst, **apcer[worst]} if worst else None),
        "bpcer": bpcer,
        "attempts": len(rows),
    }


def _pct(value):
    return "  n/a" if value is None else "%5.1f%%" % (100 * value)


def report(result):
    """The result as a short table for a terminal."""
    lines = ["%-18s %5s %7s %8s  %s" % ("", "n", "errors", "rate", "95% CI")]

    def row(label, body):
        low, high = body["ci95"]
        ci = ("%s - %s" % (_pct(low).strip(), _pct(high).strip())
              if low is not None else "n/a")
        lines.append("%-18s %5d %7d %8s  %s" % (label, body["n"],
                                                body["errors"],
                                                _pct(body["rate"]), ci))

    for kind, body in result["apcer"].items():
        row("APCER " + kind, body)
    if result["apcerWorst"]:
        row("APCER (worst)", result["apcerWorst"])
    row("BPCER", result["bpcer"])
    unknown = result["bpcer"]["unknown"]
    if unknown:
        lines.append("%d bona fide attempt(s) never reached a verdict and are "
                     "counted as rejected." % unknown)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="attempt results: attack_type,ground_truth,verdict")
    ap.add_argument("--json", action="store_true", help="print JSON instead")
    args = ap.parse_args(argv)
    with io.open(args.csv, newline="", encoding="utf-8") as handle:
        try:
            result = evaluate(read_rows(handle))
        except ValueError as exc:
            print("pad_eval: %s" % exc, file=sys.stderr)
            return 2
    print(json.dumps(result, indent=2) if args.json else report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
