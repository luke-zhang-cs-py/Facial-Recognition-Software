"""How does identification hold up as the gallery grows?

calibration.py predicts, from 4.77e9 FairFace impostor pairs, that past about
10,000 enrolled people no measured threshold keeps the false-match risk under
1%. That was extrapolated arithmetic. This measures it directly on a gallery
built from CASIA-WebFace identities.

Protocol per gallery size N:
  - take N identities, build each centroid from 3 embeddings
  - probe with the 4th, held-out embedding of each
  - a probe is CORRECT if its own identity is the nearest AND clears threshold,
    REJECTED if nothing clears it, MISIDENTIFIED if somebody else wins
  - separately measure the rank-1 rate ignoring the threshold, which is the
    number usually quoted as "accuracy" and hides the false-match problem

Run tools_build_gallery.py first; this reads what that writes, and both agree
on the location through corpus_paths so neither hardcodes it.

    python tools_scale_test.py
    CASIA_DIR=/data/casia python tools_scale_test.py
"""
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibration                                   # noqa: E402
from corpus_paths import gallery_path                # noqa: E402


def load_gallery():
    """The embeddings, or a message saying how to build them.

    The corpus is several GB and is not in the repository, so its absence is
    the ordinary state of a fresh clone rather than something worth a
    traceback.
    """
    path = gallery_path()
    if not os.path.exists(path):
        sys.exit(f"No gallery at {path}\n"
                 f"Build one first:  python tools_build_gallery.py\n"
                 f"Or point CASIA_DIR at an existing one.")
    return np.load(path, allow_pickle=True)


G = load_gallery()
labels, vecs = G["labels"], G["vecs"]
print(f"loaded {len(vecs)} embeddings, {len(set(labels.tolist()))} identities\n")

by = collections.defaultdict(list)
for lab, v in zip(labels.tolist(), vecs):
    by[lab].append(v)
usable = {k: v for k, v in by.items() if len(v) >= 4}
print(f"{len(usable)} identities have the 4 embeddings this needs\n")

ids = sorted(usable)
rng = np.random.default_rng(0)
rng.shuffle(ids)

SIZES = [100, 250, 500, 1000, 2500, 5000, 7500, 10000]
SIZES = [n for n in SIZES if n <= len(ids)]
if len(ids) not in SIZES:
    SIZES.append(len(ids))

print(f"{'gallery':>8}{'thresh':>8}{'rank1':>8}{'correct':>9}{'rejected':>10}"
      f"{'MISID':>8}{'predicted risk':>16}")
print("-" * 70)
for n in SIZES:
    sub = ids[:n]
    cents, probes = [], []
    for i in sub:
        e = usable[i]
        c = np.mean(np.vstack(e[:3]), axis=0)
        c = c / (np.linalg.norm(c) or 1.0)
        cents.append(c)
        p = e[3]; probes.append(p / (np.linalg.norm(p) or 1.0))
    C = np.vstack(cents); P = np.vstack(probes)

    thr, risk, reachable = calibration.recommend_threshold(n)

    rank1 = correct = rejected = misid = 0
    B = 512
    for s in range(0, n, B):
        S = P[s:s+B] @ C.T
        idx = S.argmax(axis=1)
        best = S.max(axis=1)
        truth = np.arange(s, min(s+B, n))
        rank1 += int((idx == truth).sum())
        above = best >= thr
        correct += int(((idx == truth) & above).sum())
        misid += int(((idx != truth) & above).sum())
        rejected += int((~above).sum())

    mark = "" if reachable else "  UNREACHABLE"
    print(f"{n:>8}{thr:>8.3f}{100*rank1/n:>7.1f}%{100*correct/n:>8.1f}%"
          f"{100*rejected/n:>9.1f}%{100*misid/n:>7.1f}%{100*risk:>14.2f}%{mark}")

print("\nrank1 ignores the threshold: it asks only 'was the right person nearest',")
print("which stays high while the usable accuracy falls away underneath it.")
