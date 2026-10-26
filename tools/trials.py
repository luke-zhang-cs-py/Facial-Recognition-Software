"""100 randomised trials per enrolled person.

One clean image says whether recognition works in the best case. It says
nothing about how it behaves at 3pm when somebody walks past at an angle in
bad light. Each trial randomises distance, motion blur, rotation, gamma,
sensor noise and JPEG quality together, so the run covers the space rather
than one point in it.

Three outcomes, and the distinction matters:
  CORRECT  named the right person
  UNKNOWN  declined -- either under threshold or too close between two people
  WRONG    named somebody else. The only genuinely dangerous one.
"""
import io
import os
import sys
import json
import time
import collections
import numpy as np
import cv2

# The project root -- one level up, because this file lives in tools/.
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from core import corpus_paths   # noqa: E402
from pipeline import recognition    # noqa: E402

DEFAULT_TRIALS = 100


def load_lfw(path):
    """(label names, per-image labels, per-image cells) from the LFW shard.

    pyarrow is imported here rather than at the top, so the module imports
    without the corpus readers installed and the absence of the corpus is the
    message rather than an ImportError.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = json.loads(pf.schema_arrow.metadata[b'huggingface'].decode())["info"]["features"]["label"]["names"]
    tab = pf.read()
    return names, tab.column("label").to_pylist(), tab.column("image").to_pylist()


def load(raw):
    """One parquet image cell -> BGR."""
    from PIL import Image
    d = raw["bytes"] if isinstance(raw, dict) else raw
    return cv2.cvtColor(np.array(Image.open(io.BytesIO(d)).convert("RGB")), cv2.COLOR_RGB2BGR)


def degrade(img, rng):
    """One random point in the operating space."""
    h, w = img.shape[:2]
    f = rng.uniform(0.25, 1.0)
    s = cv2.resize(img, (max(16, int(w * f)), max(16, int(h * f))), interpolation=cv2.INTER_AREA)
    img = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    k = int(rng.choice([1, 1, 3, 3, 5, 7, 9]))
    if k >= 3:
        kern = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            kern[k // 2, :] = 1.0 / k
        else:
            kern[:, k // 2] = 1.0 / k
        img = cv2.filter2D(img, -1, kern)
    deg = rng.uniform(-18, 18)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    g = rng.uniform(0.45, 2.0)
    lut = np.array([((i / 255.0)**(1.0 / g)) * 255 for i in range(256)], np.uint8)
    img = cv2.LUT(img, lut)
    if rng.random() < 0.5:
        img = np.clip(img.astype(np.int16) + rng.normal(0, rng.uniform(2, 10), img.shape), 0, 255).astype(np.uint8)
    q = int(rng.integers(18, 96))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def heldout(idxs, n_enrolled):
    """The images a person was not enrolled from (seed_demo takes the first N
    that detect), or their last one if every image was used."""
    tail = idxs[n_enrolled:]
    return tail if tail else idxs[-1:]


def score(person, pool, trials, gal, rng):
    """(correct, unknown, wrong, similarities) over `trials` degraded draws
    from `pool`, a person's held-out image cells, taken in turn."""
    c = u = w = 0
    sims = []
    for t in range(trials):
        img = degrade(load(pool[t % len(pool)]), rng)
        r = recognition.identify(img, gal)
        if not r["ok"]:
            u += 1
            continue
        if r["match"] is None:
            u += 1
        elif r["match"]["name"].replace("[demo] ", "") == person:
            c += 1
            sims.append(r["best"]["similarity"])
        else:
            w += 1
    return c, u, w, sims


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    trials = int(argv[0]) if argv else DEFAULT_TRIALS
    if trials < 1:
        # Every per-person line divides by the trial count.
        sys.exit("The trial count must be at least 1.")

    # This read os.environ["TEMP"], which is unset outside Windows, so the script
    # died with a KeyError here rather than saying which corpus was missing.
    # corpus_paths uses tempfile.gettempdir(), which is defined everywhere.
    lfw = corpus_paths.lfw_parquet()
    if not os.path.exists(lfw):
        sys.exit(f"No LFW corpus at {lfw}\n"
                 "Set FACE_CORPORA if it is cached somewhere else.")
    names, labels, images = load_lfw(lfw)
    by = collections.defaultdict(list)
    for i, l in enumerate(labels):
        by[l].append(i)

    gal = recognition.gallery()
    enrolled = {v["name"].replace("[demo] ", ""): uid for uid, v in gal.items()
                if v["name"].startswith("[demo] ")}
    print(f"gallery: {len(gal)} enrolled, {len(enrolled)} demo identities")
    print(f"{trials} randomised trials each\n")

    rng = np.random.default_rng(7)

    gal_samples = {v["name"].replace("[demo] ", ""): v["samples"] for v in gal.values()}
    name_to_lab = {n.replace("_", " "): i for i, n in enumerate(names)}

    rows = []
    t0 = time.time()
    print(f"  {'person':<26}{'n':>4}{'correct':>9}{'unknown':>9}{'WRONG':>7}{'mean sim':>10}")
    print("  " + "-" * 66)
    tot = collections.Counter()
    for person in sorted(enrolled):
        lab = name_to_lab.get(person)
        if lab is None:
            continue
        pool = heldout(by[lab], gal_samples.get(person, 3))
        c, u, w, sims = score(person, [images[i] for i in pool], trials, gal,
                              rng)
        n = c + u + w
        tot["c"] += c
        tot["u"] += u
        tot["w"] += w
        tot["n"] += n
        rows.append((person, c, u, w))
        print(f"  {person:<26}{n:>4}{100 * c / n:>8.0f}%{100 * u / n:>8.0f}%{100 * w / n:>6.0f}%"
              f"{(np.mean(sims) if sims else 0):>10.3f}")

    n = tot["n"]
    if not n:
        # Nobody enrolled is in this corpus (or nobody is enrolled at all).
        # This used to divide by zero on the line below.
        sys.exit("No enrolled [demo] identity is in the LFW corpus. "
                 "Run `python -m cli.seed_demo` first.")
    print("  " + "-" * 66)
    print(f"  {'ALL':<26}{n:>4}{100 * tot['c'] / n:>8.1f}%{100 * tot['u'] / n:>8.1f}%{100 * tot['w'] / n:>6.1f}%")
    print(f"\n  correct   {tot['c']}")
    print(f"  unknown   {tot['u']}   (declined: under threshold or too close to call)")
    print(f"  WRONG     {tot['w']}   (named the wrong person)")
    print(f"  {(time.time() - t0) / 60:.1f} min")
    worst = sorted(rows, key=lambda r: -r[3])[:5]
    print("\n  most misidentifications:")
    for p, c, u, w in worst:
        print(f"    {p:<26} {w} wrong / {trials}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
