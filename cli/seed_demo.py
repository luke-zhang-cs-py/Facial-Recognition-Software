"""
cli/seed_demo.py
----------------
Enroll well-known faces from LFW so recognition can be tested without
registering real people first.

    python -m cli.seed_demo --people 8 --samples 12
    python -m cli.seed_demo --list
    python -m cli.seed_demo --remove          # delete every demo entry

Why LFW: it is the standard academic face-recognition benchmark, assembled
from press photographs of public figures and published for exactly this kind
of evaluation. Using it here is what it is for.

Demo entries are prefixed so they can never be confused with a real person,
and --remove takes them out completely -- database rows and image folders
both. They are enrolled from stills rather than a webcam, which is a fair
test of recognition and a poor one of capture quality; the enrollment report
will rightly complain that they have no pose variety.

Note that a photo held up to the camera is now refused by the liveness check,
which is correct. Test these through the Identify panel (or --test below),
which works on images and does not pretend a photograph is a person.
"""

import argparse
import io
import os
import shutil
import sys

import numpy as np

# The project root -- one level up, because this file lives in cli/.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)   # importable when run as a script from anywhere

from core import corpus_paths              # noqa: E402
import cv2                       # noqa: E402
from core import db                        # noqa: E402
from core import paths                     # noqa: E402
from pipeline import traits                    # noqa: E402
from core import vision                    # noqa: E402

PREFIX = "[demo] "

# CLI defaults. Eight people with twelve samples each is enough for the
# Identify panel to show a real match and a real near-miss without a long
# download.
DEFAULT_PEOPLE = 8
DEFAULT_SAMPLES = 12

# Never enroll fewer than this: below three samples the centroid is barely a
# centroid and the demo demonstrates nothing.
MIN_SAMPLES_KEPT = 3

# Progress reporting. `--all-with 20` enrolls hundreds of people, and one
# line each is not progress, it is a wall.
SMALL_RUN = 20          # at or under this many people, report every one
REPORT_EVERY = 200      # otherwise roughly every this many images

# The fallback here was ".", which does not fail -- it looks for the corpus
# beside whatever directory you happened to run this from and reports it
# absent.
LFW = corpus_paths.lfw_parquet()


def demo_users():
    return [(uid, name) for uid, name in db.get_all_users()
            if name.startswith(PREFIX)]


def remove_all():
    removed = 0
    for uid, name in demo_users():
        db.delete_user(uid)
        for folder in os.listdir(paths.dataset_dir()) if os.path.isdir(paths.dataset_dir()) else []:
            if folder.startswith(f"{uid}_"):
                shutil.rmtree(os.path.join(paths.dataset_dir(), folder), ignore_errors=True)
        print(f"  removed {name}")
        removed += 1
    if not removed:
        print("  no demo entries to remove")
    return removed


def load_lfw(min_images, wanted, requested=None):
    import pyarrow.parquet as pq
    import json
    from collections import defaultdict

    if not os.path.exists(LFW):
        print(f"LFW parquet not found at {LFW}")
        print("Download it with:")
        print("  curl -L https://huggingface.co/api/datasets/logasja/lfw/parquet/"
              "default/train/0.parquet -o \"%s\"" % LFW)
        return None

    pf = pq.ParquetFile(LFW)
    names = None
    meta = pf.schema_arrow.metadata
    if meta and b"huggingface" in meta:
        feats = json.loads(meta[b"huggingface"].decode())["info"]["features"]
        names = feats.get("label", {}).get("names")

    table = pf.read()
    labels = table.column("label").to_pylist()
    images = table.column("image").to_pylist()

    by_person = defaultdict(list)
    for i, lab in enumerate(labels):
        by_person[lab].append(i)

    # Most-photographed first: more images means a more stable centroid, and
    # these are the identities LFW actually supports testing on.
    ranked = sorted(by_person.items(), key=lambda kv: -len(kv[1]))
    if requested:
        # Explicit names take priority and ignore the min-images floor -- if
        # somebody asked for a specific person, enroll whatever exists for them
        # and let the report say the sample count is thin.
        lookup = {n.lower(): i for i, n in enumerate(names)} if names else {}
        picked = []
        for want in requested:
            key = want.strip().lower().replace(" ", "_")
            lab = lookup.get(key)
            if lab is None or lab not in by_person:
                print(f"  !! not in LFW: {want}")
                continue
            picked.append((lab, by_person[lab]))
        return names, images, picked
    if wanted is None:
        # everyone deep enough, not just the top N
        picked = [(lab, idx) for lab, idx in ranked if len(idx) >= min_images]
        return names, images, picked
    picked = [(lab, idx) for lab, idx in ranked if len(idx) >= min_images][:wanted]
    return names, images, picked


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=DEFAULT_PEOPLE)
    ap.add_argument("--names", default="",
                    help="comma-separated LFW names to enroll explicitly, "
                         "e.g. LeBron_James,Yao_Ming")
    ap.add_argument("--all-with", type=int, default=0,
                    help="enroll every LFW identity having at least this many "
                         "images (0 = off). --people is ignored.")
    ap.add_argument("--keep", action="store_true",
                    help="add to the existing demo entries instead of replacing")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--remove", action="store_true")
    return ap


def sample_budget(available, wanted):
    """How many of one person's images to enroll.

    One is held back when the person has few to begin with, so there is
    always something left to identify against -- a demo where every image is
    also a training image cannot demonstrate recognition. MIN_SAMPLES_KEPT is
    the floor, below which the enrollment is too thin to be worth making.
    """
    if available >= wanted:
        return wanted
    return min(wanted, max(MIN_SAMPLES_KEPT, available - 1))


def should_report(index, people, total, kept):
    """Whether to print a progress line for the person just enrolled.

    Every person on a small run; on a large one, the last person and roughly
    every REPORT_EVERY images.

    The condition this replaced read

        len(picked) <= 20 or (len(picked) - 0) and (idxs is picked[-1][1] ...)

    where `(len(picked) - 0)` is a subtraction of zero standing in for a
    truthiness test, and `idxs is picked[-1][1]` asks "is this the last one"
    by object identity on a list -- correct only because no two people
    happened to share a list object. The index says it directly.
    """
    if people <= SMALL_RUN:
        return True
    return index == people - 1 or total % REPORT_EVERY < kept


def enroll_person(folder, idxs, images, budget):
    """Write up to `budget` cropped faces into `folder`; return how many.

    Images with no detectable face, or a box that crops to nothing, are
    skipped rather than counted -- LFW is press photography and some of it
    defeats the detector.
    """
    from PIL import Image

    kept = 0
    for i in idxs:
        if kept >= budget:
            break
        raw = images[i]
        data = raw["bytes"] if isinstance(raw, dict) else raw
        bgr = cv2.cvtColor(
            np.array(Image.open(io.BytesIO(data)).convert("RGB")),
            cv2.COLOR_RGB2BGR)
        rows = traits.detect(bgr)
        if not rows:
            continue
        x, y, w, h = traits.geometry(rows[0])["box"]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        crop = gray[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            continue
        kept += 1
        cv2.imwrite(os.path.join(folder, f"{kept}.jpg"),
                    cv2.resize(crop, vision.LBPH_INPUT_SIZE))
    return kept


def list_demo_users():
    users = demo_users()
    plural = "y" if len(users) == 1 else "ies"
    print(f"{len(users)} demo entr{plural}:")
    for user_id, name in users:
        print(f"  [{user_id}] {name}")


def enroll_all(picked, names, images, wanted):
    """Enroll every picked identity; return how many images were written."""
    os.makedirs(paths.dataset_dir(), exist_ok=True)
    print(f"enrolling {len(picked)} people, up to {wanted} samples each")

    total = 0
    for index, (label, idxs) in enumerate(picked):
        person = (names[label] if names else str(label)).replace("_", " ")
        display = PREFIX + person
        user_id = db.add_user(display)
        folder = os.path.join(paths.dataset_dir(),
                              f"{user_id}_{person.replace(' ', '_')}")
        os.makedirs(folder, exist_ok=True)

        kept = enroll_person(folder, idxs, images,
                             sample_budget(len(idxs), wanted))
        total += kept
        if should_report(index, len(picked), total, kept):
            print(f"  {display:<34} {kept} samples   "
                  f"({total} images so far)", flush=True)
    return total


def main():
    args = build_parser().parse_args()
    db.init_db()

    if args.list:
        list_demo_users()
        return 0

    if args.remove:
        remove_all()
        return 0

    if not args.keep:
        remove_all()   # re-seeding replaces rather than duplicates

    if args.all_with:
        loaded = load_lfw(args.all_with, None, None)
    else:
        loaded = load_lfw(args.samples, args.people,
                          [n for n in args.names.split(",") if n.strip()])
    if loaded is None:
        return 1

    names, images, picked = loaded
    total = enroll_all(picked, names, images, args.samples)

    print(f"\n{total} images written. Analysing so embeddings exist...")
    from analysis import analytics
    analytics.scan(progress=None, use_cache=False)

    from pipeline import train_model
    train_model.train()
    print("Done. Try the Identify panel, or: python -m cli.seed_demo --list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
