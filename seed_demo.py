"""
seed_demo.py
-------------
Enroll well-known faces from LFW so recognition can be tested without
registering real people first.

    python seed_demo.py --people 8 --samples 12
    python seed_demo.py --list
    python seed_demo.py --remove          # delete every demo entry

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import cv2                       # noqa: E402
import db                        # noqa: E402
import traits                    # noqa: E402

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PREFIX = "[demo] "
LFW = os.path.join(os.environ.get("TEMP", "."), "lfw", "lfw.parquet")


def demo_users():
    return [(uid, name) for uid, name in db.get_all_users()
            if name.startswith(PREFIX)]


def remove_all():
    removed = 0
    for uid, name in demo_users():
        db.delete_user(uid)
        for folder in os.listdir(DATASET_DIR) if os.path.isdir(DATASET_DIR) else []:
            if folder.startswith(f"{uid}_"):
                shutil.rmtree(os.path.join(DATASET_DIR, folder), ignore_errors=True)
        print(f"  removed {name}")
        removed += 1
    if not removed:
        print("  no demo entries to remove")
    return removed


def load_lfw(min_images, wanted, requested=None):
    import pyarrow.parquet as pq
    import json
    from PIL import Image
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
    picked = [(lab, idx) for lab, idx in ranked if len(idx) >= min_images][:wanted]
    return names, images, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=8)
    ap.add_argument("--names", default="",
                    help="comma-separated LFW names to enroll explicitly, "
                         "e.g. LeBron_James,Yao_Ming")
    ap.add_argument("--keep", action="store_true",
                    help="add to the existing demo entries instead of replacing")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()

    db.init_db()

    if args.list:
        users = demo_users()
        print(f"{len(users)} demo entr{'y' if len(users)==1 else 'ies'}:")
        for uid, name in users:
            print(f"  [{uid}] {name}")
        return 0

    if args.remove:
        remove_all()
        return 0

    if not args.keep:
        remove_all()   # re-seeding replaces rather than duplicates

    loaded = load_lfw(args.samples, args.people,
                      [n for n in args.names.split(",") if n.strip()])
    if loaded is None:
        return 1
    names, images, picked = loaded

    from PIL import Image
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"enrolling {len(picked)} people, up to {args.samples} samples each")

    total = 0
    for label, idxs in picked:
        person = (names[label] if names else str(label)).replace("_", " ")
        display = PREFIX + person
        user_id = db.add_user(display)
        folder = os.path.join(DATASET_DIR, f"{user_id}_{person.replace(' ', '_')}")
        os.makedirs(folder, exist_ok=True)

        # Leave one image out for testing when the person has few to begin
        # with, so there is always something held back to identify against.
        budget = min(args.samples, max(3, len(idxs) - 1)) if len(idxs) < args.samples             else args.samples
        kept = 0
        for i in idxs:
            if kept >= budget:
                break
            raw = images[i]
            data = raw["bytes"] if isinstance(raw, dict) else raw
            bgr = cv2.cvtColor(np.array(Image.open(io.BytesIO(data)).convert("RGB")),
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
                        cv2.resize(crop, (200, 200)))
        print(f"  {display:<34} {kept} samples")
        total += kept

    print(f"\n{total} images written. Analysing so embeddings exist...")
    import analytics
    analytics.scan(progress=None, use_cache=False)

    import train_model
    train_model.train()
    print("Done. Try the Identify panel, or: python seed_demo.py --list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
