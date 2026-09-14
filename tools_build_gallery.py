"""Build a large identity gallery from CASIA-WebFace, embeddings only.

Downloads one shard at a time, keeps a few images per identity, computes SFace
embeddings, then deletes the shard. Storing embeddings instead of images is
what makes a 10k-person gallery practical: 128 float32 is 512 bytes, so ten
thousand people is ~16 MB rather than gigabytes of face crops nobody needs.

    python tools_build_gallery.py [--shards N] [--per-id 4] [--workers 8]

Writes where tools_scale_benchmark.py reads, via corpus_paths -- set
CASIA_DIR to put the shards and the gallery somewhere other than the system
temporary directory.
"""
import argparse
import collections
import io
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

from corpus_paths import gallery_dir, gallery_path   # noqa: E402

D = gallery_dir()
BASE = "https://huggingface.co/api/datasets/SaffalPoosh/casia_web_face/parquet/default/train"
OUT = gallery_path()

PER_ID = 4          # 3 build the centroid, 1 is held out for testing
WORKERS = 8
SHARDS = 20         # the whole set is larger; 20 is a few hours of work

# SFace's output width. Written out as a bare 128 in the empty-array
# fallback, where getting it wrong would produce a gallery that loads and
# then fails to matrix-multiply against anything.
EMBEDDING_DIMS = 128

# Parquet row-group batch. The point of iterating batches at all is to keep
# peak memory to one group rather than a whole shard, so this is the knob
# that decides how much memory the download step costs.
BATCH_ROWS = 2000

# A shard smaller than this is a truncated download or an error page, not
# data. Without the check the parquet reader fails much later with something
# that does not mention the network.
MIN_SHARD_BYTES = 10_000_000
DOWNLOAD_TIMEOUT_SECONDS = 1800

BYTES_PER_MB = 1e6
SECONDS_PER_MINUTE = 60.0


def embed_batch(payload):
    """Worker: decode + embed a list of (global_label, jpeg_bytes)."""
    sys.path.insert(0, PROJ)
    os.chdir(PROJ)
    import cv2
    from PIL import Image
    import traits
    cv2.setNumThreads(1)
    out = []
    for label, data in payload:
        try:
            bgr = cv2.cvtColor(np.array(Image.open(io.BytesIO(data)).convert("RGB")),
                               cv2.COLOR_RGB2BGR)
            rows = traits.detect(bgr)
            if not rows:
                continue
            v = traits.embed(bgr, rows[0])
            if v is not None:
                out.append((label, v.astype(np.float32)))
        except Exception:
            pass
    return out


def build_parser():
    """The command line.

    argparse, not a hand-rolled walk over `sys.argv`. The previous version
    looped over every argument testing it against three names and reading
    `sys.argv[i+1]` -- which accepts `--shards` as the last argument and dies
    with IndexError, silently ignores a misspelled flag, and takes
    `--workers --shards 4` as workers=0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", type=int, default=SHARDS)
    parser.add_argument("--per-id", type=int, default=PER_ID, dest="per_id")
    parser.add_argument("--workers", type=int, default=WORKERS)
    return parser


def download_shard(index, path):
    """Fetch one parquet shard. True if it is now on disk and plausible.

    curl, not urllib: this machine's Python rejects the CA chain ("Basic
    Constraints of CA cert not marked critical"), the same failure that
    stopped sklearn fetching LFW. curl is happy with it.
    """
    print(f"[{index:>2}] downloading...", flush=True)
    code = subprocess.call(
        ["curl", "-sL", "--max-time", str(DOWNLOAD_TIMEOUT_SECONDS),
         f"{BASE}/{index}.parquet", "-o", path])
    if code != 0 or not os.path.exists(path):
        print(f"[{index:>2}] download failed (rc={code})", flush=True)
        return False
    if os.path.getsize(path) < MIN_SHARD_BYTES:
        print(f"[{index:>2}] download truncated "
              f"({os.path.getsize(path)} bytes)", flush=True)
        return False
    return True


def collect_images(path, index, per_id):
    """Up to `per_id` JPEGs per identity, without loading the whole shard.

    Row groups keep peak memory to one batch at a time, which is the only
    reason a multi-gigabyte shard fits.
    """
    wanted = collections.defaultdict(list)
    parquet = pq.ParquetFile(path)
    try:
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS,
                                          columns=["image", "label"]):
            labels = batch.column("label").to_pylist()
            images = batch.column("image").to_pylist()
            for label, image in zip(labels, images):
                key = f"{index}_{label}"
                if len(wanted[key]) < per_id:
                    data = image["bytes"] if isinstance(image, dict) else image
                    wanted[key].append(data)
    finally:
        # Windows will not unlink a file that still has an open handle, and
        # ParquetFile keeps one until told otherwise.
        parquet.close()
    return wanted


def discard(path):
    try:
        os.remove(path)
    except OSError as exc:
        print(f"     (could not delete shard: {exc})", flush=True)


def embed_all(pool, wanted, workers):
    """Embed every collected image across the worker pool."""
    payload = [(key, data) for key, items in wanted.items() for data in items]
    chunks = [payload[i::workers] for i in range(workers)]
    results = []
    for batch in pool.imap_unordered(embed_batch, chunks):
        results.extend(batch)
    return results


def save(labels, vectors):
    stacked = (np.vstack(vectors) if vectors
               else np.zeros((0, EMBEDDING_DIMS), np.float32))
    np.savez_compressed(OUT, labels=np.array(labels), vecs=stacked)


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Was a directory that happened to exist on one machine; now it is
    # derived, so it has to be created. curl will not make a missing parent
    # and would fail every shard with an unhelpful exit code.
    os.makedirs(D, exist_ok=True)

    labels, vectors, seen = [], [], set()
    started = time.time()

    pool = mp.Pool(args.workers)
    try:
        for index in range(args.shards):
            path = os.path.join(D, f"s{index}.parquet")
            if not os.path.exists(path) and not download_shard(index, path):
                print(f"[{index:>2}] stopping here", flush=True)
                break

            size = os.path.getsize(path) / BYTES_PER_MB
            wanted = collect_images(path, index, args.per_id)
            discard(path)

            for key, vector in embed_all(pool, wanted, args.workers):
                labels.append(key)
                vectors.append(vector)
                seen.add(key)

            elapsed = (time.time() - started) / SECONDS_PER_MINUTE
            print(f"[{index:>2}] {size:.0f}MB  +{len(wanted)} ids  "
                  f"total {len(seen)} ids / {len(vectors)} embeddings  "
                  f"{elapsed:.1f}m", flush=True)
            save(labels, vectors)
    finally:
        pool.close()
        pool.join()

    minutes = (time.time() - started) / SECONDS_PER_MINUTE
    print(f"\ndone: {len(seen)} identities, {len(vectors)} embeddings, "
          f"{minutes:.1f} min -> {OUT}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
