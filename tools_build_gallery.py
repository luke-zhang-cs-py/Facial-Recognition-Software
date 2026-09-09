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
import io, os, sys, time, subprocess, collections
import numpy as np, pyarrow.parquet as pq
import multiprocessing as mp

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

from corpus_paths import gallery_dir, gallery_path   # noqa: E402

D = gallery_dir()
BASE = "https://huggingface.co/api/datasets/SaffalPoosh/casia_web_face/parquet/default/train"
OUT = gallery_path()

PER_ID = 4          # 3 build the centroid, 1 is held out for testing
WORKERS = 8


def embed_batch(payload):
    """Worker: decode + embed a list of (global_label, jpeg_bytes)."""
    sys.path.insert(0, PROJ); os.chdir(PROJ)
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


def main():
    shards = 20; per_id = PER_ID; workers = WORKERS
    for i, a in enumerate(sys.argv):
        if a == "--shards": shards = int(sys.argv[i+1])
        if a == "--per-id": per_id = int(sys.argv[i+1])
        if a == "--workers": workers = int(sys.argv[i+1])

    # Was a directory that happened to exist on one machine; now it is derived,
    # so it has to be created. curl will not make a missing parent and would
    # fail every shard with an unhelpful exit code.
    os.makedirs(D, exist_ok=True)

    labels_all, vecs_all = [], []
    seen_ids = set()
    t0 = time.time()

    pool = mp.Pool(workers)
    try:
        for k in range(shards):
            path = os.path.join(D, f"s{k}.parquet")
            if not os.path.exists(path):
                print(f"[{k:>2}] downloading...", flush=True)
                # curl, not urllib: this machine's Python rejects the CA chain
                # ("Basic Constraints of CA cert not marked critical"), the same
                # failure that stopped sklearn fetching LFW earlier. curl is
                # happy with it.
                rc = subprocess.call(["curl", "-sL", "--max-time", "1800",
                                      f"{BASE}/{k}.parquet", "-o", path])
                if rc != 0 or not os.path.exists(path) or os.path.getsize(path) < 10_000_000:
                    print(f"[{k:>2}] download failed (rc={rc}); stopping here", flush=True)
                    break
            size = os.path.getsize(path)/1e6

            # collect up to per_id images per identity without loading the
            # whole shard: row groups keep peak memory to one group at a time
            wanted = collections.defaultdict(list)
            pf = pq.ParquetFile(path)
            try:
                for batch in pf.iter_batches(batch_size=2000, columns=["image", "label"]):
                    labs = batch.column("label").to_pylist()
                    imgs = batch.column("image").to_pylist()
                    for lab, im in zip(labs, imgs):
                        gid = f"{k}_{lab}"
                        if len(wanted[gid]) < per_id:
                            data = im["bytes"] if isinstance(im, dict) else im
                            wanted[gid].append(data)
            finally:
                # Windows will not unlink a file that still has an open handle,
                # and ParquetFile keeps one until told otherwise.
                pf.close()
            try:
                os.remove(path)
            except OSError as exc:
                print(f"     (could not delete shard: {exc})", flush=True)

            payload = [(gid, d) for gid, ds in wanted.items() for d in ds]
            chunks = [payload[i::workers] for i in range(workers)]
            for res in pool.imap_unordered(embed_batch, chunks):
                for gid, v in res:
                    labels_all.append(gid); vecs_all.append(v); seen_ids.add(gid)

            el = time.time()-t0
            print(f"[{k:>2}] {size:.0f}MB  +{len(wanted)} ids  "
                  f"total {len(seen_ids)} ids / {len(vecs_all)} embeddings  "
                  f"{el/60:.1f}m", flush=True)
            np.savez_compressed(OUT, labels=np.array(labels_all),
                                vecs=np.vstack(vecs_all) if vecs_all else np.zeros((0,128),np.float32))
    finally:
        pool.close(); pool.join()

    print(f"\ndone: {len(seen_ids)} identities, {len(vecs_all)} embeddings, "
          f"{(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
