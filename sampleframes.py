"""Pick usable sample frames out of a video or a folder of photographs.

Three tests want a real photograph with a detectable face and skip without
one, because synthesising a face a detector accepts is not realistic. They
look for best.png / live.png / now.png, and nothing in this project has
ever written those files -- so they were not conditionally skipped, they
were permanently skipped. This is what writes them.

The filtering is the project's own. `traits.detect` finds the face,
`count_people` refuses a frame with somebody else in it, and
`quality_flags` decides whether it is a good photograph -- the same
thresholds the enrollment gate uses, and the same ones measured for
skin-tone bias. The full set applies here, pose included: a sample frame
should be a good frontal portrait, which is exactly what a registration
frame is allowed not to be.

What this does not do is fetch anything. Point it at a video you already
have the right to use, or at the still corpora the benchmark tools already
download. Scraping faces off the internet to build a recognition database
is prohibited outright under the EU AI Act, and a face template is
special-category data under UK GDPR, so "it was on a public website" is not
a licence. The corpora in corpus_paths.py are there because they come with
one.
"""
import collections
import os

import cv2
import numpy as np

import corpus_paths
import traits as facetraits

# A 30fps clip hands over thirty near-identical frames a second, and
# scoring each one costs a full quality read. Every fifth is plenty to find
# the good moments.
STRIDE = 5

# The names tests/conftest.py looks for, in the order it tries them.
NAMES = ("best.png", "live.png", "now.png")

# Cosine similarity above which two frames are "the same shot". SFace calls
# the same *person* about 0.36, so this is deliberately far higher: we are
# rejecting duplicate moments, not duplicate people.
SAME_SHOT = 0.92

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})

# Rows read from a corpus shard at a time.
PARQUET_BATCH = 64

# Frames examined before stopping, when a caller does not say.
# An LFW shard is 13,233 rows and each one costs a network
# forward pass, so scanning the lot to keep three is hours of
# work for no better answer. 0 means no limit.
LIMIT = 200

Candidate = collections.namedtuple("Candidate", "frame score row")


def parquet_frames(path):
    """Yield BGR images out of a corpus shard.

    The corpora this tool points at are parquet, not folders: LFW arrives
    as one file of 13,233 rows with an `image` column of encoded bytes.
    Telling somebody to point this at a licensed corpus and then handing
    the path to cv2.VideoCapture, which cannot read parquet, was advice
    that did not work -- proven by having to write this extraction by hand
    to use it.

    pyarrow is imported here rather than at the top: it is what the
    benchmark tools use, and a video or a folder should not need it
    installed.
    """
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    if "image" not in handle.schema_arrow.names:
        raise ValueError("%s has no 'image' column; its columns are %s"
                         % (path, handle.schema_arrow.names))
    for batch in handle.iter_batches(batch_size=PARQUET_BATCH):
        for cell in batch.column("image"):
            value = cell.as_py()
            if value is None:
                continue
            # HuggingFace image columns are {bytes, path}; a plain bytes
            # column is also allowed.
            raw = value["bytes"] if isinstance(value, dict) else value
            if not raw:
                continue
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield img


def frames(path, stride=STRIDE):
    """Yield BGR frames from a video, an image, a folder, or a corpus shard."""
    if os.path.splitext(path)[1].lower() == ".parquet":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        for img in parquet_frames(path):
            yield img
        return

    if os.path.isdir(path):
        # Walked, not globbed at the top level. The corpora this is meant to
        # be pointed at are one folder per identity -- LFW is
        # lfw/Person_Name/Person_Name_0001.jpg -- so a top-level glob found
        # nothing at all in exactly the case the tool advertises. Suffixes
        # are matched case-insensitively too: cameras write .JPG.
        found = []
        for folder, _subfolders, names in os.walk(path):
            for name in names:
                if os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES:
                    found.append(os.path.join(folder, name))
        found.sort()
        # No stride here. It exists because a clip hands over thirty
        # near-identical frames a second; a folder of photographs is
        # already somebody's selection, and thinning it just loses images.
        for name in found:
            img = cv2.imread(name)
            if img is not None:
                yield img
        return

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # cv2.VideoCapture reads stills as a one-frame video, so a photograph
    # and a clip take the same path from here.
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("could not open %s as a video or an image" % path)
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % max(1, stride) == 0:
                yield frame
            index += 1
    finally:
        cap.release()


def assess(frame):
    """(Candidate, None) if the frame is usable, else (None, reason)."""
    rows = facetraits.detect(frame)
    if not rows:
        return None, "no face"
    if facetraits.count_people(rows) > 1:
        return None, "more than one person"

    row = rows[0]
    geom = facetraits.geometry(row)
    x, y, w, h = geom["box"]
    x0, y0 = max(0, x), max(0, y)
    crop = frame[y0:y + h, x0:x + w]
    if crop.size == 0:
        return None, "face outside the frame"

    # Once, not twice. quality_metrics runs eDifFIQA, a network forward
    # pass, and this was calling it to get the flags and then again to get
    # the score -- doubling the cost of the most expensive step on every
    # frame of the video.
    metrics = facetraits.quality_metrics(crop)

    flags = facetraits.quality_flags(metrics, geom, grayscale_source=False)
    if flags:
        return None, ", ".join(flags)

    # The learned score ranks what survives. None when the model is absent,
    # in which case sharpness is the fallback ordering -- worse, but it
    # still prefers the crisper frame.
    score = metrics["qualityScore"]
    if score is None:
        score = metrics["sharpness"] / 1000.0
    return Candidate(frame, float(score), row), None


def choose(candidates, keep=len(NAMES)):
    """The best `keep` candidates, skipping repeats of the same moment.

    Best first by quality, then each further frame has to be visibly
    different from the ones already chosen -- three views of one instant
    would test the detector once, not three times.

    Capped at len(NAMES), because `write` has that many names and a fourth
    frame would have nowhere to go.
    """
    keep = min(keep, len(NAMES))
    ranked = sorted(candidates, key=lambda c: -c.score)
    chosen, vectors = [], []
    for cand in ranked:
        if len(chosen) >= keep:
            break
        vec = facetraits.embed(cand.frame, cand.row)
        if vec is not None and any(
                (facetraits.cosine(vec, other) or 0.0) > SAME_SHOT
                for other in vectors):
            continue
        chosen.append(cand)
        if vec is not None:
            vectors.append(vec)
    return chosen


def write(chosen, directory=None):
    """Write the chosen frames under the names the tests look for.

    At most len(NAMES) of them: there are three names and nothing reads a
    fourth file, so writing one would leave dead weight on disk. This used
    to fall out of `zip` truncating, which meant --keep 10 quietly wrote
    three and said nothing. `choose` is capped instead, so the count the
    caller asks for is the count it gets or it hears why not.
    """
    directory = directory or corpus_paths.sample_frame_dir()
    if not os.path.isdir(directory):
        os.makedirs(directory)
    written = []
    for name, cand in zip(NAMES, chosen):
        path = os.path.join(directory, name)
        if cv2.imwrite(path, cand.frame):
            written.append(path)
    return written


def scan(path, stride=STRIDE, limit=LIMIT):
    """Assess frames: (usable candidates, why the rest were rejected).

    Separate from `build` because the tool's --dry-run wants exactly this
    and nothing after it. It had its own copy of the loop, which is one
    more place for the two to disagree about what counts as usable.

    `limit` caps how many frames are examined, because a corpus shard has
    thousands of rows and only three are ever kept. 0 examines everything.
    """
    candidates = []
    rejected = collections.Counter()
    for seen, frame in enumerate(frames(path, stride)):
        if limit and seen >= limit:
            break
        cand, reason = assess(frame)
        if cand is None:
            rejected[reason] += 1
        else:
            candidates.append(cand)
    return candidates, rejected


def build(path, keep=len(NAMES), stride=STRIDE, directory=None, limit=LIMIT):
    """End to end: (written paths, why frames were rejected)."""
    candidates, rejected = scan(path, stride, limit)
    return write(choose(candidates, keep), directory), rejected


def summarise(written, rejected):
    """Lines describing what happened, for a tool to print."""
    lines = []
    if written:
        lines.append("wrote %d sample frame(s):" % len(written))
        lines.extend("  " + os.path.basename(p) for p in written)
        lines.append("  in " + os.path.dirname(written[0]))
    else:
        lines.append("no frame was good enough to keep.")
    if rejected:
        lines.append("rejected:")
        for reason, count in rejected.most_common():
            lines.append("  %-34s %d" % (reason, count))
    return lines
