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
import glob
import os

import cv2

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

IMAGE_TYPES = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

Candidate = collections.namedtuple("Candidate", "frame score row")


def frames(path, stride=STRIDE):
    """Yield BGR frames from a video, a single image, or a folder of them."""
    if os.path.isdir(path):
        files = []
        for pattern in IMAGE_TYPES:
            files.extend(glob.glob(os.path.join(path, pattern)))
        for name in sorted(files):
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

    flags = facetraits.quality_flags(
        facetraits.quality_metrics(crop), geom, grayscale_source=False)
    if flags:
        return None, ", ".join(flags)

    # The learned score ranks what survives. None when the model is absent,
    # in which case sharpness is the fallback ordering -- worse, but it
    # still prefers the crisper frame.
    metrics = facetraits.quality_metrics(crop)
    score = metrics["qualityScore"]
    if score is None:
        score = metrics["sharpness"] / 1000.0
    return Candidate(frame, float(score), row), None


def choose(candidates, keep=len(NAMES)):
    """The best `keep` candidates, skipping repeats of the same moment.

    Best first by quality, then each further frame has to be visibly
    different from the ones already chosen -- three views of one instant
    would test the detector once, not three times.
    """
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
    """Write the chosen frames under the names the tests look for."""
    directory = directory or corpus_paths.sample_frame_dir()
    if not os.path.isdir(directory):
        os.makedirs(directory)
    written = []
    for name, cand in zip(NAMES, chosen):
        path = os.path.join(directory, name)
        if cv2.imwrite(path, cand.frame):
            written.append(path)
    return written


def build(path, keep=len(NAMES), stride=STRIDE, directory=None):
    """End to end: (written paths, why frames were rejected)."""
    candidates = []
    rejected = collections.Counter()
    for frame in frames(path, stride):
        cand, reason = assess(frame)
        if cand is None:
            rejected[reason] += 1
        else:
            candidates.append(cand)
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
