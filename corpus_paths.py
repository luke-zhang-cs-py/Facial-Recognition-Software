"""Where the benchmark corpora live.

None of this is in the repository -- LFW, FairFace, faceage and the
CASIA-WebFace gallery are gigabytes of third-party data, downloaded on demand
and cached. But *where* they are cached had grown five different spellings
across the tools, and three of them were wrong:

  - tools_trials.py read os.environ["TEMP"], which does not exist outside
    Windows, so it raised KeyError before its first trial rather than
    reporting a missing corpus.
  - seed_demo.py and tests/conftest.py fell back to ".", which does not fail
    at all -- it quietly looks in whatever directory the process happens to
    be standing in and concludes the corpus is absent.
  - tools_benchmark_loop.py had FairFace under a Claude Code session
    scratchpad, session id and all. That directory belonged to one session on
    one machine and will never exist again, so the fairness round was
    permanently skipped for everybody, including its author.

One definition each, over tempfile.gettempdir(), which honours TEMP and TMP
on Windows and /tmp elsewhere. Set FACE_CORPORA to keep the corpora outside
the temporary directory -- worth doing, since they take a long time to
download and some systems clear /tmp on boot.
"""
import os
import tempfile


def corpora_dir():
    """Base directory for every cached corpus."""
    return os.environ.get("FACE_CORPORA") or tempfile.gettempdir()


def gallery_dir():
    """Holds the downloaded CASIA shards and the built gallery.

    Overridable on its own with CASIA_DIR, because the shards are the largest
    single thing here and are often worth putting on another disk.
    """
    return os.environ.get("CASIA_DIR") or os.path.join(corpora_dir(), "casia")


def gallery_path():
    """The built gallery: labels and embeddings, one .npz."""
    return os.path.join(gallery_dir(), "gallery.npz")


def lfw_parquet():
    """LFW, used for the identity trials."""
    return os.path.join(corpora_dir(), "lfw", "lfw.parquet")


def fairface_dir():
    """FairFace shards, used for the detection and fairness rounds."""
    return os.path.join(corpora_dir(), "fairface")


def faceage_parquet():
    """The faceage validation split, used for the age-error round."""
    return os.path.join(corpora_dir(), "faceage", "val.parquet")


def sample_frame_dir():
    """Where the app writes best.png / live.png / now.png.

    The tests that need a real photograph look here and skip if there is
    none, because synthesising a face a detector accepts is not realistic.
    """
    return corpora_dir()
