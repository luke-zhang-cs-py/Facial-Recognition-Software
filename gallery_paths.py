"""Where the CASIA-WebFace gallery lives.

tools_build_gallery.py writes it and tools_scale_benchmark.py reads it, so the two
have to agree. Both used to spell out an absolute path on one particular
machine -- the same literal, twice, which meant nobody else could run either
and a change to one would silently orphan the other.

One definition, overridable with CASIA_DIR, defaulting under the system
temporary directory because the corpus is several GB of derived data that
nothing should be backing up.
"""
import os
import tempfile


def gallery_dir():
    """The directory holding the downloaded shards and the built gallery."""
    return os.environ.get("CASIA_DIR") or os.path.join(tempfile.gettempdir(),
                                                       "casia")


def gallery_path():
    """The built gallery itself: labels and embeddings, one .npz."""
    return os.path.join(gallery_dir(), "gallery.npz")
