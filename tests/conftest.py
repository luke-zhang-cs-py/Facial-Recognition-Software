import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def isolated_root():
    """Point the whole project -- database and dataset -- at a fresh folder.

    This used to monkeypatch db.DB_PATH, because the modules read their paths
    once at import and paths.use() could not move them afterwards. That is
    fixed, so the redirection is the one call it was always supposed to be,
    and the database and the dataset can no longer end up in different places.
    """
    import paths
    tmp = tempfile.mkdtemp()
    previous = paths.use(tmp)
    try:
        yield tmp
    finally:
        paths.use(previous)


@pytest.fixture
def isolated_db(isolated_root):
    """A throwaway database, so tests never touch the real attendance.db."""
    import db
    db.init_db()
    return db


@pytest.fixture
def face_image():
    """A real photograph with a detectable face, or skip.

    Synthesising a face that a detector accepts is not realistic, so the tests
    that need one use a genuine frame if the corpus is present and skip
    otherwise rather than asserting against a fake.
    """
    import cv2
    for name in ("best.png", "live.png", "now.png"):
        p = os.path.join(os.environ.get("TEMP", "."), name)
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                return img
    pytest.skip("no sample face image available")


@pytest.fixture
def blank_frame():
    return np.full((480, 640, 3), 90, np.uint8)
