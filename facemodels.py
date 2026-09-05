"""
facemodels.py
--------------
Lazy loader for the pretrained networks that the trait analysis uses.

Nothing here is required for the original LBPH pipeline to work. Every model
is optional: if the weights are missing, `available()` says so and the
callers degrade to whatever they can still compute. That keeps `python
attendance.py` working on a machine that never ran `fetch_models.py`.

    yunet     face detection + 5 landmarks (eyes, nose, mouth corners).
              The landmarks are what make pose estimation and SFace
              alignment possible.
    sface     128-d face embedding. Unlike LBPH's histogram distance,
              these live in a metric space, so cosine similarity between
              two faces is meaningful and thresholds transfer.
    ediffiqa  learned face-image-quality score in 0..1.
    age       Levi & Hassner (2015) 8-bucket age classifier.
    gender    Levi & Hassner (2015) binary classifier. See traits.py for
              why its output is reported the way it is.

Weights live in models/ and are git-ignored — they are ~134 MB of
third-party binaries, not our source.
"""

import os
import threading

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# name -> (files it needs, human description, size hint)
SPECS = {
    "yunet": (["face_detection_yunet_2023mar.onnx"],
              "face detection + 5 landmarks"),
    "sface": (["face_recognition_sface_2021dec.onnx"],
              "128-d face embeddings"),
    "ediffiqa": (["ediffiqa_tiny_jun2024.onnx"],
                 "learned face image quality"),
    "age": (["age_deploy.prototxt", "age_net.caffemodel"],
            "8-bucket age estimate"),
    "gender": (["gender_deploy.prototxt", "gender_net.caffemodel"],
               "binary gender estimate"),
}

# Levi & Hassner trained on BGR with this mean subtracted and no scaling.
CAFFE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
GENDER_LABELS = ["male", "female"]

_cache = {}
_lock = threading.Lock()


def path_for(filename):
    return os.path.join(MODELS_DIR, filename)


def have(name):
    """True if every file this model needs is on disk."""
    files, _ = SPECS[name]
    return all(os.path.exists(path_for(f)) for f in files)


def available():
    """Report which models are present, for the UI and the CLI."""
    return {
        name: {"ready": have(name), "description": desc,
               "files": [f for f in files if not os.path.exists(path_for(f))]}
        for name, (files, desc) in SPECS.items()
    }


def _build(name):
    if name == "yunet":
        # Input size is re-set per image before every detect() call.
        return cv2.FaceDetectorYN.create(
            path_for("face_detection_yunet_2023mar.onnx"), "", (320, 320),
            score_threshold=0.6, nms_threshold=0.3, top_k=50)
    if name == "sface":
        return cv2.FaceRecognizerSF.create(
            path_for("face_recognition_sface_2021dec.onnx"), "")
    if name == "ediffiqa":
        return cv2.dnn.readNet(path_for("ediffiqa_tiny_jun2024.onnx"))
    if name == "age":
        return cv2.dnn.readNetFromCaffe(path_for("age_deploy.prototxt"),
                                        path_for("age_net.caffemodel"))
    if name == "gender":
        return cv2.dnn.readNetFromCaffe(path_for("gender_deploy.prototxt"),
                                        path_for("gender_net.caffemodel"))
    raise KeyError(name)


def get(name):
    """Return the loaded model, or None if its weights are missing.

    Loading is done once and cached. cv2 nets are not thread-safe for
    concurrent forward passes, so callers hold `lock_for` while inferring.
    """
    if not have(name):
        return None
    with _lock:
        if name not in _cache:
            _cache[name] = (_build(name), threading.Lock())
        return _cache[name][0]


def lock_for(name):
    with _lock:
        entry = _cache.get(name)
    return entry[1] if entry else _lock


def missing_summary():
    """One-line description of what still needs downloading, or None."""
    gone = [n for n in SPECS if not have(n)]
    if not gone:
        return None
    return f"{len(gone)} model(s) not downloaded: {', '.join(gone)}. Run: python fetch_models.py"
