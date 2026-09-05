"""
liveness.py
------------
Presentation-attack detection: is this a real face in front of the camera, or
a photograph of one.

Why this is here
----------------
Without it the attendance system has an obvious hole. Hold a phone showing
someone's photo up to the lens and they are marked present. Every serious
face-recognition project ships liveness detection for exactly this reason,
and this one did not have it. Recognition accuracy is irrelevant if the thing
being recognised is a picture.

Model: MiniFASNet V2 (Silent-Face-Anti-Spoofing, minivision-ai), 1.7 MB ONNX.
Passive -- it works from a single frame, so nobody is asked to blink or turn
on command. Three output classes; index 1 is the genuine face.

It wants a crop with context around the head, not a tight face box: the giveaways
are things like the edge of a phone, a screen's moire, or the flatness of paper,
which live outside the face itself. SCALE controls how much context.

Honest limits
-------------
This has been checked against genuine photographs, where it correctly reports
live. It has NOT been validated against real presentation attacks on this
hardware -- no printed photos or replayed screens were tested, because that
needs someone physically holding one up to the camera. Published figures for
this model family are far from perfect, and lighting and camera quality move
them a lot. Treat LIVE as "no obvious attack detected", not proof.

The threshold is deliberately permissive. A false SPOOF stops a real person
being marked present, which they will notice and be annoyed by; a false LIVE
is one photograph getting through. For an attendance log in a room where
people can see each other, the first is the worse failure.
"""

import os
import threading

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "minifasnet_v2.onnx")

# How much context to include around the face box. 2.7x is what the model was
# trained with; a tight crop drops the very cues it looks for.
SCALE = 2.7
INPUT = (80, 80)
LIVE_CLASS = 1

# Below this the frame is called a spoof. See the note above on which way to
# err -- this is set low on purpose.
LIVE_THRESHOLD = 0.35

# A single frame is noisy, so attendance decisions use a rolling vote rather
# than whatever the latest frame happened to say.
VOTE_WINDOW = 7
VOTE_REQUIRED = 4

_net = None
_lock = threading.Lock()


def available():
    return os.path.exists(MODEL_PATH)


def _get():
    global _net
    if not available():
        return None
    with _lock:
        if _net is None:
            _net = cv2.dnn.readNet(MODEL_PATH)
        return _net


def score(bgr, box):
    """Probability the face in `box` is a live person. None if unavailable."""
    net = _get()
    if net is None or bgr is None:
        return None

    x, y, w, h = (float(v) for v in box)
    cx, cy = x + w / 2.0, y + h / 2.0
    half = max(w, h) * SCALE / 2.0
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(bgr.shape[1], cx + half)), int(min(bgr.shape[0], cy + half))
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    blob = cv2.dnn.blobFromImage(crop, 1.0, INPUT, (0, 0, 0), swapRB=False)
    with _lock:
        net.setInput(blob)
        out = np.ravel(net.forward())
    exp = np.exp(out - out.max())
    probs = exp / exp.sum()
    return float(probs[LIVE_CLASS])


class LivenessVote:
    """Rolling vote over recent frames.

    One frame is a coin flip in poor light. Requiring several of the last few
    frames to agree turns that into something worth acting on, and it costs
    only a fraction of a second of extra dwell time in front of the camera.
    """

    def __init__(self, window=VOTE_WINDOW, required=VOTE_REQUIRED):
        self.window = window
        self.required = required
        self._recent = []

    def push(self, live_score):
        if live_score is None:
            return
        self._recent.append(live_score)
        if len(self._recent) > self.window:
            self._recent.pop(0)

    def reset(self):
        self._recent = []

    @property
    def samples(self):
        return len(self._recent)

    @property
    def mean(self):
        return float(np.mean(self._recent)) if self._recent else None

    def verdict(self):
        """'live', 'spoof', or 'unknown' while there is not enough evidence."""
        if len(self._recent) < self.required:
            return "unknown"
        passing = sum(1 for s in self._recent if s >= LIVE_THRESHOLD)
        return "live" if passing >= self.required else "spoof"
