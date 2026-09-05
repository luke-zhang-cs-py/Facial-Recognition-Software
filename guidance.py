"""
guidance.py
------------
Turns a live trait read into one instruction the person in front of the
camera can act on.

Design notes
------------
*One* instruction at a time, not a list. A panel reading "move closer, turn
towards the camera, remove your glasses, improve the lighting" gets ignored;
people fix one thing at a time. So the checks are ordered by how much they
block enrollment, and the first failing one is what gets shown.

The order is deliberate. No face detected is absolute -- nothing else can be
measured until it is fixed. Then framing (too small / off-centre), then pose,
then image quality. Telling someone their lighting is poor while the camera
cannot even see them is noise.

Nothing here gates on skin tone. The old quality gate thresholded absolute
brightness, which encoded skin tone (2.15x disparity across race groups; see
BENCHMARK.md); saying "your face is too dark" to somebody because of their
complexion is that same defect wearing a friendlier voice. Exposure advice
here fires only on clipping -- genuinely crushed or blown pixels -- which is
a property of the photograph.

On head coverings: the honest failure is "the face is not being detected",
not a guess about what someone is wearing. A hijab, turban, or kippah does
not interfere with face detection and there is no reason to ask anyone to
remove one. Brims and dark lenses do occlude, so those are named specifically
and only when detection is actually failing or the eyes are not being found.
"""

# Framing
MIN_FACE_PX = 110          # below this the crop carries too little detail
IDEAL_FACE_PX = 160
MAX_FACE_FRACTION = 0.85   # face filling the frame means it is clipped

# Pose. Enrollment wants near-frontal; a little variety is good, a profile
# is not. These are looser than the analysis gate because this is live
# coaching, not a pass/fail on a stored sample.
MAX_YAW = 22.0
MAX_ROLL = 15.0

# Image quality. eDifFIQA was the fairest signal measured (1.21x).
MIN_QUALITY = 0.22
MIN_SHARPNESS = 25.0

# Exposure: clipping only, never average level.
MAX_SHADOW_CLIP = 0.45
MAX_HIGHLIGHT_CLIP = 0.30

# Instructions are two or three words. A 300px sidebar and a bar burned into a
# 640px frame are both too narrow for a sentence, and someone squinting at a
# webcam does not read prose -- they read a verb. The longer explanation goes
# in `detail`, which the checklist shows underneath, so nothing is lost.
READY = "Ready"


def _face_fraction(traits, frame_area):
    px = traits.get("facePx") or 0
    if not frame_area or not px:
        return 0.0
    return (px * px) / float(frame_area)


def _out(sev, msg, detail=None, ready=False):
    return {"severity": sev, "message": msg, "detail": detail, "ready": ready}


class Reading:
    """One trait read, with the derived numbers the rules ask about.

    A small object rather than passing `traits`, `frame_shape` and four
    recomputed values to every rule: the rules below read like sentences
    about a reading, and adding a derived quantity is one property here
    instead of an argument threaded through eighteen places.
    """

    def __init__(self, traits, frame_shape=None):
        self.traits = traits
        self.frame_area = (frame_shape[0] * frame_shape[1]) if frame_shape else None
        self.faces = traits.get("faces", 0)
        self.detected = traits.get("detected", False)
        self.face_px = traits.get("facePx") or 0
        self.yaw = traits.get("yaw")
        self.roll = traits.get("roll")
        self.part_flags = (traits.get("parts") or {}).get("flags") or []

    def get(self, key, default=None):
        return self.traits.get(key, default)

    def face_fraction(self):
        return _face_fraction(self.traits, self.frame_area)


# ---------------------------------------------------------------------------
# The rules, in priority order
# ---------------------------------------------------------------------------
# Each returns a verdict or None. RULES below is the order, and the order is
# the whole design: one instruction at a time, and the first thing that is
# actually blocking is the one worth saying.
#
# This was written as eighteen sequential `if ... return` statements in one
# function. It worked, and the priority order was invisible -- to see it you
# had to read the whole body, and to change it you had to move blocks of code
# past each other. Now the order is a list you can read in one glance.

def _rule_no_face(r):
    if not r.detected or r.faces == 0:
        return _out("block", "Look at the centre of the camera",
                    "No face detected — take off sunglasses, or a cap or hood "
                    "with a brim shading your eyes.")


def _rule_crowd(r):
    if r.faces > 1:
        return _out("block", "One person only",
                    f"{r.faces} faces in view — others should step out of frame.")


def _rule_too_far(r):
    if r.face_px and r.face_px < MIN_FACE_PX:
        return _out("block", "Move closer",
                    "Your face is too small in the frame to capture detail.")


def _rule_too_close(r):
    if r.frame_area and r.face_fraction() > MAX_FACE_FRACTION:
        return _out("warn", "Move back", "Your face is filling the frame.")


def _rule_yaw(r):
    """Yaw before roll: turning away hides half the face, tilt only rotates it.

    Same headline as the no-face case on purpose. From the user's side both
    are the same problem -- the camera cannot see their face properly -- and
    one consistent instruction is easier to act on than two that mean nearly
    the same thing. The eyewear reminder rides along because a brim shading
    the eyes is a common reason the pose never reads as frontal however far
    someone turns.
    """
    if r.yaw is not None and abs(r.yaw) > MAX_YAW:
        side = "left" if r.yaw > 0 else "right"
        return _out("block", "Look at the centre of the camera",
                    f"Turn slightly to the {side}. If it still will not lock on, "
                    f"take off sunglasses or a brim shading your eyes.")


def _rule_roll(r):
    if r.roll is not None and abs(r.roll) > MAX_ROLL:
        return _out("warn", "Head upright", "Your head is tilted.")


def _rule_eyes_closed(r):
    if "eyes closed" in r.part_flags:
        return _out("block", "Open your eyes",
                    "Both eyes read as closed — the sample would be unusable.")


def _rule_one_eye_closed(r):
    if "one eye closed" in r.part_flags:
        return _out("warn", "Open both eyes", "One eye reads as closed.")


def _rule_obscured(r):
    if "face partly obscured" in r.part_flags:
        return _out("block", "Uncover your face",
                    "One side is measuring very differently from the other — "
                    "something may be covering it, or the light is only hitting "
                    "one side.")


def _rule_mouth_open(r):
    if "mouth open" in r.part_flags:
        return _out("warn", "Neutral expression",
                    "An open mouth changes the shape of the lower face.")


def _rule_shadow(r):
    if (r.get("shadowClip") or 0) > MAX_SHADOW_CLIP:
        return _out("warn", "More light in front", "Detail is being lost in shadow.")


def _rule_highlight(r):
    if (r.get("highlightClip") or 0) > MAX_HIGHLIGHT_CLIP:
        return _out("warn", "Less light behind",
                    "Move away from the window, or turn to face the light.")


def _rule_blur(r):
    sharp = r.get("sharpness")
    if sharp is not None and sharp < MIN_SHARPNESS:
        return _out("warn", "Hold still", "The image is blurred.")


def _rule_quality(r):
    q = r.get("qualityScore")
    if q is not None and q < MIN_QUALITY:
        return _out("warn", "Improve lighting",
                    "Image quality is low. Try more even light, or clean the lens.")


# Read this top to bottom and you have read the design: a face at all, then
# how many, then framing, then pose, then the face itself, then exposure,
# then general quality. Telling somebody their lighting is poor while the
# camera cannot see them is noise.
RULES = (
    _rule_no_face,
    _rule_crowd,
    _rule_too_far,
    _rule_too_close,
    _rule_yaw,
    _rule_roll,
    _rule_eyes_closed,
    _rule_one_eye_closed,
    _rule_obscured,
    _rule_mouth_open,
    _rule_shadow,
    _rule_highlight,
    _rule_blur,
    _rule_quality,
)


def instruction(traits, frame_shape=None, mode="idle"):
    """Return a dict describing the single most important thing to fix.

    Keys: severity ('block' | 'warn' | 'ok'), message (terse), detail (the
    why, or None), ready (bool -- nothing is wrong).
    """
    if not traits:
        return _out("block", "Starting camera")
    if traits.get("error"):
        return _out("block", "Camera error", str(traits["error"]))

    reading = Reading(traits, frame_shape)
    for rule in RULES:
        verdict = rule(reading)
        if verdict is not None:
            return verdict

    if mode == "register":
        return _out("ok", "Hold still", "Capturing samples.", ready=True)
    return _out("ok", READY, None, ready=True)


def checklist(traits, frame_shape=None):
    """Every check with its current state, for a UI that wants the whole list.

    The single `instruction` is what people act on; this is for the panel that
    shows what is and is not satisfied, so a failure is never a mystery.
    """
    if not traits:
        return []
    detected = traits.get("detected", False)
    faces = traits.get("faces", 0)
    px = traits.get("facePx") or 0
    yaw, roll = traits.get("yaw"), traits.get("roll")
    q, sharp = traits.get("qualityScore"), traits.get("sharpness")
    pflags = (traits.get("parts") or {}).get("flags") or []

    items = [
        ("Face visible", detected and faces >= 1,
         "Look straight into the camera"),
        ("Only one person", faces <= 1, "Others should step out of frame"),
        ("Close enough", bool(px) and px >= MIN_FACE_PX, "Move closer"),
        ("Facing forward", yaw is None or abs(yaw) <= MAX_YAW,
         "Turn to face the camera"),
        ("Head upright", roll is None or abs(roll) <= MAX_ROLL,
         "Straighten your head"),
        ("Eyes unobstructed", detected and faces >= 1,
         "Remove sunglasses or a shading brim"),
        ("Eyes open", "eyes closed" not in pflags and "one eye closed" not in pflags,
         "Open both eyes"),
        ("Neutral expression", "mouth open" not in pflags, "Close your mouth"),
        ("Both sides visible", "face partly obscured" not in pflags,
         "Uncover your face, or even out the light"),
        ("Sharp", sharp is None or sharp >= MIN_SHARPNESS, "Hold still"),
        ("Good quality", q is None or q >= MIN_QUALITY, "Improve lighting"),
    ]
    return [{"label": label, "ok": bool(ok), "fix": fix}
            for label, ok, fix in items]
