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


def instruction(traits, frame_shape=None, mode="idle"):
    """Return a dict describing the single most important thing to fix.

    Keys: severity ('block' | 'warn' | 'ok'), message (terse), detail (the
    why, or None), ready (bool -- nothing is wrong).
    """
    def out(sev, msg, detail=None, ready=False):
        return {"severity": sev, "message": msg, "detail": detail, "ready": ready}

    if not traits:
        return out("block", "Starting camera")

    if traits.get("error"):
        return out("block", "Camera error", str(traits["error"]))

    faces = traits.get("faces", 0)
    detected = traits.get("detected", False)

    # 1. Is there a face at all? Nothing else is measurable until there is.
    if not detected or faces == 0:
        return out("block", "Look at the centre of the camera",
                   "No face detected — take off sunglasses, or a cap or hood "
                   "with a brim shading your eyes.")

    if faces > 1:
        return out("block", "One person only",
                   f"{faces} faces in view — others should step out of frame.")

    # 2. Framing.
    px = traits.get("facePx") or 0
    frame_area = (frame_shape[0] * frame_shape[1]) if frame_shape else None
    if px and px < MIN_FACE_PX:
        return out("block", "Move closer",
                   "Your face is too small in the frame to capture detail.")
    if frame_area and _face_fraction(traits, frame_area) > MAX_FACE_FRACTION:
        return out("warn", "Move back", "Your face is filling the frame.")

    # 3. Pose. Yaw first: turning away hides half the face, tilt only rotates it.
    yaw = traits.get("yaw")
    if yaw is not None and abs(yaw) > MAX_YAW:
        side = "left" if yaw > 0 else "right"
        # Same headline as the no-face case on purpose. From the user's side
        # both are the same problem -- the camera cannot see their face
        # properly -- and a single consistent instruction is easier to act on
        # than two that mean nearly the same thing. The eyewear reminder rides
        # along because a brim shading the eyes is a common reason the pose
        # never reads as frontal however far someone turns.
        return out("block", "Look at the centre of the camera",
                   f"Turn slightly to the {side}. If it still will not lock on, "
                   f"take off sunglasses or a brim shading your eyes.")

    roll = traits.get("roll")
    if roll is not None and abs(roll) > MAX_ROLL:
        return out("warn", "Head upright", "Your head is tilted.")

    # 4. The face itself, from the 68-point fit. These sit above exposure
    # because a blink ruins a sample no matter how well lit it is, and unlike
    # lighting the person can fix them instantly.
    parts = traits.get("parts") or {}
    pflags = parts.get("flags") or []
    if "eyes closed" in pflags:
        return out("block", "Open your eyes",
                   "Both eyes read as closed — the sample would be unusable.")
    if "one eye closed" in pflags:
        return out("warn", "Open both eyes", "One eye reads as closed.")
    if "face partly obscured" in pflags:
        return out("block", "Uncover your face",
                   "One side is measuring very differently from the other — "
                   "something may be covering it, or the light is only hitting "
                   "one side.")
    if "mouth open" in pflags:
        return out("warn", "Neutral expression",
                   "An open mouth changes the shape of the lower face.")

    # 5. Exposure, by clipping only -- never by average level.
    if (traits.get("shadowClip") or 0) > MAX_SHADOW_CLIP:
        return out("warn", "More light in front",
                   "Detail is being lost in shadow.")
    if (traits.get("highlightClip") or 0) > MAX_HIGHLIGHT_CLIP:
        return out("warn", "Less light behind",
                   "Move away from the window, or turn to face the light.")

    # 6. General image quality, last because it is the least specific.
    sharp = traits.get("sharpness")
    if sharp is not None and sharp < MIN_SHARPNESS:
        return out("warn", "Hold still", "The image is blurred.")

    q = traits.get("qualityScore")
    if q is not None and q < MIN_QUALITY:
        return out("warn", "Improve lighting",
                   "Image quality is low. Try more even light, or clean the lens.")

    if mode == "register":
        return out("ok", "Hold still", "Capturing samples.", ready=True)
    return out("ok", READY, None, ready=True)


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
    return [{"label": l, "ok": bool(ok), "fix": fix} for l, ok, fix in items]
