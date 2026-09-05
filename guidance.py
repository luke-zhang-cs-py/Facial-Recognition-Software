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

READY = "Hold still — you're framed correctly."


def _face_fraction(traits, frame_area):
    px = traits.get("facePx") or 0
    if not frame_area or not px:
        return 0.0
    return (px * px) / float(frame_area)


def instruction(traits, frame_shape=None, mode="idle"):
    """Return (severity, message, ready) for the current frame.

    severity is 'block' (cannot proceed), 'warn' (will work but poorly), or
    'ok'. `ready` is True only when nothing is wrong, which is what the UI
    uses to decide whether capture may start.
    """
    if not traits:
        return "block", "Starting the camera…", False

    if traits.get("error"):
        return "block", f"Camera problem: {traits['error']}", False

    faces = traits.get("faces", 0)
    detected = traits.get("detected", False)

    # 1. Is there a face at all? Nothing else is measurable until there is.
    if not detected or faces == 0:
        return "block", (
            "No face detected. Look straight into the camera and make sure "
            "your whole face is visible — take off sunglasses or anything with "
            "a brim that shades your eyes."), False

    if faces > 1:
        return "block", (
            f"{faces} faces in view. Only the person being registered should "
            "be in frame."), False

    # 2. Framing.
    px = traits.get("facePx") or 0
    frame_area = (frame_shape[0] * frame_shape[1]) if frame_shape else None
    if px and px < MIN_FACE_PX:
        return "block", "Move closer to the camera.", False
    if frame_area and _face_fraction(traits, frame_area) > MAX_FACE_FRACTION:
        return "warn", "Move back slightly — your face fills the frame.", False

    # 3. Pose. Yaw first: turning away hides half the face, tilt only rotates it.
    yaw = traits.get("yaw")
    if yaw is not None and abs(yaw) > MAX_YAW:
        side = "left" if yaw > 0 else "right"
        return "block", f"Turn your head slightly to the {side} — look straight at the camera.", False

    roll = traits.get("roll")
    if roll is not None and abs(roll) > MAX_ROLL:
        return "warn", "Straighten your head — it's tilted.", False

    # 4. Exposure, by clipping only.
    if (traits.get("shadowClip") or 0) > MAX_SHADOW_CLIP:
        return "warn", ("Add light in front of you — the camera is losing "
                        "detail in shadow."), False
    if (traits.get("highlightClip") or 0) > MAX_HIGHLIGHT_CLIP:
        return "warn", ("Too much light behind you — move away from the window "
                        "or turn to face the light."), False

    # 5. General image quality, last because it is the least specific.
    sharp = traits.get("sharpness")
    if sharp is not None and sharp < MIN_SHARPNESS:
        return "warn", "Hold still — the image is blurred.", False

    q = traits.get("qualityScore")
    if q is not None and q < MIN_QUALITY:
        return "warn", ("Image quality is low. Try better lighting, or clean "
                        "the camera lens."), False

    if mode == "register":
        return "ok", "Looking good — keep still while samples are captured.", True
    return "ok", READY, True


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
        ("Sharp", sharp is None or sharp >= MIN_SHARPNESS, "Hold still"),
        ("Good quality", q is None or q >= MIN_QUALITY, "Improve lighting"),
    ]
    return [{"label": l, "ok": bool(ok), "fix": fix} for l, ok, fix in items]
