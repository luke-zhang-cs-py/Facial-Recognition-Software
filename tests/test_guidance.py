"""guidance.py decides what a person is told to do. Its ordering is the whole
point: telling somebody their lighting is poor while the camera cannot see
them at all is noise."""
import guidance


def base(**over):
    t = {"detected": True, "faces": 1, "facePx": 200, "yaw": 2.0, "roll": 1.0,
         "sharpness": 200.0, "qualityScore": 0.5, "shadowClip": 0.0,
         "highlightClip": 0.0, "parts": {"flags": []}}
    t.update(over)
    return t


def msg(t):
    return guidance.instruction(t, frame_shape=(480, 640))


def test_clean_frame_is_ready():
    g = msg(base())
    assert g["ready"] and g["severity"] == "ok"


def test_no_traits_at_all_does_not_crash():
    assert msg(None)["severity"] == "block"


def test_no_face_outranks_every_other_problem():
    g = msg(base(detected=False, faces=0, facePx=10, yaw=80.0, sharpness=1.0))
    assert "centre" in g["message"].lower()
    assert not g["ready"]


def test_multiple_faces_blocks():
    assert msg(base(faces=3))["severity"] == "block"


def test_too_far_outranks_pose():
    g = msg(base(facePx=40, yaw=80.0))
    assert "closer" in g["message"].lower()


def test_turned_away_is_reported_as_look_at_the_centre():
    g = msg(base(yaw=45.0))
    assert "centre" in g["message"].lower() and g["severity"] == "block"


def test_eyes_closed_outranks_exposure():
    g = msg(base(parts={"flags": ["eyes closed"]}, shadowClip=0.9))
    assert "eyes" in g["message"].lower()


def test_obscured_face_blocks():
    g = msg(base(parts={"flags": ["face partly obscured"]}))
    assert g["severity"] == "block"


def test_open_mouth_only_warns():
    g = msg(base(parts={"flags": ["mouth open"]}))
    assert g["severity"] == "warn" and not g["ready"]


def test_exposure_uses_clipping_not_brightness():
    """A dark face that is well exposed must pass. Gating on average
    brightness is the bias fixed in a43e2f2 and must not creep back."""
    dark_but_clean = base(brightness=35.0, shadowClip=0.02, highlightClip=0.0)
    assert msg(dark_but_clean)["ready"], \
        "a dark, well-exposed face must not be refused"
    crushed = base(shadowClip=0.9)
    assert not msg(crushed)["ready"]


def test_messages_stay_short_enough_for_the_bar():
    seen = set()
    for over in ({}, {"detected": False, "faces": 0}, {"faces": 2},
                 {"facePx": 30}, {"yaw": 50.0}, {"roll": 40.0},
                 {"shadowClip": 0.9}, {"highlightClip": 0.9},
                 {"sharpness": 1.0}, {"qualityScore": 0.01},
                 {"parts": {"flags": ["eyes closed"]}}):
        g = msg(base(**over))
        seen.add(g["message"])
        assert len(g["message"]) <= 40, f"too long for the bar: {g['message']}"
    assert len(seen) > 5, "distinct problems must give distinct instructions"


def test_checklist_covers_every_failure_with_a_fix():
    items = guidance.checklist(base(detected=False, faces=0), (480, 640))
    assert items
    for c in items:
        assert c["label"] and c["fix"]
