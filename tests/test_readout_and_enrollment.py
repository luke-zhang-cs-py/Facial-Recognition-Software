"""The two modules split out of CameraManager.

Both were methods on a class that owns a webcam, which is why they were the
least covered code in the project: `_maybe_traits` and `_build_report`
between them were 190 lines reachable only with a camera attached and a
person sitting in front of it.

Neither does anything a camera is needed for. `readout` reshapes one
`traits.analyze` result into the JSON the sidebar renders; `enrollment` reads
files and database rows. As functions they take a dictionary and return a
dictionary, so these tests are dictionaries in and assertions out.

What is deliberately not faked: `enrollment.nearest_other` runs against a
real (temporary) database, because the query it makes -- every other user's
embeddings, as raw bytes out of a BLOB column -- is most of what could go
wrong with it, and a fake `db` would assert that my idea of the schema is
right rather than that the schema is.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import enrollment                      # noqa: E402
import readout                         # noqa: E402


def trait_result(**overrides):
    """A `traits.analyze` result with every key readout.summarise reads."""
    result = {
        "detected": True,
        "faces": 1,
        "sharpness": 42.0,
        "brightness": 120.0,
        "contrast": 38.0,
        "qualityScore": 0.71,
        "facePx": 180,
        "flags": ["soft"],
        "usable": True,
        "shadowClip": 0.02,
        "highlightClip": 0.01,
        "dynamicRange": 88.0,
        "geometry": {"yaw": -4.5, "roll": 1.2, "pitchRatio": 0.48,
                     "box": (10, 10, 100, 100), "score": 0.9,
                     "landmarks": []},
    }
    result.update(overrides)
    return result


# ------------------------------------------------------------------ readout

def test_the_summary_carries_every_field_the_panel_shows():
    summary = readout.summarise(trait_result())
    for key in readout.DIRECT_KEYS + readout.OPTIONAL_KEYS:
        assert key in summary, key
    assert summary["yaw"] == -4.5
    assert summary["roll"] == 1.2


def test_the_summary_is_json_safe():
    """The whole point of this data is that it gets serialised to a browser.
    A numpy array in it is a 500, and `traits.analyze` returns several."""
    import json

    result = trait_result()
    result["embedding"] = np.zeros(128, dtype=np.float32)
    result["geometry"]["landmarks"] = np.zeros((5, 2), dtype=np.float32)

    summary = readout.summarise(result)
    json.dumps(summary)     # raises TypeError if anything numpy survived

    assert "embedding" not in summary, (
        "the embedding is being shipped to the browser")
    assert "landmarks" not in summary


def test_the_summary_flattens_geometry_to_the_two_angles_used():
    """`geometry` also carries a box, a score and landmark coordinates.
    Nothing on the page reads them and they are numpy, so they stay behind."""
    summary = readout.summarise(trait_result())
    assert "geometry" not in summary
    assert set(summary) - set(readout.DIRECT_KEYS) - set(
        readout.OPTIONAL_KEYS) == {"yaw", "roll"}


def test_a_result_with_no_geometry_still_summarises():
    """No face detected means no angles -- reported as None rather than
    omitted, so the panel can say "unavailable" instead of rendering a gap."""
    summary = readout.summarise(trait_result(geometry=None, detected=False))
    assert summary["yaw"] is None and summary["roll"] is None
    assert summary["detected"] is False


def test_demographics_arrive_with_their_uncertainty():
    """A guess without its confidence is the thing this project is careful
    not to present, so every field travels or none of them do."""
    result = trait_result(demographics={
        "age": {"label": "25-32", "confidence": 0.6, "uncertain": True,
                "runnerUp": "38-43", "estimate": 28},
        "gender": {"label": "Woman", "confidence": 0.9, "uncertain": False,
                   "runnerUp": "Man"},
    })
    summary = readout.add_demographics(readout.summarise(result), result)

    for key in ("age", "gender"):
        for field in readout.DEMOGRAPHIC_FIELDS:
            assert field in summary[key], (key, field)
    assert summary["age"]["estimate"] == 28


def test_a_skipped_demographic_pass_says_so():
    """"We did not look" and "we looked and are unsure" are different
    answers, and the panel shows which."""
    result = trait_result(demographicsSkipped="no face")
    summary = readout.add_demographics(readout.summarise(result), result)
    assert summary["demographicsSkipped"] == "no face"
    assert "age" not in summary and "gender" not in summary


def test_demographics_without_an_age_estimate_do_not_invent_one():
    result = trait_result(demographics={
        "gender": {"label": "Man", "confidence": 0.8, "uncertain": False,
                   "runnerUp": "Woman"},
    })
    summary = readout.add_demographics(readout.summarise(result), result)
    assert "gender" in summary
    assert "age" not in summary, "an absent age was filled in"


def test_part_metrics_are_an_enhancement_not_a_dependency(blank_frame):
    """`add_part_metrics` swallows its exceptions, and this is the one place
    in the module where that is right: the landmark model is the most likely
    of the five to be missing, and losing a section of the sidebar is the
    correct degradation. Losing the whole readout is not."""
    summary = readout.summarise(trait_result())
    before = dict(summary)

    returned = readout.add_part_metrics(summary, blank_frame)
    assert returned is summary, "it should augment in place and return it"
    # A blank frame has no face, so there is nothing to add -- and nothing
    # should have been taken away either.
    assert summary["flags"] == before["flags"]
    assert summary["usable"] == before["usable"]


def test_part_metrics_survive_a_landmark_model_that_raises(blank_frame,
                                                           monkeypatch):
    """Demonstrated rather than asserted from the source."""
    def explode(*_args, **_kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr(readout.facetraits, "detect", explode)
    summary = readout.summarise(trait_result())
    readout.add_part_metrics(summary, blank_frame)
    assert "parts" not in summary
    assert summary["usable"] is True, "the readout was lost with the parts"


def test_guidance_is_attached_for_the_mode_given(blank_frame):
    summary = readout.summarise(trait_result())
    readout.add_guidance(summary, blank_frame.shape[:2], "register")
    assert summary["guidance"]
    assert isinstance(summary["checklist"], (list, tuple))


def test_build_runs_all_four_steps(blank_frame):
    result = trait_result(demographics={
        "gender": {"label": "Man", "confidence": 0.8, "uncertain": False,
                   "runnerUp": "Woman"}})
    summary = readout.build(result, blank_frame, "idle")

    assert summary["qualityScore"] == 0.71      # summarise
    assert "guidance" in summary                # add_guidance
    assert "checklist" in summary
    assert "gender" in summary                  # add_demographics


# --------------------------------------------------------------- enrollment

def test_pose_coverage_counts_each_stage():
    poses = [{"stage": "front", "yaw": 1.0}, {"stage": "front", "yaw": -2.0},
             {"stage": "left", "yaw": -20.0}, {"stage": "right", "yaw": 21.0}]
    stages, spread, span = enrollment.pose_coverage(poses)

    assert stages == {"front": 2, "left": 1, "right": 1}
    assert spread == pytest.approx(np.std([1.0, -2.0, -20.0, 21.0]), abs=0.05)
    assert span == [-20.0, 21.0]


def test_one_sample_has_no_yaw_spread_rather_than_a_spread_of_zero():
    """A standard deviation of one value is 0, which reads as "every sample
    was at the same angle" -- the opposite of "not enough data to say"."""
    stages, spread, span = enrollment.pose_coverage(
        [{"stage": "front", "yaw": 3.0}])
    assert stages == {"front": 1}
    assert spread is None
    assert span == [3.0, 3.0]


def test_poses_with_no_yaw_at_all_still_count_towards_their_stage():
    """The landmark model can fail on a frame that was still captured."""
    stages, spread, span = enrollment.pose_coverage(
        [{"stage": "front"}, {"stage": "front", "yaw": None}])
    assert stages == {"front": 2}
    assert spread is None and span is None


def test_the_centroid_is_a_unit_vector():
    """A centroid is far more stable than any single shot, which is the whole
    point of capturing thirty. It has to be re-normalised or the cosine
    similarities it is compared with stop being cosines."""
    records = [{"embedding": np.array([3.0, 0.0, 0.0], dtype=np.float32)},
               {"embedding": np.array([0.0, 4.0, 0.0], dtype=np.float32)}]
    centroid = enrollment.centroid_of(records)
    assert float(np.linalg.norm(centroid)) == pytest.approx(1.0, abs=1e-6)


def test_records_without_embeddings_have_no_centroid():
    assert enrollment.centroid_of([]) is None
    assert enrollment.centroid_of([{"embedding": None}]) is None


def test_a_zero_centroid_does_not_divide_by_zero():
    """Two opposite embeddings average to the origin. `or 1.0` is what stops
    that being a divide-by-zero, and a NaN vector downstream."""
    records = [{"embedding": np.array([1.0, 0.0], dtype=np.float32)},
               {"embedding": np.array([-1.0, 0.0], dtype=np.float32)}]
    centroid = enrollment.centroid_of(records)
    assert centroid is not None
    assert np.isfinite(centroid).all(), "the centroid came back as NaN"


def test_nearest_other_finds_the_most_similar_enrolled_person(isolated_db):
    """Against a real database, because the BLOB round-trip is most of what
    could go wrong here."""
    db = isolated_db
    same = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    near = np.array([0.95, 0.31, 0.0], dtype=np.float32)
    far = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    mine = db.add_user("Subject")
    close_id = db.add_user("Close")
    distant_id = db.add_user("Distant")

    for user_id, vector in ((close_id, near), (distant_id, far)):
        db.save_traits({
            "path": f"/tmp/{user_id}.jpg", "user_id": user_id, "mtime": 1.0,
            "sharpness": 1.0, "brightness": 1.0, "contrast": 1.0,
            "quality": 1.0, "face_px": 100, "yaw": 0.0, "roll": 0.0,
            "detected": 1, "flags": "",
            "embedding": vector.tobytes(),
            "age_label": None, "age_conf": None,
            "gender_label": None, "gender_conf": None,
        })

    found = enrollment.nearest_other(mine, [{"embedding": same}])
    assert found is not None
    assert found["name"] == "Close", "it picked the further person"
    assert 0.0 < found["similarity"] <= 1.0


def test_nearest_other_excludes_the_person_being_enrolled(isolated_db):
    """Otherwise every enrollment reports itself as its own nearest match at
    similarity 1.0, which is both useless and alarming."""
    db = isolated_db
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mine = db.add_user("Only Person")
    db.save_traits({
        "path": "/tmp/mine.jpg", "user_id": mine, "mtime": 1.0,
        "sharpness": 1.0, "brightness": 1.0, "contrast": 1.0, "quality": 1.0,
        "face_px": 100, "yaw": 0.0, "roll": 0.0, "detected": 1, "flags": "",
        "embedding": vector.tobytes(),
        "age_label": None, "age_conf": None,
        "gender_label": None, "gender_conf": None,
    })

    assert enrollment.nearest_other(mine, [{"embedding": vector}]) is None


def test_nearest_other_skips_people_with_no_embeddings(isolated_db):
    """Somebody enrolled before the SFace model was installed has rows with
    a NULL embedding. Those are skipped, not treated as the origin."""
    db = isolated_db
    mine = db.add_user("Subject")
    bare = db.add_user("No Embeddings")
    db.save_traits({
        "path": "/tmp/bare.jpg", "user_id": bare, "mtime": 1.0,
        "sharpness": 1.0, "brightness": 1.0, "contrast": 1.0, "quality": 1.0,
        "face_px": 100, "yaw": 0.0, "roll": 0.0, "detected": 1, "flags": "",
        "embedding": None,
        "age_label": None, "age_conf": None,
        "gender_label": None, "gender_conf": None,
    })

    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert enrollment.nearest_other(mine, [{"embedding": vector}]) is None


def test_samples_from_a_missing_folder_is_empty_not_an_error():
    """The registration may have been cancelled before anything was written."""
    assert enrollment.samples_from_folder(1, None) == []
    assert enrollment.samples_from_folder(1, "/does/not/exist") == []


def test_an_enrollment_that_produced_nothing_reports_nothing(isolated_db,
                                                             tmp_path):
    """None rather than a report full of nulls: a registration with no
    analysable samples has nothing to say about itself, and a table of
    nulls reads as a measurement rather than as an absence."""
    empty = tmp_path / "no-samples"
    empty.mkdir()
    assert enrollment.build(1, "Nobody", [], str(empty)) is None
