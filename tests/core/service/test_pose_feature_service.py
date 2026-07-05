import math

import numpy as np
import pytest
from hypothesis import assume, given, strategies as st

from core.entities.pose_entity import Keypoint, KeypointCollection, PoseFeatures
from core.service.pose_feature_service import PoseFeatureService

_coord = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
_pt = st.tuples(_coord, _coord)


def kc(**joints):
    """Build a KeypointCollection from name=(x, y[, confidence]) tuples."""
    kps = {}
    for name, vals in joints.items():
        x, y = vals[0], vals[1]
        conf = vals[2] if len(vals) > 2 else 1.0
        kps[name] = Keypoint(x=float(x), y=float(y), confidence=float(conf))
    return KeypointCollection(keypoints=kps)


class TestExtractBodyAlignment:
    def test_upright_torso_has_near_zero_vertical_alignment(self):
        # Shoulders directly above hips. In image space (y grows downward) an
        # upright torso should read ~0 deg from vertical, not ~180.
        service = PoseFeatureService()
        keypoints = kc(
            left_shoulder=(0.4, 0.3),
            right_shoulder=(0.6, 0.3),
            left_hip=(0.4, 0.6),
            right_hip=(0.6, 0.6),
        )
        vertical, lateral = service.extract_body_alignment(keypoints)
        assert vertical == pytest.approx(0.0, abs=1.0)

    def test_vertical_alignment_preserved_when_lateral_is_degenerate(self):
        # Coincident shoulders make the lateral (shoulder-vs-hip) computation
        # degenerate, but the vertical alignment is still well-defined here
        # (~18.4 deg) and must not be discarded as 0.
        service = PoseFeatureService()
        keypoints = kc(
            left_shoulder=(0.6, 0.3),
            right_shoulder=(0.6, 0.3),  # zero-length shoulder vector
            left_hip=(0.4, 0.6),
            right_hip=(0.6, 0.6),
        )
        vertical, lateral = service.extract_body_alignment(keypoints)
        assert vertical == pytest.approx(18.43, abs=1.0)


class TestExtractJointAngles:
    def test_angle_ignores_confidence_as_coordinate(self):
        # A right angle at the elbow in the image plane. The three keypoints have
        # different confidences, which must NOT enter the angle math as a z-axis.
        service = PoseFeatureService()
        keypoints = kc(
            left_shoulder=(0.5, 0.5, 0.2),
            left_elbow=(0.5, 0.7, 0.9),
            left_wrist=(0.7, 0.7, 0.5),
        )
        angles = service.extract_joint_angles(keypoints)
        assert angles["left_shoulder_left_elbow_left_wrist"] == pytest.approx(
            90.0, abs=0.5
        )


class TestDetectFormIssues:
    def _features(self, joint_angles):
        return PoseFeatures(
            keypoints=KeypointCollection(keypoints={}), joint_angles=joint_angles
        )

    def test_missing_elbow_angle_does_not_flag_elbow_position(self):
        # No elbow angle available (occluded wrist). A missing angle must not be
        # treated as a bad-form 180-degree default and flagged.
        service = PoseFeatureService()
        issues = service.detect_form_issues(self._features({}), "bench_press")
        assert not issues.get("elbow_position")

    def test_present_bad_elbow_angle_still_flags_elbow_position(self):
        # A real, measured over-extended elbow must still be flagged, so the
        # missing-angle fix doesn't render the check dead.
        service = PoseFeatureService()
        features = self._features(
            {
                "left_shoulder_left_elbow_left_wrist": 130.0,
                "right_shoulder_right_elbow_right_wrist": 130.0,
            }
        )
        issues = service.detect_form_issues(features, "bench_press")
        assert issues.get("elbow_position")


class TestProcessFeatures:
    def test_builds_body_alignment_without_error(self):
        # The BodyAlignment pydantic model must be constructed with keyword
        # fields; positional construction raises TypeError.
        service = PoseFeatureService()
        keypoints = kc(
            left_shoulder=(0.4, 0.3),
            right_shoulder=(0.6, 0.3),
            left_hip=(0.4, 0.6),
            right_hip=(0.6, 0.6),
        )
        features = service.process_features(keypoints)
        assert features.body_alignment is not None
        assert features.body_alignment.vertical_alignment == pytest.approx(
            0.0, abs=1.0
        )

    def test_populates_speeds_field_from_movement(self):
        # Movement between frames must land in the `speeds` field (not a dropped
        # `speed` kwarg that pydantic silently ignores).
        service = PoseFeatureService()
        prev = kc(left_wrist=(0.5, 0.5), left_elbow=(0.5, 0.6))
        curr = kc(left_wrist=(0.6, 0.5), left_elbow=(0.5, 0.6))
        features = service.process_features(curr, previous_keypoints=prev)
        assert features.speeds
        assert "left_wrist" in features.speeds


class TestCalculateAngleProperties:
    @given(a=_pt, b=_pt, c=_pt)
    def test_angle_is_bounded_and_symmetric(self, a, b, c):
        service = PoseFeatureService()
        assume(np.linalg.norm(np.subtract(a, b)) > 1.0)
        assume(np.linalg.norm(np.subtract(c, b)) > 1.0)
        forward = service.calculate_angle(a, b, c)
        reverse = service.calculate_angle(c, b, a)
        assert -1e-6 <= forward <= 180.0 + 1e-6
        assert forward == pytest.approx(reverse, abs=1e-6)

    @given(a=_pt, b=_pt, c=_pt, tx=_coord, ty=_coord)
    def test_angle_is_translation_invariant(self, a, b, c, tx, ty):
        service = PoseFeatureService()
        assume(np.linalg.norm(np.subtract(a, b)) > 1.0)
        assume(np.linalg.norm(np.subtract(c, b)) > 1.0)
        base = service.calculate_angle(a, b, c)
        shifted = service.calculate_angle(
            (a[0] + tx, a[1] + ty), (b[0] + tx, b[1] + ty), (c[0] + tx, c[1] + ty)
        )
        assert base == pytest.approx(shifted, abs=1e-2)

    @given(delta=st.floats(min_value=0.0, max_value=0.5, allow_nan=False))
    def test_joint_angles_are_invariant_to_confidence(self, delta):
        # Generalizes finding #2: shifting every keypoint's confidence by a
        # constant must not change the computed geometric angle.
        service = PoseFeatureService()
        base = kc(
            left_shoulder=(0.5, 0.5, 0.3),
            left_elbow=(0.5, 0.7, 0.4),
            left_wrist=(0.7, 0.7, 0.5),
        )
        shifted = kc(
            left_shoulder=(0.5, 0.5, 0.3 + delta),
            left_elbow=(0.5, 0.7, 0.4 + delta),
            left_wrist=(0.7, 0.7, 0.5 + delta),
        )
        key = "left_shoulder_left_elbow_left_wrist"
        assert service.extract_joint_angles(base)[key] == pytest.approx(
            service.extract_joint_angles(shifted)[key], abs=1e-6
        )
