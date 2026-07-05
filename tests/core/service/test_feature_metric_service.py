import pytest

from core.entities.pose_entity import Keypoint, KeypointCollection, BodyAlignment
from core.service.feature_metric_service import FeatureMetricService
from core.service.pose_feature_service import PoseFeatureService


def _ba(vertical, lateral):
    return BodyAlignment(vertical_alignment=vertical, lateral_alignment=lateral)


class TestBodyAlignmentScore:
    """ADR 0013 guardrail: reintroduced/relied-on metrics must prove they produce
    non-trivial, discriminating scores (the load_control-always-0 lesson)."""

    def test_better_alignment_scores_higher(self):
        service = FeatureMetricService()
        good = service.compute_ba_score(_ba(1.0, 1.0), 10)
        bad = service.compute_ba_score(_ba(40.0, 30.0), 10)
        assert good > bad

    def test_score_is_not_a_degenerate_constant(self):
        service = FeatureMetricService()
        scores = {
            service.compute_ba_score(_ba(v, v), 10)
            for v in (0.0, 5.0, 15.0, 45.0)
        }
        assert len(scores) > 1  # actually varies with input
        assert scores != {0.0}
        assert scores != {100.0}

    def test_upright_torso_scores_well_end_to_end(self):
        # Ties finding #1 to the metric: an upright torso must produce a HIGH body
        # alignment score. Before the vertical-reference fix it read ~180 deg and
        # scored ~0.
        pose = PoseFeatureService()
        vertical, lateral = pose.extract_body_alignment(
            KeypointCollection(
                keypoints={
                    "left_shoulder": Keypoint(x=0.4, y=0.3, confidence=1.0),
                    "right_shoulder": Keypoint(x=0.6, y=0.3, confidence=1.0),
                    "left_hip": Keypoint(x=0.4, y=0.6, confidence=1.0),
                    "right_hip": Keypoint(x=0.6, y=0.6, confidence=1.0),
                }
            )
        )
        score = FeatureMetricService().compute_ba_score(_ba(vertical, lateral), 10)
        assert score > 90.0
