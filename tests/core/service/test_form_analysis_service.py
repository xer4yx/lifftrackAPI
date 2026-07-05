import pytest

from core.entities.pose_entity import KeypointCollection, PoseFeatures
from core.service.form_analysis_service import FormAnalysisService


def _features(joint_angles, objects=None):
    return PoseFeatures(
        keypoints=KeypointCollection(keypoints={}),
        joint_angles=joint_angles,
        objects=objects or {},
    )


_BAD_BENCH = {
    "left_shoulder_left_elbow_left_wrist": 130.0,
    "right_shoulder_right_elbow_right_wrist": 130.0,
}
_GOOD_BENCH = {
    "left_shoulder_left_elbow_left_wrist": 90.0,
    "right_shoulder_right_elbow_right_wrist": 90.0,
}


class TestTableDrivenAnalyzeForm:
    def test_bad_joint_angle_lowers_accuracy_and_suggests_cue(self):
        svc = FormAnalysisService()
        result = svc.analyze_form(_features(_BAD_BENCH), "bench_press")
        assert result.accuracy < 1.0
        assert any("elbow" in s.lower() for s in result.suggestions)

    def test_good_form_is_full_accuracy(self):
        svc = FormAnalysisService()
        result = svc.analyze_form(_features(_GOOD_BENCH), "bench_press")
        assert result.accuracy == pytest.approx(1.0)

    def test_resting_state_is_idling(self):
        svc = FormAnalysisService()
        result = svc.analyze_form(
            _features({}, objects={"resting_state": {"is_resting": True}}),
            "bench_press",
        )
        assert result.accuracy == 0.0
        assert result.suggestions == ["Idling"]

    def test_unknown_exercise_is_not_scored(self):
        svc = FormAnalysisService()
        result = svc.analyze_form(_features({"x": 1.0}), "zumba")
        assert "not recognized" in result.suggestions[0].lower()

    def test_per_exercise_method_matches_analyze_form(self):
        svc = FormAnalysisService()
        f = _features(_BAD_BENCH)
        assert svc.analyze_bench_press_form(f, "bench_press") == svc.analyze_form(
            f, "bench_press"
        )
