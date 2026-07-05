from lifttrack.utils.logging_config import setup_logger
from core.interface.form_analysis_interface import FormAnalysisInterface
from core.entities.pose_entity import PoseFeatures, FormAnalysis
from core.service.angle_range_table import get_angle_ranges, score_form

# Setup logging
logger = setup_logger("form-analysis-service", "comvis.log")

_GOOD_FORM = "Form looks good! Keep it up!"
_UNKNOWN = "Exercise type not recognized for form analysis."


class FormAnalysisService(FormAnalysisInterface):
    """Form analysis driven entirely by the per-exercise angle-range table
    (ADR 0005), which is the single source of truth for the heatmap bands, the
    form score, and the coaching suggestions.

    Body-alignment and stability are body-level signals scored separately by the
    feature-metric layer (compute_ba_score / compute_os_score, tuned by
    EXERCISE_THRESHOLDS), so they are no longer duplicated here.
    """

    def _analyze(self, features: PoseFeatures, exercise_name: str) -> FormAnalysis:
        # A detected resting state overrides form scoring.
        resting = {}
        if isinstance(features.objects, dict):
            resting = features.objects.get("resting_state", {}) or {}
        if resting.get("is_resting", False):
            logger.info(
                f"User is resting - Position: {resting.get('position', 'unknown')}"
            )
            return FormAnalysis(accuracy=0.0, suggestions=["Idling"])

        if not get_angle_ranges(exercise_name):
            logger.warning(f"Unknown exercise type: {exercise_name}")
            return FormAnalysis(accuracy=1.0, suggestions=[_UNKNOWN])

        accuracy, suggestions = score_form(features.joint_angles, exercise_name)
        return FormAnalysis(accuracy=accuracy, suggestions=suggestions or [_GOOD_FORM])

    def analyze_bench_press_form(
        self, features: PoseFeatures, exercise_name: str
    ) -> FormAnalysis:
        return self._analyze(features, "bench_press")

    def analyze_deadlift_form(
        self, features: PoseFeatures, exercise_name: str
    ) -> FormAnalysis:
        return self._analyze(features, "deadlift")

    def analyze_rdl_form(
        self, features: PoseFeatures, exercise_name: str
    ) -> FormAnalysis:
        return self._analyze(features, "rdl")

    def analyze_shoulder_press_form(
        self, features: PoseFeatures, exercise_name: str
    ) -> FormAnalysis:
        return self._analyze(features, "shoulder_press")

    def analyze_form(self, features: PoseFeatures, exercise_name: str) -> FormAnalysis:
        return self._analyze(features, exercise_name)
