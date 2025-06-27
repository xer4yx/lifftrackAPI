from typing import Any, Dict

from core.interface.feature_metric_repository_interface import (
    FeatureMetricRepositoryInterface,
)
from core.entities import BodyAlignment, FeatureMetrics
from core.service import FeatureMetricService


class FeatureMetricRepository(FeatureMetricRepositoryInterface):
    """
    Implementation of the FeatureMetricRepositoryInterface that handles feature metric calculations.
    This repository delegates to the FeatureMetricService to avoid duplicating business logic.
    """

    def __init__(self):
        """Initialize the repository with a service instance."""
        self._service = FeatureMetricService()

    def compute_ba_score(
        self, body_alignment: BodyAlignment, max_allowed_devition: int
    ) -> float:
        """Delegate to service implementation."""
        return self._service.compute_ba_score(body_alignment, max_allowed_devition)

    def compute_jc_score(
        self, joint_angles: Dict[str, float], max_allowed_variance: int
    ) -> float:
        """Delegate to service implementation."""
        return self._service.compute_jc_score(joint_angles, max_allowed_variance)

    def compute_lc_score(
        self, objects: Dict[str, Any], max_allowed_variance: int
    ) -> float:
        """Delegate to service implementation."""
        return self._service.compute_lc_score(objects, max_allowed_variance)

    def compute_sc_score(self, speeds: Dict[str, float], max_jerk: float) -> float:
        """Delegate to service implementation."""
        return self._service.compute_sc_score(speeds, max_jerk)

    def compute_os_score(self, stability_raw: float, max_displacement: float) -> float:
        """Delegate to service implementation."""
        return self._service.compute_os_score(stability_raw, max_displacement)

    def compute_all_metrics(
        self,
        body_alignment: BodyAlignment,
        joint_angles: Dict[str, float],
        objects: Dict[str, Any],
        speeds: Dict[str, float],
        stability_raw: float,
        max_allowed_deviation: int = 10,
        max_allowed_variance: int = 15,
        max_jerk: float = 5.0,
        max_displacement: float = 20.0,
    ) -> Dict[str, float]:
        """
        Compute all feature metrics in one call for efficiency.

        This method combines all individual metric calculations into a single operation,
        reducing the overhead of multiple service calls.
        """
        ba_score = self._service.compute_ba_score(body_alignment, max_allowed_deviation)
        jc_score = self._service.compute_jc_score(joint_angles, max_allowed_variance)
        lc_score = self._service.compute_lc_score(objects, max_allowed_variance)
        sc_score = self._service.compute_sc_score(speeds, max_jerk)
        os_score = self._service.compute_os_score(stability_raw, max_displacement)

        return {
            "body_alignment": ba_score,
            "joint_consistency": jc_score,
            "load_control": lc_score,
            "speed_control": sc_score,
            "overall_stability": os_score,
        }

    def reset_history(self) -> None:
        """Reset frame history for new exercise session."""
        self._service.reset_history()
