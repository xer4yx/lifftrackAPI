from abc import ABC, abstractmethod
from typing import Any, Dict

from core.entities import BodyAlignment, FeatureMetrics


class FeatureMetricRepositoryInterface(ABC):
    """
    Abstract interface for feature metric repositories.
    Defines the contract that all feature metric repository implementations must fulfill.
    """

    @abstractmethod
    def compute_ba_score(
        self, body_alignment: BodyAlignment, max_allowed_devition: int
    ) -> float:
        """
        Measures how well the lifter maintains proper posture. Compute the score based on the given body alignment.

        Args:
            body_alignment: The body alignment to compute the score for.
            max_allowed_devition: The maximum allowed deviation from the ideal body alignment.

        Returns:
            The body alignment score.
        """
        pass

    @abstractmethod
    def compute_jc_score(
        self, joint_angles: Dict[str, float], max_allowed_variance: int
    ) -> float:
        """
        Evaluates whether the lifter maintains optimal joint angles throughout the movement.
        Compute the joint consistency score based on the joint angles.

        Args:
            joint_angles: The joint angles to compute the score for.
            max_allowed_variance: The maximum allowed variance in joint angles.

        Returns:
            The joint consistency score.
        """
        pass

    @abstractmethod
    def compute_lc_score(
        self, objects: Dict[str, Any], max_allowed_variance: int
    ) -> float:
        """
        Measures how steady the dumbbell/barbell is during the lift.
        Compute the load control score based on the object stability.

        Args:
            objects: Dictionary containing object detection data.
            max_allowed_variance: The maximum allowed variance in the object stability.

        Returns:
            The load control score.
        """
        pass

    @abstractmethod
    def compute_sc_score(self, speeds: Dict[str, float], max_jerk: float) -> float:
        """
        Checks for controlled movement, avoiding excessive acceleration/deceleration.
        Compute the speed control score based on the speeds.

        Args:
            speeds: The speeds to compute the score for.
            max_jerk: The maximum allowed acceleration/deceleration.

        Returns:
            The speed control score.
        """
        pass

    @abstractmethod
    def compute_os_score(self, stability_raw: float, max_displacement: float) -> float:
        """
        Combines balance and center-of-mass tracking.
        Compute the overall stability score based on the stabilities.

        Args:
            stability_raw: The raw stability score.
            max_displacement: The maximum allowed displacement of the center-of-mass.

        Returns:
            The overall stability score.
        """
        pass

    @abstractmethod
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

        Args:
            body_alignment: The body alignment data
            joint_angles: Dictionary of joint angles
            objects: Dictionary of object detection data
            speeds: Dictionary of speed measurements
            stability_raw: Raw stability measurement
            max_allowed_deviation: Maximum allowed body alignment deviation
            max_allowed_variance: Maximum allowed variance for joints and objects
            max_jerk: Maximum allowed acceleration/deceleration
            max_displacement: Maximum allowed displacement for stability

        Returns:
            FeatureMetrics object containing all computed scores
        """
        pass

    @abstractmethod
    def reset_history(self) -> None:
        """Reset frame history for new exercise session."""
        pass
