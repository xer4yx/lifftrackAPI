import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

from core.interface import FeatureMetricInterface
from core.entities import BodyAlignment, Object


class FeatureMetricService(FeatureMetricInterface):
    def __init__(self):
        """Initialize the service with frame history for joint consistency tracking."""
        # Store last 5 frames of joint angles for consistency calculation
        self._joint_angle_history: deque = deque(maxlen=5)
        # Store last 5 frames of object positions for load control
        self._object_history: deque = deque(maxlen=5)
        # Store last 5 frames of speeds for speed control analysis
        self._speed_history: deque = deque(maxlen=5)
        # Store last 10 frames of stability values for moving average
        self._stability_history: deque = deque(maxlen=10)

    def compute_ba_score(
        self, body_alignment: BodyAlignment, max_allowed_devition: int
    ) -> float:
        """
        Compute body alignment score based on vertical and lateral alignment deviations.
        Uses weighted sum of normalized scores for vertical and lateral alignment.

        Args:
            body_alignment: BodyAlignment entity containing vertical and lateral alignment values
            max_allowed_devition: Maximum allowed deviation from ideal alignment

        Returns:
            Body alignment score between 0-100
        """
        if not body_alignment or max_allowed_devition <= 0:
            return 0.0

        try:
            # Get absolute deviations (assuming 0 is ideal alignment)
            vertical_deviation = abs(body_alignment.vertical_alignment)
            lateral_deviation = abs(body_alignment.lateral_alignment)

            # Normalize scores (higher deviation = lower score)
            vertical_score = max(
                0, 100 - (vertical_deviation / max_allowed_devition) * 100
            )
            lateral_score = max(
                0, 100 - (lateral_deviation / max_allowed_devition) * 100
            )

            # Apply weighted sum (vertical alignment typically more important)
            vertical_weight = 0.6
            lateral_weight = 0.4

            final_score = (vertical_score * vertical_weight) + (
                lateral_score * lateral_weight
            )
            return min(100.0, max(0.0, final_score))

        except (AttributeError, ZeroDivisionError, TypeError) as e:
            return 0.0

    def compute_jc_score(
        self, joint_angles: Dict[str, float], max_allowed_variance: int
    ) -> float:
        """
        Compute joint consistency score based on standard deviation of joint angles over last 5 frames.
        Lower deviations result in higher scores.

        Args:
            joint_angles: Dictionary of joint names to angle values
            max_allowed_variance: Maximum allowed variance in joint angles

        Returns:
            Joint consistency score between 0-100
        """
        if not joint_angles or max_allowed_variance <= 0:
            return 0.0

        try:
            # Add current frame to history
            self._joint_angle_history.append(joint_angles.copy())

            # Need at least 2 frames to compute standard deviation
            if len(self._joint_angle_history) < 2:
                return 100.0  # Perfect score for first frame

            # Compute standard deviation for each joint across frames
            joint_scores = []

            for joint_name in joint_angles.keys():
                # Extract angle values for this joint across all frames
                angle_values = []
                for frame_angles in self._joint_angle_history:
                    if joint_name in frame_angles:
                        angle_values.append(frame_angles[joint_name])

                if len(angle_values) >= 2:
                    # Compute standard deviation
                    std_dev = np.std(angle_values)

                    # Normalize to 0-100 score (lower std_dev = higher score)
                    score = max(0, 100 - (std_dev / max_allowed_variance) * 100)
                    joint_scores.append(score)

            # Return average score across all joints
            return np.mean(joint_scores) if joint_scores else 0.0

        except (ValueError, TypeError, ZeroDivisionError) as e:
            return 0.0

    def compute_lc_score(
        self, objects: Dict[str, Any], max_allowed_variance: int
    ) -> float:
        """
        Compute load control score using Short-Term Wobble Detection.
        Analyzes object position stability across frames.

        Args:
            objects: Dictionary containing object detection data
            max_allowed_variance: Maximum allowed variance in object positions

        Returns:
            Load control score between 0-100
        """
        if not objects or max_allowed_variance <= 0:
            return 0.0

        try:
            # Convert objects to Object entities and extract positions
            current_positions = {}
            for obj_id, obj_data in objects.items():
                if isinstance(obj_data, dict):
                    # Extract center position
                    x = obj_data.get("x", 0) + obj_data.get("width", 0) / 2
                    y = obj_data.get("y", 0) + obj_data.get("height", 0) / 2
                    current_positions[obj_id] = {"x": x, "y": y}
                elif hasattr(obj_data, "x") and hasattr(obj_data, "y"):
                    # Object entity
                    x = obj_data.x + obj_data.width / 2
                    y = obj_data.y + obj_data.height / 2
                    current_positions[obj_id] = {"x": x, "y": y}

            # Add current frame to object history
            self._object_history.append(current_positions.copy())

            # Need at least 2 frames for wobble detection
            if len(self._object_history) < 2:
                return 100.0  # Perfect score for first frame

            # Compute frame-to-frame positional variance for each object
            object_scores = []

            for obj_id in current_positions.keys():
                # Extract position history for this object
                x_positions = []
                y_positions = []

                for frame_objects in self._object_history:
                    if obj_id in frame_objects:
                        x_positions.append(frame_objects[obj_id]["x"])
                        y_positions.append(frame_objects[obj_id]["y"])

                if len(x_positions) >= 2:
                    # Compute positional variance (wobble)
                    x_variance = np.var(x_positions)
                    y_variance = np.var(y_positions)
                    total_variance = x_variance + y_variance

                    # Normalize to 0-100 score (lower variance = higher score)
                    score = max(0, 100 - (total_variance / max_allowed_variance) * 100)
                    object_scores.append(score)

            # Return average score across all objects
            return np.mean(object_scores) if object_scores else 0.0

        except (ValueError, TypeError, ZeroDivisionError, KeyError) as e:
            return 0.0

    def compute_sc_score(self, speeds: Dict[str, float], max_jerk: float) -> float:
        """
        Compute speed control score based on speed smoothness and acceleration control.
        Uses frame history to calculate speed standard deviation and frame-to-frame acceleration.

        Args:
            speeds: Dictionary of speed measurements
            max_jerk: Maximum allowed acceleration/deceleration

        Returns:
            Speed control score between 0-100
        """
        if not speeds or max_jerk <= 0:
            return 0.0

        try:
            # Add current frame speeds to history
            self._speed_history.append(speeds.copy())

            # Need at least 2 frames for acceleration analysis
            if len(self._speed_history) < 2:
                return 100.0  # Perfect score for first frame

            # Calculate smoothness score using standard deviation across frames
            smoothness_scores = []
            acceleration_scores = []

            for speed_key in speeds.keys():
                # Extract speed values for this measurement across frames
                speed_values = []
                for frame_speeds in self._speed_history:
                    if speed_key in frame_speeds:
                        speed_values.append(frame_speeds[speed_key])

                if len(speed_values) >= 2:
                    # 1. Speed smoothness (using standard deviation)
                    speed_std = np.std(speed_values)
                    smoothness_score = max(0, 100 - (speed_std / max_jerk) * 100)
                    smoothness_scores.append(smoothness_score)

                    # 2. Frame-to-frame acceleration analysis
                    accelerations = []
                    for i in range(1, len(speed_values)):
                        acceleration = abs(speed_values[i] - speed_values[i - 1])
                        accelerations.append(acceleration)

                    if accelerations:
                        # Penalize excessive acceleration/deceleration
                        max_acceleration = max(accelerations)
                        if max_acceleration > max_jerk:
                            accel_penalty = (
                                (max_acceleration - max_jerk) / max_jerk * 50
                            )
                            accel_score = max(0, 100 - accel_penalty)
                        else:
                            accel_score = 100.0
                        acceleration_scores.append(accel_score)

            # Combine smoothness and acceleration scores
            smoothness_avg = np.mean(smoothness_scores) if smoothness_scores else 100.0
            acceleration_avg = (
                np.mean(acceleration_scores) if acceleration_scores else 100.0
            )

            # Weighted combination (60% smoothness, 40% acceleration control)
            final_score = (smoothness_avg * 0.6) + (acceleration_avg * 0.4)

            return min(100.0, max(0.0, final_score))

        except (ValueError, TypeError, ZeroDivisionError) as e:
            return 0.0

    def compute_os_score(self, stability_raw: float, max_displacement: float) -> float:
        """
        Compute overall stability score using moving average to smooth the stability score.

        Args:
            stability_raw: Raw stability measurement
            max_displacement: Maximum allowed displacement of center-of-mass

        Returns:
            Overall stability score between 0-100
        """
        if max_displacement <= 0:
            return 0.0

        try:
            # Add current stability to history
            self._stability_history.append(stability_raw)

            # Use moving average for smoother stability scoring
            if len(self._stability_history) == 0:
                return 0.0

            # Calculate moving average of stability
            avg_stability = np.mean(list(self._stability_history))

            # Handle negative stability values
            if avg_stability < 0:
                return 0.0

            # Convert displacement to stability score (lower displacement = higher score)
            # Using inverse relationship: higher stability = lower displacement
            score = max(0, 100 - (avg_stability / max_displacement) * 100)

            return min(100.0, max(0.0, score))

        except (ValueError, TypeError, ZeroDivisionError) as e:
            return 0.0

    def reset_history(self) -> None:
        """Reset frame history for new exercise session."""
        self._joint_angle_history.clear()
        self._object_history.clear()
        self._speed_history.clear()
        self._stability_history.clear()
