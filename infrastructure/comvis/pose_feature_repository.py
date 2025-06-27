import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import cv2
import concurrent.futures

from core.interface import PoseFeatureInterface
from core.entities import Keypoint, KeypointCollection, BodyAlignment, PoseFeatures
from core.service.pose_feature_service import PoseFeatureService


class PoseFeatureRepository(PoseFeatureInterface):
    """
    Implementation of the PoseFeatureInterface that handles keypoint-based feature extraction.
    This repository is responsible for extracting features from keypoints detected in images.
    """

    def __init__(self):
        # Use the service implementation to avoid duplicating code
        self._service = PoseFeatureService()

    def calculate_angle(
        self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
    ) -> float:
        """Delegate to service implementation."""
        return self._service.calculate_angle(a, b, c)

    def extract_joint_angles(
        self, keypoints: KeypointCollection, confidence_threshold: float = 0.1
    ) -> Dict[str, float]:
        """Delegate to service implementation."""
        return self._service.extract_joint_angles(keypoints, confidence_threshold)

    def extract_movement_patterns(
        self, keypoints: KeypointCollection, previous_keypoints: KeypointCollection
    ) -> Dict[str, float]:
        """Delegate to service implementation."""
        return self._service.extract_movement_patterns(keypoints, previous_keypoints)

    def calculate_speed(
        self, displacement: Dict[str, float], time_delta: float = 1.0
    ) -> Dict[str, float]:
        """Delegate to service implementation."""
        return self._service.calculate_speed(displacement, time_delta)

    def extract_body_alignment(
        self, keypoints: KeypointCollection
    ) -> Tuple[float, float]:
        """Delegate to service implementation."""
        return self._service.extract_body_alignment(keypoints)

    def calculate_stability(
        self,
        keypoints: KeypointCollection,
        previous_keypoints: KeypointCollection,
        window_size: int = 5,
    ) -> float:
        """Delegate to service implementation."""
        return self._service.calculate_stability(
            keypoints, previous_keypoints, window_size
        )

    def detect_form_issues(
        self, features: PoseFeatures, exercise_type: str
    ) -> Dict[str, bool]:
        """Delegate to service implementation."""
        return self._service.detect_form_issues(features, exercise_type)

    def detect_resting_state(
        self, keypoints: KeypointCollection, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delegate to service implementation."""
        return self._service.detect_resting_state(keypoints, features)

    def process_features(
        self,
        keypoints: KeypointCollection,
        previous_keypoints: Optional[KeypointCollection] = None,
        objects: Dict[str, Any] = None,
    ) -> PoseFeatures:
        """Delegate to service implementation."""
        return self._service.process_features(keypoints, previous_keypoints, objects)

    @staticmethod
    def convert_to_keypoint_collection(
        keypoints_dict: Dict[str, Tuple[float, float, float]],
    ) -> KeypointCollection:
        """
        Convert the dictionary format keypoints to a KeypointCollection object.

        Args:
            keypoints_dict: Dictionary mapping joint names to (x, y, confidence) tuples

        Returns:
            KeypointCollection object
        """
        keypoints = {}
        for joint_name, (x, y, conf) in keypoints_dict.items():
            keypoints[joint_name] = Keypoint(
                x=float(x), y=float(y), confidence=float(conf)
            )

        return KeypointCollection(keypoints=keypoints)

    def visualize_keypoints(
        self, frame: np.ndarray, keypoints: KeypointCollection, threshold: float = 0.5
    ) -> np.ndarray:
        """
        Visualize keypoints on a frame.

        Args:
            frame: The input image frame
            keypoints: KeypointCollection object containing the keypoints
            threshold: Confidence threshold for keypoint visualization

        Returns:
            Frame with keypoints visualized
        """
        frame_copy = frame.copy()
        for joint_name, keypoint in keypoints.keypoints.items():
            if keypoint.confidence > threshold:
                x, y = int(keypoint.x), int(keypoint.y)
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    frame_copy,
                    joint_name,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        return frame_copy

    def visualize_angles(
        self,
        frame: np.ndarray,
        keypoints: KeypointCollection,
        angles: Dict[str, float],
        confidence_threshold: float = 0.3,
    ) -> np.ndarray:
        """
        Visualize calculated angles on the frame for validation purposes.

        Args:
            frame: The original image/video frame
            keypoints: KeypointCollection with keypoint positions
            angles: Dictionary of calculated angles
            confidence_threshold: Minimum confidence to display

        Returns:
            Frame with angles visualized
        """
        # Create a copy of the frame
        annotated_frame = frame.copy()

        # Draw keypoints first
        for name, keypoint in keypoints.keypoints.items():
            if keypoint.confidence > confidence_threshold:
                cv2.circle(
                    annotated_frame,
                    (int(keypoint.x), int(keypoint.y)),
                    4,
                    (0, 255, 0),
                    -1,
                )

        # Draw angles
        for angle_name, angle_value in angles.items():
            # Parse the joint names from the angle name
            joint_names = angle_name.split("_")
            if len(joint_names) >= 3:
                middle_joint = joint_names[
                    1
                ]  # The middle joint is where the angle is measured

                keypoint = keypoints.get(middle_joint)
                if keypoint and keypoint.confidence > confidence_threshold:
                    x, y = int(keypoint.x), int(keypoint.y)
                    # Display the angle value near the middle joint
                    cv2.putText(
                        annotated_frame,
                        f"{angle_value:.1f}°",
                        (x + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

        return annotated_frame
