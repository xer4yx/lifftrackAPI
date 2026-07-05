from fastapi import Request
import numpy as np
import time
from typing import Any, Dict, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor

from core.entities import Object, PoseFeatures, KeypointCollection
from core.interface import FeatureRepositoryInterface
from lifttrack.utils.logging_config import setup_logger

logger = setup_logger("feature-repository", "feature_repository.log")


class FeatureRepository(FeatureRepositoryInterface):
    """
    Repository for handling feature extraction operations.
    This repository is responsible for extracting features from frames and poses.
    """

    def perform_frame_analysis(
        self, frames_buffer: List[np.ndarray], request: Request
    ) -> Tuple[Dict, Dict, List, str]:
        """Removed: the server no longer runs pose on pixel frames.

        The keypoints-in cutover (ADR 0001/0002/0009) moved pose estimation
        on-device; the live path now scores client-supplied keypoints via
        ComVisUseCase.process_keypoints. This frames-in method is retained only
        to satisfy the interface and is no longer reachable.
        """
        raise NotImplementedError(
            "Server-side frame analysis was removed; the client streams keypoints."
        )

    def load_to_object_model(self, object_inference: List[Dict]) -> Object:
        """
        Load an object inference to an Object model.

        Args:
            object_inference: Object inference data

        Returns:
            Object model

        Raises:
            Exception: If loading fails
        """
        try:
            # Handle empty inference case
            if not object_inference:
                return Object(
                    classs_id=-1,
                    type="unknown",
                    confidence=0.0,
                    x=0.0,
                    y=0.0,
                    width=0.0,
                    height=0.0,
                )

            best_confidence = max(
                object_inference, key=lambda x: x.get("confidence", 0)
            )
            return Object(
                classs_id=best_confidence.get("class_id", 0),
                type=best_confidence.get("class", "barbell"),  # Provide default value
                confidence=best_confidence.get("confidence", 0.0),
                x=best_confidence.get("x", 0.0),
                y=best_confidence.get("y", 0.0),
                width=best_confidence.get("width", 0.0),
                height=best_confidence.get("height", 0.0),
            )
        except Exception as e:
            logger.error(f"Failed to load object inference: {str(e)}")
            raise

    def load_to_features_model(
        self,
        previous_pose: Dict,
        current_pose: Dict,
        object_inference: Object,
        class_name: str,
    ) -> PoseFeatures:
        """
        Save features in a PoseFeatures model.

        Args:
            previous_pose: Previous pose data
            current_pose: Current pose data
            object_inference: Object inference data
            class_name: Exercise class name

        Returns:
            PoseFeatures model

        Raises:
            Exception: If loading fails
        """
        try:
            if not isinstance(current_pose, dict):
                raise TypeError("current_pose must be a dictionary")
            if not isinstance(previous_pose, dict):
                raise TypeError("previous_pose must be a dictionary")
            if not isinstance(object_inference, Object):
                raise TypeError("object_inference must be an Object base model")

            # Convert object inference to dict for compatibility
            object_inference_dict = object_inference.model_dump()

            # Convert pose dictionaries to KeypointCollection objects
            def dict_to_keypoint_collection(pose_dict: Dict) -> KeypointCollection:
                """Convert pose dictionary to KeypointCollection."""
                from core.entities.pose_entity import Keypoint, KeypointCollection

                keypoints = {}
                for joint_name, (x, y, confidence) in pose_dict.items():
                    keypoints[joint_name] = Keypoint(
                        x=float(x), y=float(y), confidence=float(confidence)
                    )
                return KeypointCollection(keypoints=keypoints)

            current_keypoints = dict_to_keypoint_collection(current_pose)
            previous_keypoints = (
                dict_to_keypoint_collection(previous_pose) if previous_pose else None
            )

            # Use the service-based feature extraction for better compatibility
            from core.service.pose_feature_service import PoseFeatureService

            pose_service = PoseFeatureService()

            # Extract features using the service
            joint_angles = pose_service.extract_joint_angles(current_keypoints)

            # Initialize movement-based features
            movement_patterns = {}
            speeds = {}
            stability = 0.0

            # Extract movement patterns and speeds if we have previous keypoints
            if previous_keypoints and previous_keypoints.keypoints:
                movement_patterns = pose_service.extract_movement_patterns(
                    current_keypoints, previous_keypoints
                )
                speeds = pose_service.calculate_speed(movement_patterns)
                stability = pose_service.calculate_stability(
                    current_keypoints, previous_keypoints
                )

            # Extract body alignment
            vertical_alignment, lateral_alignment = pose_service.extract_body_alignment(
                current_keypoints
            )
            from core.entities.pose_entity import BodyAlignment

            body_alignment = BodyAlignment(
                vertical_alignment=vertical_alignment,
                lateral_alignment=lateral_alignment,
            )

            # Create initial PoseFeatures object
            features = PoseFeatures(
                keypoints=current_keypoints,
                objects=(
                    object_inference_dict
                    if isinstance(object_inference_dict, dict)
                    else {}
                ),
                joint_angles=joint_angles,
                movement_patterns=movement_patterns,
                movement_pattern=class_name,
                speeds=speeds,
                body_alignment=body_alignment,
                stability=stability,
            )

            # Detect form issues using the created features
            form_issues = pose_service.detect_form_issues(features, class_name)
            features.form_issues = form_issues

            return features
        except Exception as e:
            logger.error(f"Failed to load features: {str(e)}")
            raise

    def get_suggestions(
        self, features: PoseFeatures, class_name: str
    ) -> Tuple[float, str]:
        """
        Get suggestions for a given class name and features.

        Args:
            features: PoseFeatures model
            class_name: Exercise class name

        Returns:
            Tuple of (accuracy, suggestions)
        """
        try:
            # Use the canonical clean-architecture form analysis service (replaces the
            # legacy lifttrack.v2.comvis.progress.calculate_form_accuracy). It reads the
            # PoseFeatures object directly and normalizes the exercise name internally.
            from core.service.form_analysis_service import FormAnalysisService

            analysis = FormAnalysisService().analyze_form(features, class_name)
            logger.info(
                f"Form accuracy: {analysis.accuracy}, Suggestions: {analysis.suggestions}"
            )
            # Join suggestions list into a single string, or return a default message if empty
            return (
                analysis.accuracy,
                (
                    " ".join(analysis.suggestions)
                    if analysis.suggestions
                    else "Form looks good! Keep it up!"
                ),
            )
        except Exception as e:
            logger.error(f"Failed to get suggestions: {str(e)}", exc_info=True)
            # Return a default message instead of None when an exception occurs
            return (
                0.0,
                "Unable to analyze form at this time. Please continue your exercise.",
            )
