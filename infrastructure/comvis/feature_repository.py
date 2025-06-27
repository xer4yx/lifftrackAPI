from fastapi import Request
import numpy as np
import time
from typing import Any, Dict, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor

from core.entities import Object, PoseFeatures, KeypointCollection
from core.interface import FeatureRepositoryInterface
from lifttrack.utils.logging_config import setup_logger

# Import features functions by name to avoid circular imports
from lifttrack.v2.comvis.features import (
    extract_joint_angles,
    extract_movement_patterns,
    calculate_speed,
    extract_body_alignment,
    calculate_stability,
)
from lifttrack.v2.comvis.progress import calculate_form_accuracy

logger = setup_logger("feature-repository", "feature_repository.log")


class FeatureRepository(FeatureRepositoryInterface):
    """
    Repository for handling feature extraction operations.
    This repository is responsible for extracting features from frames and poses.
    """

    def perform_frame_analysis(
        self, frames_buffer: List[np.ndarray], request: Request
    ) -> Tuple[Dict, Dict, List, str]:
        """
        Perform parallel frame analysis on the input frames.

        Args:
            frames_buffer: List of frame buffers

        Returns:
            Tuple of (current_pose, previous_pose, detected_object, class_name)

        Raises:
            Exception: If frame analysis fails
        """
        try:
            if len(frames_buffer) < 2:
                return None, None, None, None

            start_time = time.time()

            if not hasattr(request.app.state, "inference_services"):
                raise ValueError("Inference services not found in request")

            services = request.app.state.inference_services
            videoaction_service = services.get("videoaction")
            posenet_service = services.get("posenet")
            roboflow_service = services.get("roboflow")

            # Run models in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks to the executor using the appropriate service methods
                class_name_future = executor.submit(
                    videoaction_service.predict_class, frames_buffer
                )
                current_pose_future = executor.submit(
                    posenet_service.infer, frames_buffer[-1]
                )
                detected_object_future = executor.submit(
                    roboflow_service.infer, frames_buffer[-1]
                )
                previous_pose_future = executor.submit(
                    posenet_service.infer, frames_buffer[-2]
                )

                # Get results from futures
                class_name = class_name_future.result()
                _, current_pose = current_pose_future.result()
                detected_object = detected_object_future.result()
                _, previous_pose = previous_pose_future.result()

            # Log processing time
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            logger.debug(
                f"Frame processing time: {processing_time:.3f}s ({fps:.1f} FPS)"
            )

            return current_pose, previous_pose, detected_object, class_name

        except Exception as e:
            logger.error(f"Failed to perform frame analysis: {str(e)}")
            raise

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
            _features = features.model_dump()
            if not isinstance(_features, dict):
                logger.error("features must be a dictionary")
                raise TypeError("features must be a dictionary")

            # Normalize the class_name to match the format expected by calculate_form_accuracy
            normalized_class_name = class_name.lower().replace(" ", "_")

            accuracy, suggestions = calculate_form_accuracy(
                _features, normalized_class_name
            )
            logger.info(f"Form accuracy: {accuracy}, Suggestions: {suggestions}")
            # Join suggestions list into a single string, or return a default message if empty
            return (
                accuracy,
                (
                    " ".join(suggestions)
                    if suggestions
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
