from concurrent.futures import ThreadPoolExecutor
from fastapi import Request
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from core.entities.pose_entity import KeypointCollection, PoseFeatures, FormAnalysis
from core.entities import Object, Keypoint, BodyAlignment, FeatureMetrics
from core.interface import (
    PoseFeatureInterface,
    FormAnalysisInterface,
    DataHandlerInterface,
    FrameRepositoryInterface,
    FeatureRepositoryInterface,
    FeatureMetricRepositoryInterface,
)


class ComVisUseCase:
    """
    Use case for computer vision functionality.
    Handles the extraction of pose features and form analysis for exercises.
    """

    def __init__(
        self,
        pose_feature_repository: PoseFeatureInterface,
        form_analysis_repository: FormAnalysisInterface,
        frame_repository: FrameRepositoryInterface,
        feature_repository: FeatureRepositoryInterface,
        data_handler: DataHandlerInterface,
    ):
        """
        Initialize the Computer Vision use case.

        Args:
            pose_feature_repository: Repository for pose feature extraction
            form_analysis_repository: Repository for form analysis
            frame_repository: Repository for frame processing
            feature_repository: Repository for feature extraction
            data_handler: Handler for data operations
        """
        self.pose_feature_repo = pose_feature_repository
        self.form_analysis_repo = form_analysis_repository
        self.frame_repo = frame_repository
        self.feature_repo = feature_repository
        self.data_handler = data_handler
        self.previous_keypoints = None

    def process_keypoints(
        self,
        keypoints_dict: Dict[str, Tuple[float, float, float]],
        objects: Dict[str, Any] = None,
    ) -> PoseFeatures:
        """
        Process keypoints to extract pose features.

        Args:
            keypoints_dict: Dictionary of keypoints {name: (x, y, confidence)}
            objects: Optional dictionary of detected objects

        Returns:
            PoseFeatures object containing extracted features
        """
        # Convert dictionary to KeypointCollection
        keypoints = self._convert_to_keypoint_collection(keypoints_dict)

        # Process features
        features: PoseFeatures = self.pose_feature_repo.process_features(
            keypoints, self.previous_keypoints, objects
        )

        # Update previous keypoints for next call
        self.previous_keypoints = keypoints

        return features

    def _convert_to_keypoint_collection(
        self, keypoints_dict: Dict[str, Tuple[float, float, float]]
    ) -> KeypointCollection:
        """
        Convert dictionary of keypoints to KeypointCollection.

        Args:
            keypoints_dict: Dictionary of keypoints {name: (x, y, confidence)}

        Returns:
            KeypointCollection object
        """
        # This method should be part of the interface or a utility function
        # For now, keeping it as a private method

        keypoints_data = {}
        for name, (x, y, confidence) in keypoints_dict.items():
            keypoints_data[name] = Keypoint(
                x=float(x), y=float(y), confidence=float(confidence)
            )

        return KeypointCollection(keypoints=keypoints_data)

    def analyze_exercise_form(
        self, features: PoseFeatures, exercise_name: str
    ) -> FormAnalysis:
        """
        Analyze exercise form based on pose features.

        Args:
            features: PoseFeatures object containing extracted features
            exercise_name: Name of the exercise being performed

        Returns:
            FormAnalysis object with accuracy and suggestions
        """
        # First, detect form issues based on the exercise type
        features.form_issues = self.pose_feature_repo.detect_form_issues(
            features, exercise_name
        )

        # Then analyze the form
        return self.form_analysis_repo.analyze_form(features, exercise_name)

    def visualize_keypoints(
        self,
        frame: np.ndarray,
        keypoints_dict: Dict[str, Tuple[float, float, float]],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Visualize keypoints on a frame.

        Args:
            frame: Input frame
            keypoints_dict: Dictionary of keypoints {name: (x, y, confidence)}
            threshold: Confidence threshold for visualization

        Returns:
            Frame with keypoints visualized
        """
        keypoints = self._convert_to_keypoint_collection(keypoints_dict)
        return self.pose_feature_repo.visualize_keypoints(frame, keypoints, threshold)

    def visualize_form_analysis(
        self, frame: np.ndarray, features: PoseFeatures, analysis: FormAnalysis
    ) -> np.ndarray:
        """
        Visualize form analysis results on a frame.

        Args:
            frame: Input frame
            features: PoseFeatures object containing extracted features
            analysis: FormAnalysis object with accuracy and suggestions

        Returns:
            Frame with form analysis visualized
        """
        return self.form_analysis_repo.visualize_form_analysis(
            frame, features, analysis
        )

    def img_to_base64(self, image: np.ndarray) -> str:
        """
        Convert an image to base64 format.

        Args:
            image: Input image

        Returns:
            Base64 encoded string of the image
        """
        return self.form_analysis_repo.img_to_base64(image)

    def process_frame(self, frame_data: bytes | np.ndarray) -> Tuple[Any, Any]:
        """
        Process frame data to a processed frame.

        Args:
            frame_data: Byte data to convert or numpy array frame

        Returns:
            Tuple of (processed_frame, buffer)
        """
        # If frame_data is already a numpy array, use it directly
        if isinstance(frame_data, np.ndarray):
            frame = frame_data
        else:
            # If frame_data is bytes, convert to numpy array first
            frame = self.frame_repo.convert_byte_to_numpy(frame_data)

        return self.frame_repo.process_frame(frame)

    def create_frame_from_planes(
        self,
        primary_plane: bytes,
        secondary_plane_1: bytes,
        secondary_plane_2: bytes,
        width: int,
        height: int,
    ) -> Any:
        """
        Create a frame from separate color planes.

        Args:
            primary_plane: Primary color plane data
            secondary_plane_1: First secondary color plane data
            secondary_plane_2: Second secondary color plane data
            width: Frame width
            height: Frame height

        Returns:
            Processed frame data
        """
        return self.frame_repo.create_frame_from_planes(
            primary_plane, secondary_plane_1, secondary_plane_2, width, height
        )

    async def parse_frame(self, byte_data: bytes) -> Optional[Any]:
        """
        Parse a frame from byte data.

        Args:
            byte_data: Byte data to parse

        Returns:
            Processed frame data
        """
        return await self.frame_repo.parse_frame(byte_data)

    def perform_frame_analysis(
        self, frames_buffer: List[Any], request: Request
    ) -> Tuple[Dict, Dict, List, str]:
        """
        POSform analysis on a sequence of frames to extract pose and object data.

        Args:
            frames_buffer: List of frame data to analyze
            request: FastAPI request object containing inference services

        Returns:
            Tuple of (current_pose, previous_pose, detected_object, class_name)
        """
        return self.feature_repo.perform_frame_analysis(frames_buffer, request)

    def load_to_object_model(self, object_inference: List[Dict]) -> Object:
        """
        Convert raw object inference data to an Object model.

        Args:
            object_inference: Raw object inference data

        Returns:
            Object model representation
        """
        return self.feature_repo.load_to_object_model(object_inference)

    def load_to_features_model(
        self,
        previous_pose: Dict,
        current_pose: Dict,
        object_inference: Object,
        class_name: str,
    ) -> PoseFeatures:
        """
        Convert raw pose and object data to a PoseFeatures model.

        Args:
            previous_pose: Previous pose data
            current_pose: Current pose data
            object_inference: Object model data
            class_name: Exercise class name

        Returns:
            PoseFeatures model
        """
        return self.feature_repo.load_to_features_model(
            previous_pose, current_pose, object_inference, class_name
        )

    def get_suggestions(
        self, features: PoseFeatures, class_name: str
    ) -> Tuple[float, str]:
        """
        Generate suggestions and accuracy score based on pose features.

        Args:
            features: PoseFeatures model
            class_name: Exercise class name

        Returns:
            Tuple of (accuracy_score, suggestions_text)
        """
        return self.feature_repo.get_suggestions(features, class_name)

    def save_exercise_data(self, *args, **kwargs) -> Any:
        """
        Save exercise data to the database.

        Returns:
            Result of the save operation
        """
        return self.data_handler.save_data(*args, **kwargs)

    def load_exercise_data(self, *args, **kwargs) -> Any:
        """
        Load exercise data from the database.

        Returns:
            Loaded exercise data
        """
        return self.data_handler.load_to_data_model(*args, **kwargs)

    def process_frame_with_keypoints(
        self,
        frame: np.ndarray,
        keypoints_dict: Dict[str, Tuple[float, float, float]],
        exercise_name: str,
        objects: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Process a single frame with keypoints to extract features and analyze form.

        Args:
            frame: Input frame
            keypoints_dict: Dictionary of keypoints {name: (x, y, confidence)}
            exercise_name: Name of the exercise being performed
            objects: Optional dictionary of detected objects

        Returns:
            Dictionary containing processed results
        """
        # Extract features
        features: PoseFeatures = self.process_keypoints(keypoints_dict, objects)

        # Analyze form
        analysis: FormAnalysis = self.analyze_exercise_form(features, exercise_name)

        # Visualize results
        visualized_frame: np.ndarray = self.visualize_form_analysis(
            self.visualize_keypoints(frame, keypoints_dict), features, analysis
        )

        # Convert to base64
        b64_image: str = self.img_to_base64(visualized_frame)

        # Return results
        return {
            "features": features.model_dump(),
            "form_analysis": analysis.model_dump(),
            "visualized_frame": b64_image,
        }

    async def process_frame_async(
        self, frame: Any, thread_pool: ThreadPoolExecutor
    ) -> Any:
        """
        Process a frame asynchronously.

        Args:
            frame: Input frame to process
            thread_pool: ThreadPoolExecutor for concurrent processing

        Returns:
            Processed frame data
        """
        return await self.frame_repo.process_frame_async(frame, thread_pool)

    def format_date(self, date_string: str) -> str:
        """
        Format a date string to a more readable format.

        Args:
            date_string: Date string to format

        Returns:
            Formatted date string
        """
        return self.data_handler.format_date(date_string)


class FeatureMetricUseCase:
    """
    Use case for feature metric functionality.
    Handles the extraction of feature metrics.
    """

    def __init__(
        self,
        feature_metric_repository: FeatureMetricRepositoryInterface,
        data_handler: DataHandlerInterface,
    ):
        self.feature_metric_repo = feature_metric_repository
        self.data_handler = data_handler

    def compute_body_alignment(
        self, body_alignment: BodyAlignment, max_allowed_deviation: int = 10
    ) -> float:
        """
        Compute body alignment.
        """
        return self.feature_metric_repo.compute_ba_score(
            body_alignment, max_allowed_deviation
        )

    def compute_joint_consistency(
        self, joint_angles: Dict[str, float], max_allowed_variance: int = 15
    ) -> float:
        """
        Compute joint consistency.
        """
        return self.feature_metric_repo.compute_jc_score(
            joint_angles, max_allowed_variance
        )

    def compute_load_control(
        self, objects: Dict[str, Any], max_allowed_variance: int = 15
    ) -> float:
        """
        Compute load control.
        """
        return self.feature_metric_repo.compute_lc_score(objects, max_allowed_variance)

    def compute_speed_control(
        self, speeds: Dict[str, float], max_jerk: float = 5.0
    ) -> float:
        """
        Compute speed control.
        """
        return self.feature_metric_repo.compute_sc_score(speeds, max_jerk)

    def compute_overall_stability(
        self, stability_raw: float, max_displacement: float = 20.0
    ) -> float:
        """
        Compute overall stability.
        """
        return self.feature_metric_repo.compute_os_score(
            stability_raw, max_displacement
        )

    def compute_feature_metrics(
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
    ) -> Any:
        """
        Compute feature metrics.
        """
        metrics = self.data_handler.load_to_data_model(
            body_alignment=self.feature_metric_repo.compute_ba_score(
                body_alignment, max_allowed_deviation
            ),
            joint_consistency=self.feature_metric_repo.compute_jc_score(
                joint_angles, max_allowed_variance
            ),
            load_control=self.feature_metric_repo.compute_lc_score(
                objects, max_allowed_variance
            ),
            speed_control=self.feature_metric_repo.compute_sc_score(speeds, max_jerk),
            overall_stability=self.feature_metric_repo.compute_os_score(
                stability_raw, max_displacement
            ),
        )

        return metrics

    async def save_feature_metrics(
        self,
        username: str,
        exercise_name: str,
        feature_metrics: FeatureMetrics | Dict[str, Any],
    ) -> None:
        """
        Save feature metrics to the database using FeatureMetricsDataRepository.

        Args:
            username: User's username
            exercise_name: Exercise name
            feature_metrics: Feature metrics to save
            db: Database interface
        """
        if isinstance(feature_metrics, FeatureMetrics):
            return await self.data_handler.save_data(
                username, exercise_name, feature_metrics.model_dump()
            )
        else:
            return await self.data_handler.save_data(
                username, exercise_name, feature_metrics
            )
