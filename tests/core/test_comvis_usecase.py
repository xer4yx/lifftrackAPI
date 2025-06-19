import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from typing import Dict, Tuple, List, Any

from core.usecase.comvis_usecase import ComVisUseCase
from core.entities.pose_entity import (
    KeypointCollection,
    PoseFeatures,
    FormAnalysis,
    Object,
)


class TestComVisUseCase(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock objects for all dependencies
        self.mock_pose_feature_repo = Mock()
        self.mock_form_analysis_repo = Mock()
        self.mock_frame_repo = Mock()
        self.mock_feature_repo = Mock()
        self.mock_data_handler = Mock()

        # Create the use case with mock dependencies
        self.use_case = ComVisUseCase(
            self.mock_pose_feature_repo,
            self.mock_form_analysis_repo,
            self.mock_frame_repo,
            self.mock_feature_repo,
            self.mock_data_handler,
        )

        # Set up common test data
        self.test_keypoints_dict = {
            "nose": (100.0, 100.0, 0.9),
            "left_shoulder": (80.0, 130.0, 0.85),
            "right_shoulder": (120.0, 130.0, 0.87),
        }

        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_exercise_name = "squat"

        # Mock the KeypointCollection that would be created
        self.expected_keypoints = KeypointCollection(
            keypoints={
                "nose": {"x": 100.0, "y": 100.0, "confidence": 0.9},
                "left_shoulder": {"x": 80.0, "y": 130.0, "confidence": 0.85},
                "right_shoulder": {"x": 120.0, "y": 130.0, "confidence": 0.87},
            }
        )

    def test_process_keypoints(self):
        """Test the process_keypoints method."""
        # Set up mock return values
        mock_features = MagicMock(spec=PoseFeatures)
        self.mock_pose_feature_repo.process_features.return_value = mock_features

        # Call the method
        result = self.use_case.process_keypoints(self.test_keypoints_dict)

        # Assertions
        self.mock_pose_feature_repo.process_features.assert_called_once()
        self.assertEqual(result, mock_features)

        # Check if previous_keypoints is updated
        self.assertIsNotNone(self.use_case.previous_keypoints)

    def test_analyze_exercise_form(self):
        """Test the analyze_exercise_form method."""
        # Set up mock objects and return values
        mock_features = MagicMock(spec=PoseFeatures)
        mock_analysis = MagicMock(spec=FormAnalysis)

        self.mock_pose_feature_repo.detect_form_issues.return_value = [
            "Issue 1",
            "Issue 2",
        ]
        self.mock_form_analysis_repo.analyze_form.return_value = mock_analysis

        # Call the method
        result = self.use_case.analyze_exercise_form(
            mock_features, self.test_exercise_name
        )

        # Assertions
        self.mock_pose_feature_repo.detect_form_issues.assert_called_once_with(
            mock_features, self.test_exercise_name
        )
        self.mock_form_analysis_repo.analyze_form.assert_called_once_with(
            mock_features, self.test_exercise_name
        )
        self.assertEqual(result, mock_analysis)

    def test_visualize_keypoints(self):
        """Test the visualize_keypoints method."""
        # Setup mock return values
        visualized_frame = np.ones((480, 640, 3), dtype=np.uint8)
        self.mock_pose_feature_repo.visualize_keypoints.return_value = visualized_frame

        # Call the method
        result = self.use_case.visualize_keypoints(
            self.test_frame, self.test_keypoints_dict, threshold=0.6
        )

        # Assertions
        self.mock_pose_feature_repo.visualize_keypoints.assert_called_once()
        self.assertEqual(result.shape, visualized_frame.shape)
        np.testing.assert_array_equal(result, visualized_frame)

    def test_visualize_form_analysis(self):
        """Test the visualize_form_analysis method."""
        # Setup mock objects and return values
        mock_features = MagicMock(spec=PoseFeatures)
        mock_analysis = MagicMock(spec=FormAnalysis)
        visualized_frame = np.ones((480, 640, 3), dtype=np.uint8)

        self.mock_form_analysis_repo.visualize_form_analysis.return_value = (
            visualized_frame
        )

        # Call the method
        result = self.use_case.visualize_form_analysis(
            self.test_frame, mock_features, mock_analysis
        )

        # Assertions
        self.mock_form_analysis_repo.visualize_form_analysis.assert_called_once_with(
            self.test_frame, mock_features, mock_analysis
        )
        np.testing.assert_array_equal(result, visualized_frame)

    def test_img_to_base64(self):
        """Test the img_to_base64 method."""
        # Setup mock return values
        b64_string = "base64encodedstring"
        self.mock_form_analysis_repo.img_to_base64.return_value = b64_string

        # Call the method
        result = self.use_case.img_to_base64(self.test_frame)

        # Assertions
        self.mock_form_analysis_repo.img_to_base64.assert_called_once_with(
            self.test_frame
        )
        self.assertEqual(result, b64_string)

    def test_process_frame(self):
        """Test the process_frame method."""
        # Setup mock return values
        byte_data = b"test_byte_data"
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_buffer = np.zeros((480, 640, 3), dtype=np.uint8)

        self.mock_frame_repo.convert_byte_to_numpy.return_value = test_frame
        self.mock_frame_repo.process_frame.return_value = (test_frame, test_buffer)

        # Call the method
        result_frame, result_buffer = self.use_case.process_frame(byte_data)

        # Assertions
        self.mock_frame_repo.convert_byte_to_numpy.assert_called_once_with(byte_data)
        self.mock_frame_repo.process_frame.assert_called_once_with(test_frame)
        self.assertTrue(np.array_equal(result_frame, test_frame))
        self.assertTrue(np.array_equal(result_buffer, test_buffer))

    def test_create_frame_from_planes(self):
        """Test the create_frame_from_planes method."""
        # Setup test data
        primary_plane = b"primary_plane_data"
        secondary_plane_1 = b"secondary_plane_1_data"
        secondary_plane_2 = b"secondary_plane_2_data"
        width = 640
        height = 480
        test_frame = np.zeros((height, width, 3), dtype=np.uint8)

        self.mock_frame_repo.create_frame_from_planes.return_value = test_frame

        # Call the method
        result = self.use_case.create_frame_from_planes(
            primary_plane, secondary_plane_1, secondary_plane_2, width, height
        )

        # Assertions
        self.mock_frame_repo.create_frame_from_planes.assert_called_once_with(
            primary_plane, secondary_plane_1, secondary_plane_2, width, height
        )
        self.assertTrue(np.array_equal(result, test_frame))

    def test_perform_frame_analysis(self):
        """Test the perform_frame_analysis method."""
        # Setup test data
        frames_buffer = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        current_pose = {"nose": (100.0, 100.0, 0.9)}
        previous_pose = {"nose": (98.0, 99.0, 0.88)}
        detected_object = [{"class": "barbell", "box": [10, 20, 30, 40]}]
        class_name = "squat"
        mock_request = MagicMock()

        self.mock_feature_repo.perform_frame_analysis.return_value = (
            current_pose,
            previous_pose,
            detected_object,
            class_name,
        )

        # Call the method
        result = self.use_case.perform_frame_analysis(frames_buffer, mock_request)

        # Assertions
        self.mock_feature_repo.perform_frame_analysis.assert_called_once_with(
            frames_buffer, mock_request
        )
        self.assertEqual(
            result, (current_pose, previous_pose, detected_object, class_name)
        )

    def test_load_to_object_model(self):
        """Test the load_to_object_model method."""
        # Setup test data
        object_inference = [{"class": "barbell", "box": [10, 20, 30, 40]}]
        mock_object = MagicMock(spec=Object)

        self.mock_feature_repo.load_to_object_model.return_value = mock_object

        # Call the method
        result = self.use_case.load_to_object_model(object_inference)

        # Assertions
        self.mock_feature_repo.load_to_object_model.assert_called_once_with(
            object_inference
        )
        self.assertEqual(result, mock_object)

    def test_load_to_features_model(self):
        """Test the load_to_features_model method."""
        # Setup test data
        previous_pose = {"nose": (98.0, 99.0, 0.88)}
        current_pose = {"nose": (100.0, 100.0, 0.9)}
        mock_object = MagicMock(spec=Object)
        class_name = "squat"
        mock_features = MagicMock(spec=PoseFeatures)

        self.mock_feature_repo.load_to_features_model.return_value = mock_features

        # Call the method
        result = self.use_case.load_to_features_model(
            previous_pose, current_pose, mock_object, class_name
        )

        # Assertions
        self.mock_feature_repo.load_to_features_model.assert_called_once_with(
            previous_pose, current_pose, mock_object, class_name
        )
        self.assertEqual(result, mock_features)

    def test_get_suggestions(self):
        """Test the get_suggestions method."""
        # Setup test data
        mock_features = MagicMock(spec=PoseFeatures)
        class_name = "squat"
        accuracy = 0.85
        suggestions = "Keep your back straight"

        self.mock_feature_repo.get_suggestions.return_value = (accuracy, suggestions)

        # Call the method
        result_accuracy, result_suggestions = self.use_case.get_suggestions(
            mock_features, class_name
        )

        # Assertions
        self.mock_feature_repo.get_suggestions.assert_called_once_with(
            mock_features, class_name
        )
        self.assertEqual(result_accuracy, accuracy)
        self.assertEqual(result_suggestions, suggestions)

    def test_save_exercise_data(self):
        """Test the save_exercise_data method."""
        # Setup test data
        test_data = {"exercise": "squat", "reps": 10}
        expected_result = {"id": 1, "exercise": "squat", "reps": 10}

        self.mock_data_handler.save_data.return_value = expected_result

        # Call the method
        result = self.use_case.save_exercise_data(**test_data)

        # Assertions
        self.mock_data_handler.save_data.assert_called_once_with(**test_data)
        self.assertEqual(result, expected_result)

    def test_load_exercise_data(self):
        """Test the load_exercise_data method."""
        # Setup test data
        test_query = {"user_id": 123}
        expected_result = [{"id": 1, "exercise": "squat", "reps": 10}]

        self.mock_data_handler.load_to_data_model.return_value = expected_result

        # Call the method
        result = self.use_case.load_exercise_data(**test_query)

        # Assertions
        self.mock_data_handler.load_to_data_model.assert_called_once_with(**test_query)
        self.assertEqual(result, expected_result)

    def test_process_frame_with_keypoints(self):
        """Test the process_frame_with_keypoints method."""
        # Mock objects setup
        mock_features = MagicMock(spec=PoseFeatures)
        mock_features.model_dump.return_value = {"features": "data"}

        mock_analysis = MagicMock(spec=FormAnalysis)
        mock_analysis.model_dump.return_value = {"analysis": "data"}

        visualized_keypoints_frame = np.ones((480, 640, 3), dtype=np.uint8)
        visualized_analysis_frame = np.ones((480, 640, 3), dtype=np.uint8) * 2
        b64_image = "base64encodedstring"

        # Setup mock method returns
        self.use_case.process_keypoints = MagicMock(return_value=mock_features)
        self.use_case.analyze_exercise_form = MagicMock(return_value=mock_analysis)
        self.use_case.visualize_keypoints = MagicMock(
            return_value=visualized_keypoints_frame
        )
        self.use_case.visualize_form_analysis = MagicMock(
            return_value=visualized_analysis_frame
        )
        self.use_case.img_to_base64 = MagicMock(return_value=b64_image)

        # Call the method
        result = self.use_case.process_frame_with_keypoints(
            self.test_frame, self.test_keypoints_dict, self.test_exercise_name
        )

        # Assertions
        self.use_case.process_keypoints.assert_called_once_with(
            self.test_keypoints_dict, None
        )
        self.use_case.analyze_exercise_form.assert_called_once_with(
            mock_features, self.test_exercise_name
        )
        self.use_case.visualize_keypoints.assert_called_once_with(
            self.test_frame, self.test_keypoints_dict
        )
        self.use_case.visualize_form_analysis.assert_called_once_with(
            visualized_keypoints_frame, mock_features, mock_analysis
        )
        self.use_case.img_to_base64.assert_called_once_with(visualized_analysis_frame)

        # Check the result structure
        self.assertEqual(result["features"], {"features": "data"})
        self.assertEqual(result["form_analysis"], {"analysis": "data"})
        self.assertEqual(result["visualized_frame"], b64_image)


if __name__ == "__main__":
    unittest.main()
