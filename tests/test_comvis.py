import unittest
import numpy as np
import tensorflow as tf
import base64
import cv2
from unittest.mock import Mock, patch, MagicMock
from lifttrack.comvis import (
    websocket_process_frames,
    MoveNetInference,
    RoboflowInference,
    ExerciseFormAnalyzer
)
import logging
import os
from PIL import Image
import io
from lifttrack.utils.logging_config import setup_logger

# Configure logging for test_comvis
logger = setup_logger("test_comvis", "test_modules.log")

class TestMoveNetInference(unittest.TestCase):
    @patch('lifttrack.comvis.tensor.hub.load')
    def setUp(self, mock_hub_load):
        logger.info(f"Setting up {self.__class__.__name__}")
        
        # Mock the TensorFlow Hub model
        mock_model = Mock()
        mock_model.signatures = {'serving_default': Mock()}
        mock_hub_load.return_value = mock_model
        logger.debug("TensorFlow Hub model mocked successfully")
        
        self.movenet = MoveNetInference()
        logger.debug("MoveNet instance created")
        
        # Create a sample test image
        self.test_image = np.zeros((192, 192, 3), dtype=np.uint8)
        self.test_image = tf.convert_to_tensor(self.test_image)
        logger.debug("Test image created and converted to tensor")

    def test_run_keypoint_inference(self):
        logger.info(f"Started testing {self.test_run_keypoint_inference.__name__}")
        try:
            # Mock the inference output
            mock_keypoints = tf.zeros((1, 17, 3))
            self.movenet.movenet.return_value = {'output_0': mock_keypoints}
            logger.debug("Mock keypoints created")
            
            result = self.movenet.run_keypoint_inference(self.test_image)
            logger.debug("Keypoint inference completed")
            logger.debug(f"Result shape: {result.shape}")
            
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, (1, 17, 3))
            logger.debug("Assertions passed successfully")
            
        except Exception as e:
            logger.error(f"Error in keypoint inference test: {str(e)}")
            raise
            
        logger.info(f"{self.test_run_keypoint_inference.__name__} testing completed")

class TestRoboflowInference(unittest.TestCase):
    @patch('lifttrack.comvis.tensor.check_docker_container_status')
    @patch('lifttrack.comvis.tensor.InferenceHTTPClient')
    def setUp(self, mock_client, mock_docker_check):
        logger.info(f"Setting up {self.__class__.__name__}")
        
        self.roboflow = RoboflowInference()
        self.mock_client = mock_client.return_value
        logger.debug("Roboflow instance and mock client created")
        
        self.test_image = np.zeros((192, 192, 3), dtype=np.uint8)
        logger.debug("Test image created")

    def test_run_object_inference(self):
        logger.info(f"Started testing {self.test_run_object_inference.__name__}")
        try:
            mock_response = {
                'predictions': [],
                'image': {'width': 192, 'height': 192}
            }
            self.mock_client.infer.return_value = mock_response
            logger.debug("Mock response created")
            
            result = self.roboflow.run_object_inference(self.test_image)
            logger.debug("Object inference completed")
            
            self.assertIsNotNone(result)
            self.assertIn('predictions', result)
            self.assertIn('image', result)
            logger.debug("Assertions passed successfully")
            
        except Exception as e:
            logger.error(f"Error in object inference test: {str(e)}")
            raise
            
        logger.info(f"{self.test_run_object_inference.__name__} testing completed")

class TestExerciseFormAnalyzer(unittest.TestCase):
    @patch('lifttrack.comvis.Live.tf.keras.models.load_model')
    def setUp(self, mock_load_model):
        logger.info(f"Setting up {self.__class__.__name__}")
        
        self.analyzer = ExerciseFormAnalyzer()
        logger.debug("Exercise form analyzer created")
        
        self.test_frame = np.zeros((112, 112, 3), dtype=np.float32)
        self.test_features = np.zeros(9, dtype=np.float32)
        logger.debug("Test frame and features created")

    def test_process_frame_for_cnn(self):
        logger.info(f"Started testing {self.test_process_frame_for_cnn.__name__}")
        try:
            processed = self.analyzer.process_frame_for_cnn(self.test_frame)
            logger.debug("Frame processed for CNN")
            
            self.assertEqual(processed.shape, (112, 112, 3))
            self.assertEqual(processed.dtype, np.float16)
            logger.debug("Assertions passed successfully")
            
        except Exception as e:
            logger.error(f"Error in process frame test: {str(e)}")
            raise
            
        logger.info(f"{self.test_process_frame_for_cnn.__name__} testing completed")

    def test_add_to_buffer(self):
        logger.info(f"Started testing {self.test_add_to_buffer.__name__}")
        try:
            initial_index = self.analyzer.buffer_index
            logger.debug(f"Initial buffer index: {initial_index}")
            
            self.analyzer.add_to_buffer(self.test_frame, self.test_features)
            logger.debug("Frame and features added to buffer")
            
            self.assertEqual(self.analyzer.buffer_index, (initial_index + 1) % 30)
            logger.debug("Buffer index assertion passed")
            
        except Exception as e:
            logger.error(f"Error in add to buffer test: {str(e)}")
            raise
            
        logger.info(f"{self.test_add_to_buffer.__name__} testing completed")

    def test_get_buffer_for_prediction(self):
        logger.info(f"Started testing {self.test_get_buffer_for_prediction.__name__}")
        try:
            # Fill buffer with test data
            for i in range(5):
                self.analyzer.add_to_buffer(self.test_frame, self.test_features)
                logger.debug(f"Added frame {i+1} to buffer")
            
            frames, features = self.analyzer.get_buffer_for_prediction()
            logger.debug("Retrieved frames and features from buffer")
            
            self.assertEqual(frames.shape, (1, 30, 112, 112, 3))
            self.assertEqual(features.shape, (1, 30, 9))
            logger.debug("Shape assertions passed successfully")
            
        except Exception as e:
            logger.error(f"Error in get buffer test: {str(e)}")
            raise
            
        logger.info(f"{self.test_get_buffer_for_prediction.__name__} testing completed")

class TestComVisUtils(unittest.TestCase):
    def setUp(self):
        logger.info(f"Setting up {self.__class__.__name__}")
        
        # Load the exercise GIF and extract first frame
        try:
            gif = Image.open(os.path.join(os.path.dirname(__file__), 'res', 'rdl_test.gif'))
            # Convert first frame to numpy array
            first_frame = np.array(gif.convert('RGB'))
            self.test_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
            logger.debug("Exercise GIF loaded and first frame extracted")
            
            # Create bytes for websocket testing
            _, buffer = cv2.imencode('.jpg', self.test_frame)
            self.test_frame_bytes = buffer.tobytes()
            logger.debug("Frame converted to bytes format")
            
            # Store all frames for multiple frame testing
            self.frames = []
            try:
                while True:
                    gif.seek(len(self.frames))
                    frame = np.array(gif.convert('RGB'))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    _, buffer = cv2.imencode('.jpg', frame)
                    self.frames.append(buffer.tobytes())
            except EOFError:
                pass
            logger.debug(f"Extracted {len(self.frames)} frames from GIF")
            
        except Exception as e:
            logger.error(f"Error loading exercise GIF: {str(e)}")
            raise

        # Create test annotations structure
        self.test_annotation = {
            'keypoints': {
                'left_shoulder': [0, 0],
                'left_elbow': [1, 1],
                'left_wrist': [2, 2],
                'left_hip': [0, 0],
                'left_knee': [1, 1],
                'left_ankle': [2, 2],
                'right_shoulder': [0, 1],
                'right_hip': [0, 1]
            },
            'objects': [{'class': 'dumbbell', 'x': 1, 'y': 1}],
            'image': {'width': self.test_frame.shape[1], 
                     'height': self.test_frame.shape[0]}
        }
        logger.debug("Test annotations created")

    @patch('lifttrack.comvis.movenet.run_keypoint_inference')
    @patch('lifttrack.comvis.roboflow.run_object_inference')
    def test_websocket_multiple_frames(self, mock_roboflow, mock_movenet):
        logger.info(f"Started testing {self.test_websocket_multiple_frames.__name__}")
        try:
            # Configure mocks
            mock_movenet.return_value = self.test_annotation['keypoints']
            mock_roboflow.return_value = {
                'predictions': self.test_annotation['objects'],
                'image': {'width': self.test_frame.shape[1], 
                         'height': self.test_frame.shape[0]}
            }
            logger.debug("Mock responses configured")

            # Process multiple frames from the GIF
            for i, frame_bytes in enumerate(self.frames[:3]):  # Test first 3 frames
                frame, prediction = websocket_process_frames(self.analyzer, frame_bytes)
                logger.debug(f"Frame {i+1} processed")
                
                self.assertIsNotNone(frame)
                self.assertTrue(isinstance(frame, np.ndarray))
                self.assertEqual(len(frame.shape), 3)
                self.assertIsNotNone(prediction)
                logger.debug(f"Frame {i+1} assertions passed")
                
        except Exception as e:
            logger.error(f"Error in multiple frames test: {str(e)}")
            raise
            
        logger.info(f"{self.test_websocket_multiple_frames.__name__} testing completed")

    @patch('lifttrack.comvis.movenet.run_keypoint_inference')
    @patch('lifttrack.comvis.roboflow.run_object_inference')
    def test_websocket_frame_processing(self, mock_roboflow, mock_movenet):
        logger.info(f"Started testing {self.test_websocket_frame_processing.__name__}")
        try:
            # Set up mocks
            mock_keypoints = {
                'left_shoulder': [0.5, 0.5],
                'left_elbow': [0.6, 0.6],
                'left_wrist': [0.7, 0.7],
                'left_hip': [0.4, 0.4],
                'left_knee': [0.3, 0.3],
                'left_ankle': [0.2, 0.2],
                'right_shoulder': [0.5, 0.4],
                'right_hip': [0.4, 0.3]
            }
            mock_objects = [{
                'class': 'barbell',
                'x': 0.5,
                'y': 0.5,
                'confidence': 0.95
            }]
            
            mock_movenet.return_value = mock_keypoints
            mock_roboflow.return_value = {
                'predictions': mock_objects,
                'image': {'width': 640, 'height': 480}
            }
            logger.debug("Mock responses configured")

            frame, prediction = websocket_process_frames(self.analyzer, self.test_frame_bytes)
            logger.debug("Frame processed")

            self.assertIsNotNone(frame)
            self.assertTrue(isinstance(frame, np.ndarray))
            self.assertEqual(len(frame.shape), 3)
            self.assertIsNotNone(prediction)
            self.assertIn('prediction', prediction)
            self.assertIn('predicted_class', prediction)
            self.assertIn('class_name', prediction)
            logger.debug("All assertions passed")
            
        except Exception as e:
            logger.error(f"Error in websocket frame processing test: {str(e)}")
            raise
            
        logger.info(f"{self.test_websocket_frame_processing.__name__} testing completed")

    @patch('lifttrack.comvis.movenet.run_keypoint_inference')
    @patch('lifttrack.comvis.roboflow.run_object_inference')
    def test_websocket_error_handling(self, mock_roboflow, mock_movenet):
        logger.info(f"Started testing {self.test_websocket_error_handling.__name__}")
        try:
            # Test corrupted data
            corrupted_bytes = b'corrupted_image_data'
            frame, prediction = websocket_process_frames(self.analyzer, corrupted_bytes)
            logger.debug("Tested corrupted image data")
            
            self.assertIsNone(frame)
            self.assertIsNone(prediction)
            logger.debug("Corrupted data assertions passed")

            # Test empty data
            empty_bytes = b''
            frame, prediction = websocket_process_frames(self.analyzer, empty_bytes)
            logger.debug("Tested empty data")
            
            self.assertIsNone(frame)
            self.assertIsNone(prediction)
            logger.debug("Empty data assertions passed")
            
        except Exception as e:
            logger.error(f"Error in websocket error handling test: {str(e)}")
            raise
            
        logger.info(f"{self.test_websocket_error_handling.__name__} testing completed")

    @patch('lifttrack.comvis.movenet.run_keypoint_inference')
    @patch('lifttrack.comvis.roboflow.run_object_inference')
    def test_websocket_large_image(self, mock_roboflow, mock_movenet):
        logger.info(f"Started testing {self.test_websocket_large_image.__name__}")
        try:
            large_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
            _, buffer = cv2.imencode('.jpg', large_image)
            large_image_bytes = buffer.tobytes()
            logger.debug("Large test image created")

            mock_movenet.return_value = self.test_annotation['keypoints']
            mock_roboflow.return_value = {
                'predictions': self.test_annotation['objects'],
                'image': {'width': 1920, 'height': 1080}
            }
            logger.debug("Mock responses configured")

            frame, prediction = websocket_process_frames(self.analyzer, large_image_bytes)
            logger.debug("Large frame processed")

            self.assertIsNotNone(frame)
            self.assertTrue(isinstance(frame, np.ndarray))
            self.assertEqual(len(frame.shape), 3)
            logger.debug("Assertions passed")
            
        except Exception as e:
            logger.error(f"Error in large image test: {str(e)}")
            raise
            
        logger.info(f"{self.test_websocket_large_image.__name__} testing completed")

    def test_websocket_frame_size_validation(self):
        logger.info(f"Started testing {self.test_websocket_frame_size_validation.__name__}")
        try:
            sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
            
            for width, height in sizes:
                with self.subTest(width=width, height=height):
                    logger.debug(f"Testing frame size {width}x{height}")
                    
                    test_image = np.ones((height, width, 3), dtype=np.uint8) * 255
                    _, buffer = cv2.imencode('.jpg', test_image)
                    test_bytes = buffer.tobytes()
                    logger.debug("Test image created")
                    
                    frame, _ = websocket_process_frames(self.analyzer, test_bytes)
                    logger.debug("Frame processed")
                    
                    if frame is not None:
                        actual_height, actual_width = frame.shape[:2]
                        original_aspect_ratio = width / height
                        processed_aspect_ratio = actual_width / actual_height
                        
                        self.assertAlmostEqual(
                            original_aspect_ratio,
                            processed_aspect_ratio,
                            places=2
                        )
                        logger.debug(f"Aspect ratio assertion passed for size {width}x{height}")
                    
        except Exception as e:
            logger.error(f"Error in frame size validation test: {str(e)}")
            raise
            
        logger.info(f"{self.test_websocket_frame_size_validation.__name__} testing completed")

if __name__ == '__main__':
    unittest.main() 