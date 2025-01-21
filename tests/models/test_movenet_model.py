import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch

from infrastructure.vision.movenet.model import MoveNetInference
from infrastructure.vision.movenet.constants import KEYPOINT_DICT

@pytest.fixture
def mock_config():
    return {
        'MODEL_URL': 'dummy_url'
    }

@pytest.fixture
def mock_model():
    with patch('tensorflow_hub.load') as mock_hub_load:
        # Create a mock model with a serving_default signature
        mock_model = Mock()
        mock_model.signatures = {
            "serving_default": Mock()
        }
        mock_hub_load.return_value = mock_model
        yield mock_model

@pytest.fixture
def movenet_inference(mock_config, mock_model):
    return MoveNetInference(mock_config)

def test_preprocess_frame(movenet_inference):
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Process the frame
    result = movenet_inference._preprocess_frame(frame)
    
    # Check output properties
    assert isinstance(result, tf.Tensor)
    assert result.shape == (1, 192, 192, 3)
    assert result.dtype == tf.int32

def test_process_keypoints(movenet_inference):
    # Create dummy keypoints (17 keypoints with y, x, confidence values)
    keypoints = np.random.rand(17, 3)
    image_shape = (480, 640, 3)
    
    # Process keypoints
    result = movenet_inference._process_keypoints(keypoints, image_shape)
    
    # Verify results
    assert isinstance(result, dict)
    assert len(result) == len(KEYPOINT_DICT)
    
    # Check structure of each keypoint
    for name, (x, y, conf) in result.items():
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(conf, float)
        assert 0 <= conf <= 1.0

def test_get_inference_none_frame(movenet_inference):
    result = movenet_inference.get_inference(None)
    assert result is None

def test_get_inference_valid_frame(movenet_inference):
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Mock the model inference result
    dummy_keypoints = np.random.rand(1, 1, 17, 3)  # Shape matches model output
    mock_result = {"output_0": tf.constant(dummy_keypoints)}
    movenet_inference._MoveNetInference__movenet.return_value = mock_result
    
    # Get inference
    result = movenet_inference.get_inference(frame)
    
    # Verify results
    assert isinstance(result, dict)
    assert len(result) == len(KEYPOINT_DICT)

def test_get_inference_handles_exception(movenet_inference):
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Make the model raise an exception
    movenet_inference._MoveNetInference__movenet.side_effect = Exception("Test error")
    
    # Get inference
    result = movenet_inference.get_inference(frame)
    
    # Verify it handles the exception gracefully
    assert result is None