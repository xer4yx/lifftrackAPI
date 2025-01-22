import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from infrastructure.vision.roboflow.model import RoboflowObjectDetector

@pytest.fixture
def mock_config():
    return {
        'ROBOFLOW_API_URL': 'https://test.roboflow.com',
        'ROBOFLOW_API_KEY': 'test_key',
        'ROBOFLOW_PROJECT_ID': 'test_project',
        'ROBOFLOW_MODEL_VERSION': '1',
        'CONFIDENCE_THRESHOLD': 0.5
    }

@pytest.fixture
def mock_frame():
    # Create a simple test frame (100x100 black image)
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def mock_predictions():
    return [
        {
            'x': 10,
            'y': 20,
            'width': 30,
            'height': 40,
            'class_name': 'person',
            'confidence': 0.85
        }
    ]

@patch('infrastructure.vision.roboflow.model.get_vision_settings')
@patch('infrastructure.vision.roboflow.model.InferenceHTTPClient')
def test_initialization(mock_client, mock_settings, mock_config):
    mock_settings.return_value = mock_config
    detector = RoboflowObjectDetector()
    
    assert detector.project_id == mock_config['ROBOFLOW_PROJECT_ID']
    assert detector.model_version == mock_config['ROBOFLOW_MODEL_VERSION']
    assert detector.confidence_threshold == mock_config['CONFIDENCE_THRESHOLD']

@patch('infrastructure.vision.roboflow.model.get_vision_settings')
@patch('infrastructure.vision.roboflow.model.InferenceHTTPClient')
def test_set_frame(mock_client, mock_settings, mock_config, mock_frame):
    mock_settings.return_value = mock_config
    detector = RoboflowObjectDetector()
    
    detector.set_frame(mock_frame)
    assert detector.current_frame is not None
    assert (detector.current_frame == mock_frame).all()

@patch('infrastructure.vision.roboflow.model.get_vision_settings')
@patch('infrastructure.vision.roboflow.model.InferenceHTTPClient')
def test_detect_objects_success(mock_client, mock_settings, mock_config, mock_frame, mock_predictions):
    mock_settings.return_value = mock_config
    mock_client_instance = Mock()
    mock_client_instance.infer.return_value = mock_predictions
    mock_client.return_value = mock_client_instance
    
    detector = RoboflowObjectDetector()
    results = detector.detect_objects(mock_frame)
    
    assert len(results) == 1
    assert results[0]['class_name'] == 'person'
    assert results[0]['confidence'] == 0.85

@patch('infrastructure.vision.roboflow.model.get_vision_settings')
@patch('infrastructure.vision.roboflow.model.InferenceHTTPClient')
def test_detect_objects_failure(mock_client, mock_settings, mock_config, mock_frame):
    mock_settings.return_value = mock_config
    mock_client_instance = Mock()
    mock_client_instance.infer.side_effect = Exception("API Error")
    mock_client.return_value = mock_client_instance
    
    detector = RoboflowObjectDetector()
    results = detector.detect_objects(mock_frame)
    
    assert results == []

@patch('infrastructure.vision.roboflow.model.get_vision_settings')
@patch('infrastructure.vision.roboflow.model.InferenceHTTPClient')
def test_get_inference_success(mock_client, mock_settings, mock_config, mock_frame, mock_predictions):
    mock_settings.return_value = mock_config
    mock_client_instance = Mock()
    mock_client_instance.infer.return_value = mock_predictions
    mock_client.return_value = mock_client_instance
    
    detector = RoboflowObjectDetector()
    detector.set_frame(mock_frame)
    
    result = detector.get_inference()
    assert result is not None
    assert 'predictions' in result
    assert len(result['predictions']) == 1

@patch('infrastructure.vision.roboflow.model.get_vision_settings')
@patch('infrastructure.vision.roboflow.model.InferenceHTTPClient')
def test_draw_predictions(mock_client, mock_settings, mock_config, mock_frame, mock_predictions):
    mock_settings.return_value = mock_config
    detector = RoboflowObjectDetector()
    
    result_frame = detector.draw_predictions(mock_frame, mock_predictions)
    assert result_frame is not None
    assert result_frame.shape == mock_frame.shape 