import numpy as np
import pytest
from unittest.mock import Mock, patch

from infrastructure.vision.cnn.model import CNNExerciseClassifier

@pytest.fixture
def mock_keras_model():
    mock_model = Mock()
    # Mock predict to return a sample prediction array for 4 classes
    mock_model.predict.return_value = np.array([[0.1, 0.7, 0.1, 0.1]])
    return mock_model

@pytest.fixture
def classifier(mock_keras_model):
    with patch('keras.models.load_model', return_value=mock_keras_model):
        config = {'MODEL_PATH': 'dummy/path/model.h5'}
        return CNNExerciseClassifier(
            input_shape=(224, 224),
            max_video_length=16,
            config=config
        )

def test_preprocess_frame(classifier):
    # Test RGB frame
    rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed = classifier._preprocess_frame(rgb_frame)
    
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float16
    assert np.max(processed) <= 1.0
    assert np.min(processed) >= 0.0

def test_analyze_frame(classifier):
    # Create a sample frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # First frame should return None (buffer not full)
    result = classifier.analyze_frame(frame)
    assert result is None
    
    # Fill the buffer
    for _ in range(15):
        result = classifier.analyze_frame(frame)
        assert result is None
    
    # Buffer should be full now, expect a prediction
    result = classifier.analyze_frame(frame)
    assert isinstance(result, dict)
    assert 'class_name' in result
    assert 'confidence' in result
    assert result['class_name'] == 'deadlift'  # Based on mock prediction
    assert result['confidence'] == 0.7

def test_get_inference(classifier):
    result = classifier.get_inference()
    
    assert isinstance(result, dict)
    assert result['class_name'] == 'deadlift'
    assert result['confidence'] == 0.7

def test_prepare_frames_for_inference(classifier):
    # Test with empty buffer
    frames = classifier._prepare_frames_for_inference()
    assert frames.shape == (16, 224, 224, 3)
    
    # Test with partially filled buffer
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    classifier.analyze_frame(frame)
    frames = classifier._prepare_frames_for_inference()
    assert frames.shape == (16, 224, 224, 3)
