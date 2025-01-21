from typing import Dict, Any, Optional, Tuple, Annotated, List

import cv2
import numpy as np
import tensorflow as tf
import keras

from core.interfaces import Inference
from utilities.monitoring import MonitoringFactory
from utilities.config import get_vision_settings

logger = MonitoringFactory.get_logger("cnn-exercise-classifier")

class CNNExerciseClassifier(Inference):
    def __init__(self, input_shape: Tuple[int, int], max_video_length: int, config: Annotated[dict, get_vision_settings]):
        self.model = keras.models.load_model(config.get('MODEL_PATH'))
        self.input_shape = input_shape
        self.max_video_length = max_video_length
        self.frame_buffer = np.zeros((max_video_length, *input_shape, 3), dtype=np.float16)
        self.feature_buffer = np.zeros((max_video_length, 9), dtype=np.float16)
        self.buffer_index = 0
        self.frames_collected = 0  # Track total frames collected
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame for the model.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Preprocessed frame
        """
        # Convert grayscale to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # type: ignore
        # Convert BGR to RGB if needed
        elif frame.shape[-1] == 3 and not np.array_equal(frame[:,:,0], frame[:,:,2]):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
        # Handle RGBA
        elif frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)  # type: ignore
            
        # Resize frame
        frame_resized = cv2.resize(frame, self.input_shape)  # type: ignore
        
        # Normalize
        return (frame_resized.astype(np.float32) / 255.0).astype(np.float16)
        
    def _prepare_frames_for_inference(self) -> np.ndarray:
        """Prepare buffered frames for model inference.
        
        Returns:
            Processed frames ready for model input
        """
        # Handle temporal dimension
        if self.buffer_index < self.max_video_length:
            # Pad with last frame if buffer not full
            last_frame = self.frame_buffer[self.buffer_index - 1] if self.buffer_index > 0 else np.zeros_like(self.frame_buffer[0])
            padding = np.array([last_frame] * (self.max_video_length - self.buffer_index))
            frames = np.concatenate([self.frame_buffer[:self.buffer_index], padding])
        else:
            frames = self.frame_buffer
            
        return frames
        
    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process a new frame and run inference if buffer is ready.
        
        Args:
            frame: New frame to process
            
        Returns:
            Inference results if buffer is full (max_video_length frames), None otherwise
        """
        # Check if we should return inference before processing new frame
        if self.frames_collected == self.max_video_length and self.buffer_index == 0:
            return self.get_inference()
            
        processed_frame = self._preprocess_frame(frame)
        self.frame_buffer[self.buffer_index] = processed_frame
        
        # Update counters after storing frame
        if self.frames_collected < self.max_video_length:
            self.frames_collected += 1
        self.buffer_index = (self.buffer_index + 1) % self.max_video_length
        
        return None

    def get_inference(self) -> Optional[Dict[str, Any]]:
        """Get inference from model using buffered frames
        
        Returns:
            Dictionary containing exercise classification results:
            {
                'class_name': str,  # Name of predicted exercise
                'confidence': float  # Confidence score of prediction
            }
            or None if inference fails
        """
        try:
            frames = self._prepare_frames_for_inference()
            frames_batch = np.expand_dims(frames, axis=0)
            
            with tf.device('/CPU:0'):
                predictions = self.model.predict(frames_batch, verbose=0)
            
            class_names = {
                0: "benchpress",
                1: "deadlift",
                2: "romanian_deadlift",
                3: "shoulder_press",
            }
            
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][predicted_class_index])
            
            return {
                'class_name': class_names[predicted_class_index],
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None
