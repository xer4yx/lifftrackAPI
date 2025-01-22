from inference_sdk import InferenceHTTPClient
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from utilities.monitoring.factory import MonitoringFactory
from interfaces.api.schemas.vision_schema import DetectedObject
from utilities.config import get_vision_settings

logger = MonitoringFactory.get_logger("roboflow-object-detector")

class RoboflowObjectDetector:
    def __init__(self):
        config = get_vision_settings()
        self.client = InferenceHTTPClient(
            api_url=config.get('ROBOFLOW_API_URL'),
            api_key=config.get('ROBOFLOW_API_KEY')
        )
        self.project_id = config.get('ROBOFLOW_PROJECT_ID')
        self.model_version = config.get('ROBOFLOW_MODEL_VERSION')
        self.confidence_threshold = config.get('CONFIDENCE_THRESHOLD')
        self.current_frame = None

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame before inference"""
        try:
            # TODO: Add any necessary preprocessing steps here
            # For example: resizing, normalization, etc.
            return frame
        except Exception as e:
            logger.error(f"Frame preprocessing error: {e}")
            return frame

    def set_frame(self, frame: np.ndarray) -> None:
        """Set the current frame for processing"""
        self.current_frame = frame

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in the frame using Roboflow"""
        try:
            results = self.client.infer(
                frame, 
                model_id=f"{self.project_id}/{self.model_version}"
            )
            
            if not results:
                return []
                
            return [DetectedObject(**pred).model_dump() for pred in results]
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []

    def get_inference(self) -> Optional[Dict[str, Any]]:
        """Get inference from model following the Inference protocol"""
        try:
            if self.current_frame is None:
                logger.error("No frame set for inference")
                return None

            # Preprocess the frame
            processed_frame = self.preprocess_frame(self.current_frame)
            
            # Get predictions
            predictions = self.detect_objects(processed_frame)
            
            return {"predictions": predictions}
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def draw_predictions(self, frame: np.ndarray, predictions: List[Dict[str, Any]]) -> np.ndarray:
        """Draw predictions on the frame"""
        try:
            for pred in predictions:
                if pred['confidence'] < self.confidence_threshold:
                    continue
                    
                x1 = int(pred['x'])
                y1 = int(pred['y'])
                x2 = int(pred['x'] + pred['width'])
                y2 = int(pred['y'] + pred['height'])
                
                # Draw bounding box
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence
                label = f"{pred['class']} {pred['confidence']:.2f}"
                frame = cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing predictions: {e}")
            return frame