import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from typing import Dict, Tuple, Annotated, Optional, Any

from core.interfaces import Inference
from infrastructure.vision.movenet.constants import KEYPOINT_DICT
from infrastructure.vision.movenet.utils import resize_to_192x192
from interfaces.api.schemas.vision_schema import Keypoint
from utilities.monitoring.factory import MonitoringFactory
from utilities.config import get_vision_settings

logger = MonitoringFactory.get_logger("movenet-pose-estimator")

class MoveNetInference(Inference):
    def __init__(self, config: Annotated[dict, get_vision_settings]):
        self.__model = hub.load(config.get('MODEL_URL'))
        self.__movenet = self.__model.signatures["serving_default"]
        logger.info("Movenet model loaded successfully")
        
    def _preprocess_frame(self, frame: np.ndarray) -> tf.Tensor:
        """Preprocess a frame for inference.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Preprocessed frame as TensorFlow tensor
        """
        # Prepare the image
        resized_frame = resize_to_192x192(frame)
        img = tf.image.resize_with_pad(tf.expand_dims(resized_frame, axis=0), 192, 192)
        return tf.cast(img, dtype=tf.int32)
        
    def _process_keypoints(self, keypoints: np.ndarray, image_shape: Tuple) -> Dict[str, Tuple[int, int, float]]:
        """Process raw keypoints into a structured format.
        
        Args:
            keypoints: Raw keypoints from model
            image_shape: Original image shape for coordinate scaling
            
        Returns:
            Dictionary mapping keypoint names to (x, y, confidence) tuples
        """
        y, x, _ = image_shape
        shaped_keypoints = {}

        for name, index in KEYPOINT_DICT.items():
            ky, kx, kp_conf = keypoints[index]   
            cx, cy = int(kx * x), int(ky * y)
            shaped_keypoints[name] = (cx, cy, float(kp_conf))

        return shaped_keypoints
    
    def get_inference(self, frame: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """Get pose estimation inference from a frame.
        
        Args:
            frame: Input frame to process. If None, returns None.
            
        Returns:
            Dictionary mapping keypoint names to (x, y, confidence) tuples,
            or None if inference fails
        """
        if frame is None:
            return None
            
        try:
            # Preprocess frame
            input_img = self._preprocess_frame(frame)
            
            # Run inference
            results = self.__movenet(input_img)
            keypoints = results["output_0"].numpy()[0, 0, :, :3]
            
            # Process keypoints
            return self._process_keypoints(keypoints, frame.shape)
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return None
    