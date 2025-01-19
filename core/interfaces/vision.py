from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class ModelInference(ABC):
    @abstractmethod
    def get_inference(
        self,
        input_data: Any,
        model: object
    ) -> Dict[str, Any]:
        """Get inference from specified model type
        
        Args:
            input_data: Input data to run inference on. Could be:
                - Single frame (np.ndarray) for pose estimation
                - Single frame (np.ndarray) for object detection  
                - Multiple frames (np.ndarray) with features for exercise classification
            model_type: Type of model to use ('pose', 'object', 'exercise')
            
        Returns:
            Dictionary containing inference results based on model type:
                - For 'pose': {keypoint_name: (x, y, confidence)}
                - For 'object': List of detected objects with properties
                - For 'exercise': {exercise_type: confidence_score}
        """
        pass