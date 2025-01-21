from typing import Optional, Dict, Any, Protocol, runtime_checkable

@runtime_checkable
class Inference(Protocol):
    def get_inference(self) -> Optional[Dict[str, Any]]:
        """Get inference from model
        
        Args:
            input_data: Input data to run inference on. Could be:
                - Single frame (np.ndarray) for pose estimation
                - Single frame (np.ndarray) for object detection  
                - Multiple frames (np.ndarray) with features for exercise classification
            
        Returns:
            Dictionary containing inference results based on model type:
                - For 'pose': {keypoint_name: (x, y, confidence)}
                - For 'object': List of detected objects with properties
                - For 'exercise': {exercise_type: confidence_score}
        """
        pass