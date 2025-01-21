from typing import Dict, Tuple, Any, Optional, List
import numpy as np

from core.entities.vision import Frame
from core.interfaces import Inference, FrameRepository

class VisionService:
    def __init__(
        self,
        model_inferences: List[Inference],
        frame_repository: Optional[FrameRepository] = None
    ):
        self.model_inferences = model_inferences
        self.frame_repository = frame_repository
        self.previous_keypoints = None
    
    def analyze_frame(self) -> Dict[str, Any]:
        """Analyze frame and return annotated frame and metadata"""
        inference_results = {}
        for model_inference in self.model_inferences:
            inference = model_inference.get_inference()
            if inference:
                inference_results[str(model_inference.__class__.__name__)] = inference
        return inference_results
