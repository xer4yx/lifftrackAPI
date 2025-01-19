from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class PoseEstimator(ABC):
    @abstractmethod
    async def estimate_pose(self, frame: np.ndarray) -> Dict[str, Tuple[int, int, float]]:
        """Estimate pose keypoints from a frame"""
        pass

class ObjectDetector(ABC):
    @abstractmethod
    async def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in a frame"""
        pass

class ExerciseClassifier(ABC):
    @abstractmethod
    async def classify_exercise(self, frames: np.ndarray, features: np.ndarray) -> Dict[str, float]:
        """Classify exercise type from a sequence of frames"""
        pass

class FeatureExtractor(ABC):
    @abstractmethod
    async def extract_features(
        self, 
        keypoints: Dict[str, Tuple[int, int, float]], 
        previous_keypoints: Optional[Dict[str, Tuple[int, int, float]]] = None
    ) -> Dict[str, Any]:
        """Extract features from keypoints"""
        pass

class FrameProcessor(ABC):
    @abstractmethod
    async def process_frame(
        self, 
        frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a frame and return annotated frame with metadata"""
        pass

class ExerciseAnalyzer(ABC):
    @abstractmethod
    async def analyze_frame(self, frame: Any) -> Dict[str, Any]:
        """Analyze a single frame of exercise video"""
        pass

    @abstractmethod
    async def process_exercise(self, exercise_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process complete exercise data"""
        pass