from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional

from core.entities import Object, PoseFeatures


class FeatureRepositoryInterface(ABC):
    """
    Abstract interface for feature repositories.
    Defines the contract that all feature repository implementations must fulfill.
    """
    
    @abstractmethod
    def perform_frame_analysis(self, *args, **kwargs) -> Tuple[Dict, Dict, List, str]:
        """
        Perform analysis on a sequence of frames to extract pose and object data.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            Tuple of (current_pose, previous_pose, detected_object, class_name)
        """
        pass
        
    @abstractmethod
    def load_to_object_model(self, object_inference: List[Dict]) -> Object:
        """
        Convert raw object inference data to an Object model.
        
        Args:
            object_inference: Raw object inference data
            
        Returns:
            Object model representation
        """
        pass
        
    @abstractmethod
    def load_to_features_model(self, previous_pose: Dict, current_pose: Dict, 
                            object_inference: Object, class_name: str) -> PoseFeatures:
        """
        Convert raw pose and object data to a PoseFeatures model.
        
        Args:
            previous_pose: Previous pose data
            current_pose: Current pose data
            object_inference: Object model data
            class_name: Exercise class name
            
        Returns:
            PoseFeatures model
        """
        pass
        
    @abstractmethod
    def get_suggestions(self, features: PoseFeatures, class_name: str) -> Tuple[float, str]:
        """
        Generate suggestions and accuracy score based on pose features.
        
        Args:
            features: PoseFeatures model
            class_name: Exercise class name
            
        Returns:
            Tuple of (accuracy_score, suggestions_text)
        """
        pass 