from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.entities.pose_entity import KeypointCollection, PoseFeatures


class PoseFeatureInterface(ABC):
    @abstractmethod
    def extract_joint_angles(self, keypoints: KeypointCollection, 
                          confidence_threshold: float = 0.1) -> Dict[str, float]:
        """
        Extracts joint angles from keypoints.
        
        Args:
            keypoints: Collection of keypoints with positions
            confidence_threshold: Minimum confidence value to consider a keypoint valid
            
        Returns:
            Dictionary containing joint angles (in degrees)
        """
        pass
    
    @abstractmethod
    def extract_movement_patterns(self, keypoints: KeypointCollection, 
                               previous_keypoints: KeypointCollection) -> Dict[str, float]:
        """
        Tracks the movement pattern of joints by calculating displacement from previous keypoints.
        
        Args:
            keypoints: Current keypoints
            previous_keypoints: Previous keypoints to compare against
            
        Returns:
            Dictionary of displacements (distances moved) for each joint
        """
        pass
    
    @abstractmethod
    def calculate_speed(self, displacement: Dict[str, float], 
                     time_delta: float = 1.0) -> Dict[str, float]:
        """
        Computes the speed of movement based on displacement and time delta.
        
        Args:
            displacement: Dictionary of joint displacements
            time_delta: Time interval between frames (in seconds)
            
        Returns:
            Dictionary of speeds for each joint
        """
        pass
    
    @abstractmethod
    def extract_body_alignment(self, keypoints: KeypointCollection) -> Tuple[float, float]:
        """
        Calculate body alignment by measuring the angle between shoulders and hips.
        
        Args:
            keypoints: Collection of keypoints with positions
            
        Returns:
            Tuple of (vertical_alignment, lateral_alignment)
        """
        pass
    
    @abstractmethod
    def calculate_stability(self, keypoints: KeypointCollection, 
                         previous_keypoints: KeypointCollection,
                         window_size: int = 5) -> float:
        """
        Calculate stability as the total displacement of core keypoints over time.
        
        Args:
            keypoints: Current keypoints
            previous_keypoints: Previous keypoints to compare against
            window_size: Number of frames to consider for stability calculation
            
        Returns:
            A measure of stability (lower is more stable)
        """
        pass
    
    @abstractmethod
    def detect_form_issues(self, features: PoseFeatures, 
                        exercise_type: str) -> Dict[str, bool]:
        """
        Detect common form issues for specific exercises.
        
        Args:
            features: Dictionary of extracted features
            exercise_type: Type of exercise being performed
            
        Returns:
            Dictionary of detected form issues
        """
        pass
    
    @abstractmethod
    def detect_resting_state(self, keypoints: KeypointCollection, 
                          features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if the user is in a resting state (standing or sitting).
        
        Args:
            keypoints: Collection of keypoints with positions
            features: Dictionary containing detected features
            
        Returns:
            Dictionary containing resting state information
        """
        pass
    
    @abstractmethod
    def process_features(self, keypoints: KeypointCollection, 
                      previous_keypoints: Optional[KeypointCollection] = None,
                      objects: Dict[str, Any] = None) -> PoseFeatures:
        """
        Process keypoints to extract all features.
        
        Args:
            keypoints: Current keypoints
            previous_keypoints: Previous keypoints for motion analysis
            objects: Detected objects in the frame
            
        Returns:
            PoseFeatures object with all extracted features
        """
        pass 