from typing import Dict, Any, List, Tuple
import base64
import cv2
import numpy as np

from core.interface.form_analysis_interface import FormAnalysisInterface
from core.entities import PoseFeatures, FormAnalysis
from core.service.form_analysis_service import FormAnalysisService
from lifttrack.utils.logging_config import setup_logger

# Setup logging
logger = setup_logger("form-analysis-repository", "comvis.log")


class FormAnalysisRepository(FormAnalysisInterface):
    """
    Implementation of the FormAnalysisInterface that handles exercise form analysis.
    This repository is responsible for analyzing the form of users performing exercises.
    """
    
    def __init__(self):
        # Use the service implementation to avoid duplicating code
        self._service = FormAnalysisService()
    
    def analyze_bench_press_form(self, features: PoseFeatures, 
                               exercise_name: str) -> FormAnalysis:
        """Delegate to service implementation."""
        return self._service.analyze_bench_press_form(features, exercise_name)
    
    def analyze_deadlift_form(self, features: PoseFeatures, 
                           exercise_name: str) -> FormAnalysis:
        """Delegate to service implementation."""
        return self._service.analyze_deadlift_form(features, exercise_name)
    
    def analyze_rdl_form(self, features: PoseFeatures, 
                      exercise_name: str) -> FormAnalysis:
        """Delegate to service implementation."""
        return self._service.analyze_rdl_form(features, exercise_name)
    
    def analyze_shoulder_press_form(self, features: PoseFeatures, 
                                 exercise_name: str) -> FormAnalysis:
        """Delegate to service implementation."""
        return self._service.analyze_shoulder_press_form(features, exercise_name)
    
    def analyze_form(self, features: PoseFeatures, 
                  exercise_name: str) -> FormAnalysis:
        """Delegate to service implementation."""
        return self._service.analyze_form(features, exercise_name)
    
    def img_to_base64(self, image: np.ndarray) -> str:
        """
        Convert an image to base64 format.
        
        Args:
            image: The image as a numpy array
            
        Returns:
            Base64 encoded string of the image
        """
        _, buffer = cv2.imencode('.jpg', image)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return img_b64
    
    def visualize_form_analysis(self, frame: np.ndarray, 
                             features: PoseFeatures, 
                             analysis: FormAnalysis) -> np.ndarray:
        """
        Visualize form analysis results on an image frame.
        
        Args:
            frame: The original frame to annotate
            features: The PoseFeatures used for analysis
            analysis: The FormAnalysis result
            
        Returns:
            Annotated frame with form analysis visualization
        """
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Get dimensions
        h, w = annotated_frame.shape[:2]
        
        # Add a semi-transparent overlay at the top for the form accuracy
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        
        # Add form accuracy as a colored bar
        accuracy = analysis.accuracy
        bar_color = (0, 255, 0)  # Green for good form
        
        if accuracy < 0.7:
            bar_color = (0, 0, 255)  # Red for poor form
        elif accuracy < 0.9:
            bar_color = (0, 165, 255)  # Orange for moderate form
            
        bar_width = int(w * accuracy)
        cv2.rectangle(overlay, (0, 60), (bar_width, 80), bar_color, -1)
        
        # Add text for accuracy percentage
        cv2.putText(overlay, f"Form Accuracy: {accuracy*100:.1f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add suggestions
        if analysis.suggestions:
            suggestion_text = " | ".join(analysis.suggestions)
            cv2.putText(overlay, suggestion_text[:50], 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # If suggestion is too long, continue on next line
            if len(suggestion_text) > 50:
                cv2.putText(overlay, suggestion_text[50:100], 
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Apply the overlay with transparency
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        return annotated_frame 