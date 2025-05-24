from typing import Dict, Any, List, Tuple

from lifttrack.utils.logging_config import setup_logger
from core.interface.form_analysis_interface import FormAnalysisInterface
from core.entities.pose_entity import PoseFeatures, FormAnalysis

# Setup logging
logger = setup_logger("form-analysis-service", "comvis.log")


class FormAnalysisService(FormAnalysisInterface):
    """
    Service for analyzing exercise form using pose features.
    Responsible for determining form accuracy and providing suggestions for improvement.
    """
    
    def analyze_bench_press_form(self, features: PoseFeatures, exercise_name: str) -> FormAnalysis:
        """Check bench press form using enhanced feature detection."""
        accuracy = 1.0
        suggestions = []
        
        # Get form issues from features
        form_issues = features.form_issues
        
        # Get required data for analysis
        angles = features.joint_angles
        stability = features.stability
        body_alignment = features.body_alignment
        
        # Check equipment type
        objects = features.objects
        equipment_type = objects.get('type', '').lower() if objects else ''
        
        # Five-point contact check using improved stability calculation
        if stability > 50:
            accuracy -= 0.1
            suggestions.append("Keep five-point contact with the bench.")
        
        # Check wrist alignment using form_issues
        if form_issues.get('wrist_alignment'):
            accuracy -= 0.1
            suggestions.append("Wrists stay aligned with elbows throughout.")
        
        # Check back arching using enhanced body alignment
        vertical_alignment = body_alignment.vertical_alignment if body_alignment else 0
        if vertical_alignment > 20:  # More than 20 degrees from vertical
            accuracy -= 0.15
            suggestions.append("Don't over-arch; lower back stays on the bench.")
        
        # Equipment-specific checks with improved feature detection
        if "barbell" in equipment_type:
            if form_issues.get('elbow_position'):
                accuracy -= 0.15
                suggestions.append("Maintain proper elbow angle.")
            
            # Bar path check using enhanced movement pattern detection
            lateral_alignment = body_alignment.lateral_alignment if body_alignment else 0
            if lateral_alignment > 15:  # More than 15 degrees of lateral movement
                accuracy -= 0.15
                suggestions.append("Bar path: down and forward, then up and back.")
                
        elif "dumbbell" in equipment_type or len(equipment_type) == 0:
            # Enhanced dumbbell-specific checks
            if "incline" in exercise_name:
                # Use improved angle calculation for incline position
                shoulder_angle = angles.get('left_shoulder_left_elbow_left_wrist', 180)
                if abs(shoulder_angle - 45) > 10:  # More than 10 degrees off from 45
                    accuracy -= 0.15
                    suggestions.append("Arms at 45 degrees during incline press.")
        
        accuracy = max(0.0, min(1.0, accuracy))
        if not suggestions:
            suggestions.append("Form looks good! Keep it up!")
        
        logger.info(f"Bench press form accuracy: {accuracy}, Suggestions: {suggestions}")
        return FormAnalysis(accuracy=accuracy, suggestions=suggestions)
    
    def analyze_deadlift_form(self, features: PoseFeatures, exercise_name: str) -> FormAnalysis:
        """Check deadlift form using enhanced feature detection."""
        accuracy = 1.0
        suggestions = []
        
        # Get form issues from features
        form_issues = features.form_issues
        
        # Enhanced feature checks
        if form_issues.get('back_angle'):
            accuracy -= 0.2
            suggestions.append("Back flat, shoulder blades tight.")
        
        if form_issues.get('head_position'):
            accuracy -= 0.1
            suggestions.append("Head neutral, eyes forward.")
        
        # Use improved stability calculation
        stability = features.stability
        if stability > 60:
            accuracy -= 0.15
            suggestions.append("Lift with knees, hips, and shoulders together.")
        
        # Use enhanced body alignment check
        body_alignment = features.body_alignment
        if body_alignment and body_alignment.vertical_alignment > 40:
            accuracy -= 0.1
            suggestions.append("Bar stays in contact with legs.")
        
        accuracy = max(0.0, min(1.0, accuracy))
        if not suggestions:
            suggestions.append("Form looks good! Keep it up!")
        
        logger.info(f"Deadlift form accuracy: {accuracy}, Suggestions: {suggestions}")
        return FormAnalysis(accuracy=accuracy, suggestions=suggestions)
    
    def analyze_rdl_form(self, features: PoseFeatures, exercise_name: str) -> FormAnalysis:
        """Check Romanian deadlift form using enhanced feature detection."""
        accuracy = 1.0
        suggestions = []
        
        # Get form issues from features
        form_issues = features.form_issues
        
        if form_issues.get('hip_hinge'):
            accuracy -= 0.15
            suggestions.append("Push hips back, keep back flat.")
        
        # Use improved stability calculation
        stability = features.stability
        if stability > 100:
            accuracy -= 0.15
            suggestions.append("Engage core, keep back straight.")
        
        # Enhanced body alignment checks
        body_alignment = features.body_alignment
        if body_alignment:
            vertical_alignment = body_alignment.vertical_alignment
            lateral_alignment = body_alignment.lateral_alignment
            
            if vertical_alignment > 50 or lateral_alignment > 20:
                accuracy -= 0.1
                suggestions.append("Bar stays in contact with legs.")
        
        accuracy = max(0.0, min(1.0, accuracy))
        if not suggestions:
            suggestions.append("Form looks good! Keep it up!")
        
        logger.info(f"Romanian deadlift form accuracy: {accuracy}, Suggestions: {suggestions}")
        return FormAnalysis(accuracy=accuracy, suggestions=suggestions)
    
    def analyze_shoulder_press_form(self, features: PoseFeatures, exercise_name: str) -> FormAnalysis:
        """Check shoulder press form using enhanced feature detection."""
        accuracy = 1.0
        suggestions = []
        
        # Use enhanced stability calculation with core joints
        stability = features.stability
        if stability > 20:
            accuracy -= 0.1
            suggestions.append("Minimize body sway for stability.")
        
        # Use improved body alignment calculation
        body_alignment = features.body_alignment
        if body_alignment and body_alignment.vertical_alignment > 65:
            accuracy -= 0.1
            suggestions.append("Stay vertically aligned, head to hips.")
        
        # Enhanced movement pattern detection
        angles = features.joint_angles
        left_elbow = angles.get('left_shoulder_left_elbow_left_wrist', 180)
        right_elbow = angles.get('right_shoulder_right_elbow_right_wrist', 180)
        
        if (left_elbow < 30 or left_elbow > 100) or (right_elbow < 30 or right_elbow > 100):
            accuracy -= 0.15
            suggestions.append("Keep elbows in proper position.")
        
        accuracy = max(0.0, min(1.0, accuracy))
        if not suggestions:
            suggestions.append("Form looks good! Keep it up!")
        
        logger.info(f"Shoulder press form accuracy: {accuracy}, Suggestions: {suggestions}")
        return FormAnalysis(accuracy=accuracy, suggestions=suggestions)
    
    def analyze_form(self, features: PoseFeatures, exercise_name: str) -> FormAnalysis:
        """Main function to calculate form accuracy using enhanced feature detection."""
        try:
            # Check if user is resting - pass keypoints and get resting state result
            if isinstance(features.keypoints, dict):
                # Should not happen with proper typing, but handle for safety
                keypoints = features.keypoints
            else:
                keypoints = features.keypoints.as_dict()
                
            # First, detect if the user is in a resting state
            resting_state = features.objects.get("resting_state", {})
            if resting_state.get('is_resting', False):
                logger.info(f"User is resting - Position: {resting_state.get('position', 'unknown')}")
                return FormAnalysis(accuracy=0.0, suggestions=["Idling"])
                
            # Normalize exercise name to handle different formats
            exercise_name_normalized = exercise_name.lower().replace(" ", "_")
            
            # Call appropriate exercise-specific function
            if exercise_name_normalized in ["benchpress", "bench_press"]:
                return self.analyze_bench_press_form(features, exercise_name_normalized)
            elif exercise_name_normalized == "deadlift":
                return self.analyze_deadlift_form(features, exercise_name_normalized)
            elif exercise_name_normalized in ["romanian_deadlift", "rdl"]:
                return self.analyze_rdl_form(features, exercise_name_normalized)
            elif exercise_name_normalized in ["shoulder_press", "overhead_press"]:
                return self.analyze_shoulder_press_form(features, exercise_name_normalized)
            else:
                logger.warning(f"Unknown exercise type: {exercise_name_normalized}")
                return FormAnalysis(
                    accuracy=1.0, 
                    suggestions=["Exercise type not recognized for form analysis."]
                )
                
        except Exception as e:
            logger.error(f"Error in analyze_form: {str(e)}")
            raise 