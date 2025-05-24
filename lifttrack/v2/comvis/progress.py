import cv2  
import base64
import numpy as np
from lifttrack.utils.logging_config import setup_logger
from .features import (calculate_angle, extract_joint_angles, extract_movement_patterns,
                      calculate_speed, extract_body_alignment, calculate_stability,
                      detect_form_issues, detect_resting_state)

logger = setup_logger("progress-v2", "comvis.log")

# Function to display the keypoints on the frame
def display_keypoints_on_frame(frame, keypoints, threshold=0.5):
    for joint, (x, y, score) in keypoints.items():
        if score > threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(frame, joint, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

def check_bench_press_form(features, predicted_class_name):
    """Check bench press form using enhanced feature detection."""
    accuracy = 1.0
    
    # Get form issues using the new detect_form_issues function
    form_issues = detect_form_issues(features, 'bench_press')
    suggestions = []
    
    # Common checks using enhanced feature calculations
    angles = features.get('joint_angles', {})
    stability = features.get('stability', 0)
    body_alignment = features.get('body_alignment', {})
    
    # Check equipment type
    objects = features.get('objects', {})
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
    vertical_alignment = float(body_alignment.get('vertical_alignment', body_alignment.get('0', 0)))
    if vertical_alignment > 20:  # More than 20 degrees from vertical
        accuracy -= 0.15
        suggestions.append("Don't over-arch; lower back stays on the bench.")
    
    # Equipment-specific checks with improved feature detection
    if "barbell" in equipment_type:
        if form_issues.get('elbow_position'):
            accuracy -= 0.15
            suggestions.append("Maintain proper elbow angle.")
        
        # Bar path check using enhanced movement pattern detection
        lateral_alignment = float(body_alignment.get('lateral_alignment', body_alignment.get('1', 0)))
        if lateral_alignment > 15:  # More than 15 degrees of lateral movement
            accuracy -= 0.15
            suggestions.append("Bar path: down and forward, then up and back.")
            
    elif "dumbbell" in equipment_type or len(equipment_type) == 0:
        # Enhanced dumbbell-specific checks
        if "incline" in predicted_class_name:
            # Use improved angle calculation for incline position
            shoulder_angle = angles.get('left_shoulder_left_elbow_left_wrist', 180)
            if abs(shoulder_angle - 45) > 10:  # More than 10 degrees off from 45
                accuracy -= 0.15
                suggestions.append("Arms at 45 degrees during incline press.")
    
    accuracy = max(0.0, min(1.0, accuracy))
    if not suggestions:
        suggestions.append("Form looks good! Keep it up!")
    
    logger.info(f"Bench press form accuracy: {accuracy}, Suggestions: {suggestions}")
    return accuracy, suggestions

def check_deadlift_form(features, predicted_class_name):
    """Check deadlift form using enhanced feature detection."""
    accuracy = 1.0
    
    # Get form issues using the new detect_form_issues function
    form_issues = detect_form_issues(features, 'deadlift')
    suggestions = []
    
    # Enhanced feature checks
    if form_issues.get('back_angle'):
        accuracy -= 0.2
        suggestions.append("Back flat, shoulder blades tight.")
    
    if form_issues.get('head_position'):
        accuracy -= 0.1
        suggestions.append("Head neutral, eyes forward.")
    
    # Use improved stability calculation
    stability = features.get('stability', 0)
    if stability > 60:
        accuracy -= 0.15
        suggestions.append("Lift with knees, hips, and shoulders together.")
    
    # Use enhanced body alignment check
    body_alignment = features.get('body_alignment', {})
    if float(body_alignment.get('vertical_alignment', body_alignment.get('0', 0))) > 40:
        accuracy -= 0.1
        suggestions.append("Bar stays in contact with legs.")
    
    accuracy = max(0.0, min(1.0, accuracy))
    if not suggestions:
        suggestions.append("Form looks good! Keep it up!")
    
    logger.info(f"Deadlift form accuracy: {accuracy}, Suggestions: {suggestions}")
    return accuracy, suggestions

def check_rdl_form(features, predicted_class_name):
    """Check Romanian deadlift form using enhanced feature detection."""
    accuracy = 1.0
    
    # Get form issues using the new detect_form_issues function
    form_issues = detect_form_issues(features, 'romanian_deadlift')
    suggestions = []
    
    if form_issues.get('hip_hinge'):
        accuracy -= 0.15
        suggestions.append("Push hips back, keep back flat.")
    
    # Use improved stability calculation
    stability = features.get('stability', 0)
    if stability > 100:
        accuracy -= 0.15
        suggestions.append("Engage core, keep back straight.")
    
    # Enhanced body alignment checks
    body_alignment = features.get('body_alignment', {})
    vertical_alignment = float(body_alignment.get('vertical_alignment', body_alignment.get('0', 0)))
    lateral_alignment = float(body_alignment.get('lateral_alignment', body_alignment.get('1', 0)))
    
    if vertical_alignment > 50 or lateral_alignment > 20:
        accuracy -= 0.1
        suggestions.append("Bar stays in contact with legs.")
    
    accuracy = max(0.0, min(1.0, accuracy))
    if not suggestions:
        suggestions.append("Form looks good! Keep it up!")
    
    logger.info(f"Romanian deadlift form accuracy: {accuracy}, Suggestions: {suggestions}")
    return accuracy, suggestions

def check_shoulder_press_form(features, predicted_class_name):
    """Check shoulder press form using enhanced feature detection."""
    accuracy = 1.0
    suggestions = []
    
    # Use enhanced stability calculation with core joints
    stability = features.get('stability', 0)
    if stability > 20:
        accuracy -= 0.1
        suggestions.append("Minimize body sway for stability.")
    
    # Use improved body alignment calculation
    body_alignment = features.get('body_alignment', {})
    vertical_alignment = float(body_alignment.get('vertical_alignment', body_alignment.get('0', 0)))
    if vertical_alignment > 65:
        accuracy -= 0.1
        suggestions.append("Stay vertically aligned, head to hips.")
    
    # Enhanced movement pattern detection
    angles = features.get('joint_angles', {})
    left_elbow = angles.get('left_shoulder_left_elbow_left_wrist', 180)
    right_elbow = angles.get('right_shoulder_right_elbow_right_wrist', 180)
    
    if (left_elbow < 30 or left_elbow > 100) or (right_elbow < 30 or right_elbow > 100):
        accuracy -= 0.15
        suggestions.append("Keep elbows in proper position.")
    
    accuracy = max(0.0, min(1.0, accuracy))
    if not suggestions:
        suggestions.append("Form looks good! Keep it up!")
    
    logger.info(f"Shoulder press form accuracy: {accuracy}, Suggestions: {suggestions}")
    return accuracy, suggestions

def calculate_form_accuracy(features, predicted_class_name):
    """Main function to calculate form accuracy using enhanced feature detection."""
    print(f"Features: {features}")
    try:
        # First check if user is resting
        resting_state = detect_resting_state(features.get('keypoints', {}), features)
        if resting_state['is_resting']:
            logger.info(f"User is resting - Position: {resting_state['position']}")
            return 0.0, ["Idling"]
            
        # Normalize exercise name to handle different formats
        predicted_class_name = predicted_class_name.lower().replace(" ", "_")
        
        # Call appropriate exercise-specific function
        if predicted_class_name in ["benchpress", "bench_press"]:
            return check_bench_press_form(features, predicted_class_name)
        elif predicted_class_name == "deadlift":
            return check_deadlift_form(features, predicted_class_name)
        elif predicted_class_name in ["romanian_deadlift", "rdl"]:
            return check_rdl_form(features, predicted_class_name)
        elif predicted_class_name in ["shoulder_press", "overhead_press"]:
            return check_shoulder_press_form(features, predicted_class_name)
        else:
            logger.warning(f"Unknown exercise type: {predicted_class_name}")
            return 1.0, ["Exercise type not recognized for form analysis."]
            
    except Exception as e:
        logger.error(f"Error in calculate_form_accuracy: {str(e)}")
        raise

# Function to convert an image to base64 format
def img_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64
