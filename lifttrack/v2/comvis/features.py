import numpy as np
import math
import cv2


# Function to compute the angle between three points
def calculate_angle(a, b, c):
    """Calculates the angle formed by three points a, b, c."""
    # Convert points to numpy arrays for easier calculation
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vectors BA and BC
    ba = a - b
    bc = c - b
    
    # Compute the cosine of the angle using dot product
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Avoid domain errors due to floating point precision
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Return the angle in degrees
    angle = np.arccos(cos_angle) * (180.0 / np.pi)
    return angle


# Function to calculate joint angles for all pairs of keypoints
def extract_joint_angles(keypoints, confidence_threshold=0.1):
    """
    Extracts joint angles for all 17 keypoints, filtered by confidence.
    
    Args:
    - keypoints: A dictionary of keypoints with positions {key: (x, y, score)}
    - confidence_threshold: Minimum confidence value to consider a keypoint valid
    
    Returns:
    - angles: A dictionary containing joint angles (in degrees)
    """
    # First filter keypoints by confidence
    filtered_keypoints = {k: v for k, v in keypoints.items() if v[2] >= confidence_threshold}
    
    # Add more exercise-specific joint pairs for better form analysis
    joint_pairs = [
        ('left_shoulder', 'left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee', 'left_ankle'),
        ('right_hip', 'right_knee', 'right_ankle'),
        ('left_shoulder', 'left_hip', 'left_knee'),
        ('right_shoulder', 'right_hip', 'right_knee'),
        ('left_shoulder', 'right_shoulder', 'neck'),
        ('left_hip', 'right_hip', 'waist'),
        # Add these new pairs for better exercise form analysis
        ('left_ear', 'left_shoulder', 'left_hip'),  # Head alignment for deadlift
        ('right_ear', 'right_shoulder', 'right_hip'),  # Head alignment for deadlift
        ('left_shoulder', 'left_hip', 'left_ankle'),  # Back-to-ankle alignment for deadlift
        ('right_shoulder', 'right_hip', 'right_ankle'),  # Back-to-ankle alignment for deadlift
        ('left_wrist', 'left_shoulder', 'left_hip'),  # Bar path for bench press
        ('right_wrist', 'right_shoulder', 'right_hip')  # Bar path for bench press
    ]
    
    angles = {}
    
    for pair in joint_pairs:
        joint1, joint2, joint3 = pair
        if joint1 in filtered_keypoints and joint2 in filtered_keypoints and joint3 in filtered_keypoints:
            angle = calculate_angle(filtered_keypoints[joint1], filtered_keypoints[joint2], filtered_keypoints[joint3])
            angles[f'{joint1}_{joint2}_{joint3}'] = angle
    return angles


# Function to compute movement patterns (basic example: displacement)
def extract_movement_patterns(keypoints, previous_keypoints):
    """
    Tracks the movement pattern of joints by calculating displacement from previous keypoints.
    
    Args:
    - keypoints: Current keypoints
    - previous_keypoints: Previous keypoints to compare against
    
    Returns:
    - displacement: List of displacements (distances moved) for each joint
    """
    displacement = {}
    
    for joint in keypoints:
        if joint in previous_keypoints:
            curr_pos = np.array(keypoints[joint][:2])  # Use x, y coordinates
            prev_pos = np.array(previous_keypoints[joint][:2])
            dist = np.linalg.norm(curr_pos - prev_pos)
            displacement[joint] = dist
    
    return displacement


# Function to compute speed of joints or weights
def calculate_speed(displacement, time_delta=1.0):
    """
    Computes the speed of movement based on displacement and time delta.
    
    Args:
    - displacement: Dictionary of joint displacements (in pixels or meters)
    - time_delta: Time interval between frames (in seconds)
    
    Returns:
    - speed: Dictionary of speeds for each joint (displacement / time)
    """
    speed = {}
    for joint in displacement:
        speed[joint] = displacement[joint] / time_delta
    return speed


# Function to calculate body alignment (e.g., horizontal angle between shoulders and hips)
def extract_body_alignment(keypoints):
    """
    Calculate body alignment by measuring the angle between shoulders and hips.
    
    Args:
    - keypoints: A dictionary of keypoints with positions {key: (x, y, score)}
    
    Returns:
    - alignment: Dictionary with shoulder and hip angles
    """
    alignment = [0, 0]  # Default values
    
    if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints and 'left_hip' in keypoints and 'right_hip' in keypoints:
        # Calculate horizontal alignment (deviation from vertical)
        left_shoulder = np.array(keypoints['left_shoulder'][:2])
        right_shoulder = np.array(keypoints['right_shoulder'][:2])
        left_hip = np.array(keypoints['left_hip'][:2])
        right_hip = np.array(keypoints['right_hip'][:2])
        
        # Calculate midpoints
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        hip_midpoint = (left_hip + right_hip) / 2
        
        # Calculate vertical alignment (angle with vertical axis)
        vertical_vector = np.array([0, 1])  # Vertical reference vector
        body_vector = shoulder_midpoint - hip_midpoint
        
        # Normalize vectors
        vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
        if np.linalg.norm(body_vector) > 0:
            body_vector = body_vector / np.linalg.norm(body_vector)
            
            # Calculate angle between body vector and vertical
            cos_angle = np.dot(body_vector, vertical_vector)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            vertical_alignment = np.arccos(cos_angle) * (180.0 / np.pi)
            
            # Calculate lateral tilt (left-right balance)
            shoulder_vector = right_shoulder - left_shoulder
            hip_vector = right_hip - left_hip
            
            if np.linalg.norm(shoulder_vector) > 0 and np.linalg.norm(hip_vector) > 0:
                shoulder_vector = shoulder_vector / np.linalg.norm(shoulder_vector)
                hip_vector = hip_vector / np.linalg.norm(hip_vector)
                
                cos_lateral = np.dot(shoulder_vector, hip_vector)
                cos_lateral = np.clip(cos_lateral, -1.0, 1.0)
                lateral_alignment = np.arccos(cos_lateral) * (180.0 / np.pi)
                
                alignment = [vertical_alignment, lateral_alignment]
    
    return {"0": alignment[0], "1": alignment[1]}


# Function to calculate stability (e.g., variance in body position or keypoint consistency)
def calculate_stability(keypoints, previous_keypoints, window_size=5):
    """
    Calculate stability as the total displacement of core keypoints over time.
    Lower values indicate better stability.
    
    Args:
    - keypoints: Current keypoints
    - previous_keypoints: List of previous keypoint sets (up to window_size)
    - window_size: Number of frames to consider for stability calculation
    
    Returns:
    - stability: A measure of stability (lower is more stable)
    """
    # Core keypoints that are most relevant for stability
    core_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee']
    
    # If we don't have enough previous keypoints, just calculate based on what we have
    if not previous_keypoints:
        return 0.0
    
    total_displacement = 0.0
    count = 0
    
    # Focus on core joints for stability calculation
    for joint in core_joints:
        if joint in keypoints:
            curr_pos = np.array(keypoints[joint][:2])
            
            # Calculate displacement from previous position
            if joint in previous_keypoints:
                prev_pos = np.array(previous_keypoints[joint][:2])
                displacement = np.linalg.norm(curr_pos - prev_pos)
                
                # Weight displacement by joint importance
                if 'shoulder' in joint or 'hip' in joint:
                    # Core joints have higher weight
                    total_displacement += displacement * 1.5
                else:
                    total_displacement += displacement
                
                count += 1
    
    # Normalize by number of joints to get average displacement
    if count > 0:
        return total_displacement / count
    return 0.0


# New function to detect exercise-specific form issues
def detect_form_issues(features, exercise_type):
    """
    Detect common form issues for specific exercises.
    
    Args:
    - features: Dictionary of extracted features
    - exercise_type: Type of exercise being performed
    
    Returns:
    - issues: Dictionary of detected form issues
    """
    issues = {}
    angles = features.get('joint_angles', {})
    
    if exercise_type == 'bench_press':
        # Check for wrist alignment
        left_wrist_angle = abs(90 - angles.get('left_shoulder_left_elbow_left_wrist', 90))
        right_wrist_angle = abs(90 - angles.get('right_shoulder_right_elbow_right_wrist', 90))
        
        if left_wrist_angle > 20 or right_wrist_angle > 20:
            issues['wrist_alignment'] = True
        
        # Check for elbow position
        left_elbow = angles.get('left_shoulder_left_elbow_left_wrist', 180)
        right_elbow = angles.get('right_shoulder_right_elbow_right_wrist', 180)
        
        if left_elbow > 110 or right_elbow > 110:
            issues['elbow_position'] = True
            
    elif exercise_type == 'deadlift' or exercise_type == 'romanian_deadlift':
        # Check for back angle
        back_angle = angles.get('left_shoulder_left_hip_left_knee', 180)
        
        if exercise_type == 'deadlift' and back_angle < 150:
            issues['back_angle'] = True
        elif exercise_type == 'romanian_deadlift' and (back_angle > 140 or back_angle < 60):
            issues['hip_hinge'] = True
            
        # Check for head position
        head_angle = angles.get('left_ear_left_shoulder_left_hip', 180)
        if abs(head_angle - 180) > 20:
            issues['head_position'] = True
    
    return issues


def normalize_keypoints(keypoints, image_dimensions):
    """
    Normalize keypoints to be in range [0,1] and adjust coordinate system.
    
    Args:
    - keypoints: Dictionary of keypoints {name: (x, y, confidence)}
    - image_dimensions: Tuple of (height, width) of the image
    
    Returns:
    - normalized_keypoints: Dictionary with normalized coordinates
    """
    height, width = image_dimensions
    normalized = {}
    
    for name, (x, y, conf) in keypoints.items():
        # Normalize to [0,1] range
        norm_x = x / width
        norm_y = y / height
        normalized[name] = (norm_x, norm_y, conf)
        
    return normalized


def filter_low_confidence_keypoints(keypoints, threshold=0.3):
    """
    Filter out keypoints with confidence below threshold.
    
    Args:
    - keypoints: Dictionary of keypoints
    - threshold: Minimum confidence value (default 0.3)
    
    Returns:
    - filtered_keypoints: Dictionary with only high-confidence keypoints
    """
    return {k: v for k, v in keypoints.items() if v[2] >= threshold}


def calculate_relative_distances(keypoints):
    """
    Calculate distances relative to a body-scale unit (like shoulder width)
    to make measurements scale-invariant.
    
    Args:
    - keypoints: Dictionary of keypoints
    
    Returns:
    - Dictionary of relative distances
    """
    distances = {}
    
    # Use shoulder width as the reference unit if available
    if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
        lsh = np.array(keypoints['left_shoulder'][:2])
        rsh = np.array(keypoints['right_shoulder'][:2])
        reference_distance = np.linalg.norm(lsh - rsh)
        
        if reference_distance > 0:
            # Calculate distances between meaningful pairs
            pairs = [
                ('left_elbow', 'left_wrist'),
                ('right_elbow', 'right_wrist'),
                ('left_hip', 'left_knee'),
                ('right_hip', 'right_knee'),
                # Add more meaningful pairs
            ]
            
            for p1, p2 in pairs:
                if p1 in keypoints and p2 in keypoints:
                    pt1 = np.array(keypoints[p1][:2])
                    pt2 = np.array(keypoints[p2][:2])
                    abs_distance = np.linalg.norm(pt1 - pt2)
                    # Normalize by shoulder width
                    distances[f"{p1}_{p2}"] = abs_distance / reference_distance
    
    return distances


def visualize_angles(frame, keypoints, angles, confidence_threshold=0.3):
    """
    Visualize calculated angles on the frame for validation purposes.
    
    Args:
    - frame: The original image/video frame
    - keypoints: Dictionary of keypoints
    - angles: Dictionary of calculated angles
    - confidence_threshold: Minimum confidence to display
    
    Returns:
    - frame: The annotated frame
    """
    # Create a copy of the frame
    annotated_frame = frame.copy()
    
    # Draw keypoints first (similar to the MovenetInference class)
    for name, (x, y, confidence) in keypoints.items():
        if confidence > confidence_threshold:
            cv2.circle(annotated_frame, (x, y), 4, (0, 255, 0), -1)
            
    # Draw angles
    for angle_name, angle_value in angles.items():
        # Parse the joint names from the angle name
        joint_names = angle_name.split('_')
        if len(joint_names) >= 3:
            middle_joint = joint_names[1]  # The middle joint is where the angle is measured
            
            if middle_joint in keypoints and keypoints[middle_joint][2] > confidence_threshold:
                x, y = keypoints[middle_joint][:2]
                # Display the angle value near the middle joint
                cv2.putText(annotated_frame, f"{angle_value:.1f}°", 
                           (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 0), 1)
                
    return annotated_frame
