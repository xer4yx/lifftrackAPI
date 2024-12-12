import numpy as np
import math


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
def extract_joint_angles(keypoints):
    """
    Extracts joint angles for all 17 keypoints by considering each joint as an angle 
    formed by the relevant bones (keypoints).
    
    Args:
    - keypoints: A dictionary of keypoints with positions {key: (x, y, score)}
    
    Returns:
    - angles: A dictionary containing joint angles (in degrees)
    """
    joint_pairs = [
        ('left_shoulder', 'left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee', 'left_ankle'),
        ('right_hip', 'right_knee', 'right_ankle'),
        ('left_shoulder', 'left_hip', 'left_knee'),
        ('right_shoulder', 'right_hip', 'right_knee'),
        ('left_shoulder', 'right_shoulder', 'neck'),  # You may need to define 'neck' from the keypoints
        ('left_hip', 'right_hip', 'waist')  # This would be custom, depends on the body alignment you want to track
    ]
    
    angles = {}
    
    for pair in joint_pairs:
        joint1, joint2, joint3 = pair
        if joint1 in keypoints and joint2 in keypoints and joint3 in keypoints:
            angle = calculate_angle(keypoints[joint1], keypoints[joint2], keypoints[joint3])
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
    - alignment: Body alignment angle (degrees)
    """
    if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints and 'left_hip' in keypoints and 'right_hip' in keypoints:
        shoulder_angle = calculate_angle(keypoints['left_shoulder'], keypoints['right_shoulder'], keypoints['left_hip'])
        hip_angle = calculate_angle(keypoints['left_hip'], keypoints['right_hip'], keypoints['right_shoulder'])
        return (shoulder_angle, hip_angle)
    
    return None


# Function to calculate stability (e.g., variance in body position or keypoint consistency)
def calculate_stability(keypoints, previous_keypoints):
    """
    Stability can be tracked as the variance in keypoint position over time.
    
    Args:
    - keypoints: Current keypoints
    - previous_keypoints: Previous keypoints
    
    Returns:
    - stability: A measure of stability (e.g., total displacement of keypoints over time)
    """
    total_displacement = 0.0
    for joint in keypoints:
        if joint in previous_keypoints:
            curr_pos = np.array(keypoints[joint][:2])
            prev_pos = np.array(previous_keypoints[joint][:2])
            total_displacement += np.linalg.norm(curr_pos - prev_pos)
    
    return total_displacement
