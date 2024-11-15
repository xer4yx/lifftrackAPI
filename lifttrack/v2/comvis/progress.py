import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from lifttrack.v2.comvis.Live import predict_class
from lifttrack.utils.logging_config import setup_logger

logger = setup_logger("progress-v2", "comvis.log")

# Function to display the keypoints on the frame
def display_keypoints_on_frame(frame, keypoints, threshold=0.5):
    for joint, (x, y, score) in keypoints.items():
        if score > threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(frame, joint, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

def calculate_form_accuracy(features, predicted_class_name):
    print(f"Features: {features}")
    accuracy = 1.0
    suggestions = []

    if predicted_class_name == "barbell_benchpress":
        angles = features['angles']
        if angles.get('left_shoulder_left_elbow_left_wrist', 0) > 45:
            accuracy -= 0.1
            suggestions.append("Elbows should be at a 45-degree angle.")
            
    elif predicted_class_name == "barbell_deadlift":
        speed = features['speed']
        if speed.get('left_hip', 0) > 2.0:
            accuracy -= 0.1
            suggestions.append("Control the speed of your movement.")

    elif predicted_class_name == "barbell_rdl":
        stability = features['stability']
        if stability.get('core_stability', 0) < 0.8:
            accuracy -= 0.1
            suggestions.append("Maintain core stability throughout the lift.")

    elif predicted_class_name == "barbell_shoulderpress":
        alignment = features['alignment']
        if alignment.get('head_to_hips', 0) < 0.9:
            accuracy -= 0.1
            suggestions.append("Keep alignment from head to hips for balance.")

    elif predicted_class_name == "dumbbell_benchpress":
        angles = features['angles']
        if angles.get('left_elbow_left_wrist', 0) > 45:
            accuracy -= 0.1
            suggestions.append("Keep elbows closer to a 45-degree angle.")

    elif predicted_class_name == "dumbbell_deadlift":
        alignment = features['alignment']
        if alignment.get('spine_angle', 0) < 0.85:
            accuracy -= 0.1
            suggestions.append("Maintain a neutral spine angle.")

    elif predicted_class_name == "dumbbell_shoulderpress":
        stability = features['stability']
        if stability.get('shoulder_stability', 0) < 0.75:
            accuracy -= 0.1
            suggestions.append("Keep shoulder stability through the lift.")
    logger.info(f"Form accuracy: {accuracy}, Suggestions: {suggestions}")
    return accuracy, suggestions

# Function to convert an image to base64 format
def img_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64

# def frame_by_frame_analysis(annotations, final_annotated_frame, class_names, base_url):
#     analysis_results = []
#     # Remove the analyze_annotations call and directly use the annotations parameter
#     features = annotations  

#     for frame_index, feature in enumerate(features):
#         frame_path = feature['frame_path']
#         frame = cv2.imread(frame_path)

#         analysis_features = {
#             'angles': feature['joint_angles'],
#             'speed': feature['speeds'],
#             'alignment': feature['body_alignment'],
#             'stability': {'core_stability': feature['stability']}
#         }

#         accuracy, suggestions = calculate_form_accuracy(analysis_features, predicted_class_name)
#         frame_with_keypoints = display_keypoints_on_frame(frame, feature['keypoints'])
#         frame_b64 = img_to_base64(frame_with_keypoints)

#         analysis_results.append({
#             'frame_b64': frame_b64,
#             'accuracy': accuracy,
#             'suggestions': suggestions,
#             'frame_index': frame_index,
#             'predicted_class_name': predicted_class_name
#         })

#     return {
#         'frames': analysis_results,
#         'total_frames': len(features)
#     }
