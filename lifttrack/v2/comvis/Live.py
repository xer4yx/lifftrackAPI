import numpy as np
import cv2
import asyncio
import websockets
# Remove the resize functions and import them from utils
from lifttrack.v2.comvis.utils import resize_to_128x128, resize_to_192x192
from lifttrack.v2.comvis.analyze_features import analyze_annotations

class_names = {
    0: "benchpress",
    1: "deadlift",
    2: "romanian_deadlift",
    3: "shoulder_press",
}

def resize_to_128x128(input_array):
    """Resize a given numpy array to 128x128."""
    if isinstance(input_array, np.ndarray):
        return cv2.resize(input_array, (128, 128))
    else:
        raise ValueError("Input must be a numpy array")

def resize_to_192x192(input_array):
    """Resize a given numpy array to 192x192."""
    if isinstance(input_array, np.ndarray):
        return cv2.resize(input_array, (192, 192))
    else:
        raise ValueError("Input must be a numpy array")

def prepare_frames_for_input(frame_list, num_frames=30):
    if len(frame_list) != num_frames:
        # Adjust to only take the first num_frames if more are provided
        if len(frame_list) > num_frames:
            frame_list = frame_list[:num_frames]
        else:
            raise ValueError(f"Expected {num_frames} frames, but got {len(frame_list)}")
    
    frames_resized_128 = [resize_to_128x128(frame) for frame in frame_list]
    frames_resized_192 = [resize_to_192x192(frame) for frame in frame_list]
    
    frames_stack_128 = np.stack(frames_resized_128, axis=0)
    frames_stack_192 = np.stack(frames_resized_192, axis=0)
    
    return frames_stack_128, frames_stack_192

def predict_class(model, frame_list):
    frames_input, _ = prepare_frames_for_input(frame_list)
    frames_input_batch = np.expand_dims(frames_input, axis=0)
    predictions = model.predict(frames_input_batch)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_names[predicted_class_index]

def provide_form_suggestions(predicted_class_name, features):
    """
    Provides suggestions for improving form based on the predicted class name and extracted features.
    
    Args:
    - predicted_class_name: The predicted class name for the current frame.
    - features: A dictionary containing extracted features (joint_angles, movement_patterns, speeds, etc.).
    
    Returns:
    - suggestions: A string containing suggestions for form improvement.
    """
    suggestions = []

    # Get features with proper keys and default values
    joint_angles = features.get('joint_angles', {})
    body_alignment = features.get('body_alignment', {})
    movement_patterns = features.get('movement_patterns', {})
    speeds = features.get('speeds', {})
    stability = features.get('stability', 0.0)

    
    if predicted_class_name == "benchpress":
        if joint_angles.get('left_shoulder_left_elbow_left_wrist') is not None and \
           joint_angles.get('left_shoulder_left_elbow_left_wrist') > 45:
            suggestions.append("Keep your elbows tucked at a 45-degree angle to protect your shoulders.")
        
        if body_alignment.get('shoulder_angle') is not None and \
           body_alignment.get('shoulder_angle') > 10:
            suggestions.append("Ensure your shoulders are level during the lift.")
        
        if stability > 0.1:
            suggestions.append("Maintain a stable base to enhance your lift.")

    elif predicted_class_name == "deadlift":
        if stability > 0.1:
            suggestions.append("Focus on maintaining a stable position throughout the lift.")
        
        if joint_angles.get('left_hip_left_knee_left_ankle') is not None and \
           joint_angles.get('left_hip_left_knee_left_ankle') < 170:
            suggestions.append("Keep your hips lower than your shoulders to maintain proper form.")
        
        if speeds.get('left_hip', 0.0) > 2.0:
            suggestions.append("Control your speed to avoid injury.")

    elif predicted_class_name == "romanian_deadlift":
        if joint_angles.get('left_hip_left_knee_left_ankle') is not None and \
           joint_angles.get('left_hip_left_knee_left_ankle') < 160:
            suggestions.append("Ensure your back is flat and hinge at the hips.")
        
        if stability > 0.1:
            suggestions.append("Maintain stability to prevent rounding your back.")

    elif predicted_class_name == "shoulder_press":
        if speeds.get('left_shoulder', 0.0) > 2.0:
            suggestions.append("Control the speed of your lift to avoid injury.")
        
        if joint_angles.get('left_shoulder_left_elbow_left_wrist') is not None and \
           joint_angles.get('left_shoulder_left_elbow_left_wrist') > 90:
            suggestions.append("Keep your wrists straight and elbows aligned.")
        
        if stability > 0.1:
            suggestions.append("Maintain a stable core throughout the movement.")

    return "\n".join(suggestions)


async def analyze_frames(annotations):
    analysis_results = []
    max_frames = 1800
    
    for frame_index, annotation in enumerate(annotations[:max_frames]):
        if frame_index % 30 == 0:  # Still analyzing every 30th frame
            predicted_class_name = class_names[annotation['predicted_class_index']]
            
            # Extract features from the annotation using analyze_features
            features, _ = analyze_annotations([annotation])  # Call analyze_annotations to get features
            
            # Ensure features are structured correctly for provide_form_suggestions
            suggestions = provide_form_suggestions(predicted_class_name, features[0])  # Pass the first feature set
            
            analysis_results.append({
                'frame_index': frame_index,
                'class_name': predicted_class_name,
                'suggestions': suggestions
            })
    
    return {
        'frames': analysis_results,
        'total_frames': min(len(annotations), max_frames)
    }