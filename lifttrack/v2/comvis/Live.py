import numpy as np
import cv2
import asyncio
import websockets
from lifttrack.v2.comvis.features import extract_features_from_annotations

class_names = {
    0: "barbell_benchpress",
    1: "barbell_deadlift",
    2: "barbell_rdl",
    3: "barbell_shoulderpress",
    4: "dumbbell_benchpress",
    5: "dumbbell_deadlift",
    6: "dumbbell_shoulderpress",
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
    - features: A dictionary containing extracted features (angles, movement, speed, alignment, stability).
    
    Returns:
    - suggestions: A string containing suggestions for form improvement.
    """
    suggestions = []

    # Analyze features based on predicted class
    if predicted_class_name == "barbell_benchpress":
        angles = features['angles']
        alignment = features['alignment']
        
        # Check for shoulder alignment and elbow positioning
        if angles.get('left_shoulder_left_elbow_left_wrist') > 45:  # Example threshold
            suggestions.append("Keep your elbows tucked at a 45-degree angle to protect your shoulders.")
        
        if alignment:
            shoulder_angle, _ = alignment
            if shoulder_angle > 10:  # Example threshold for shoulder alignment
                suggestions.append("Ensure your shoulders are level during the lift.")
        
        # Check stability
        if features['stability'] > 0.1:  # Example threshold for stability
            suggestions.append("Maintain a stable base to enhance your lift.")

    elif predicted_class_name == "barbell_deadlift":
        angles = features['angles']
        stability = features['stability']
        
        if stability > 0.1:  # Example threshold for stability
            suggestions.append("Focus on maintaining a stable position throughout the lift.")
        
        if angles.get('left_hip_left_knee_left_ankle') < 170:  # Example threshold for hip angle
            suggestions.append("Keep your hips lower than your shoulders to maintain proper form.")
        
        # Speed check
        speed = features['speed']
        if speed.get('left_hip') > 2.0:  # Example speed threshold
            suggestions.append("Control your speed to avoid injury.")

    elif predicted_class_name == "barbell_rdl":
        # Implement checks specific to Romanian deadlift
        if angles.get('left_hip_left_knee_left_ankle') < 160:  # Example threshold
            suggestions.append("Ensure your back is flat and hinge at the hips.")
        
        if features['stability'] > 0.1:
            suggestions.append("Maintain stability to prevent rounding your back.")

    elif predicted_class_name == "barbell_shoulderpress":
        # Implement checks specific to shoulder press
        if features['speed'].get('left_shoulder') > 2.0:  # Example speed threshold
            suggestions.append("Control the speed of your lift to avoid injury.")
        
        if angles.get('left_shoulder_left_elbow_left_wrist') > 90:  # Example threshold
            suggestions.append("Keep your wrists straight and elbows aligned.")

    elif predicted_class_name == "dumbbell_benchpress":
        # Implement checks specific to dumbbell bench press
        if angles.get('left_shoulder_left_elbow_left_wrist') > 45:  # Example threshold
            suggestions.append("Ensure your wrists are straight and elbows are at a 45-degree angle.")
        
        if features['stability'] > 0.1:
            suggestions.append("Maintain stability to enhance your lift.")

    elif predicted_class_name == "dumbbell_deadlift":
        # Implement checks specific to dumbbell deadlift
        if features['alignment'] and features['alignment'][1] > 10:  # Example threshold
            suggestions.append("Keep your back straight and chest up throughout the lift.")
        
        if features['stability'] > 0.1:
            suggestions.append("Focus on stability to prevent injury.")

    elif predicted_class_name == "dumbbell_shoulderpress":
        # Implement checks specific to dumbbell shoulder press
        if features['stability'] > 0.1:  # Example threshold
            suggestions.append("Maintain a stable base and avoid arching your back.")

    return "\n".join(suggestions)


async def analyze_frames(annotations):
    analysis_results = []
    max_frames = 1800
    
    for frame_index, annotation in enumerate(annotations[:max_frames]):
        if frame_index % 30 == 0:  # Still analyzing every 30th frame
            predicted_class_name = class_names[annotation['predicted_class_index']]
            features = extract_features_from_annotations([annotation])[0]
            suggestions = provide_form_suggestions(predicted_class_name, features)
            
            analysis_results.append({
                'frame_index': frame_index,
                'class_name': predicted_class_name,
                'suggestions': suggestions
            })
    
    return {
        'frames': analysis_results,
        'total_frames': min(len(annotations), max_frames)
    }
