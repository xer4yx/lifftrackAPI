from typing import Any, Dict
import numpy as np
import cv2
from lifttrack.v2.comvis import movenet_inference, object_tracker, three_dim_inference
from .features import (
    extract_joint_angles, 
    extract_movement_patterns, 
    calculate_speed, 
    extract_body_alignment, 
    calculate_stability
)
from .progress import calculate_form_accuracy
from lifttrack.models import Object, Features, ExerciseData
from lifttrack.v2.dbhelper.admin_rtdb import FirebaseDBHelper


def convert_byte_to_numpy(byte_data: bytes | Any) -> np.ndarray:
    """Convert a byte array or any other type to a numpy array."""
    if not isinstance(byte_data, bytes):
        raise ValueError("Input must be a byte array or numpy array")
    return cv2.imdecode(np.frombuffer(byte_data, np.uint8), cv2.IMREAD_COLOR)

def process_frame(frame: np.ndarray):
    """Process a single frame and return resized version."""
    frame = cv2.resize(frame, (192, 192))
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return frame, buffer


def perform_frame_analysis(frames_buffer: Any):
    """Perform frame analysis and return results."""
    if len(frames_buffer) < 2:
        return None, None, None, None

    # Pose Estimation Inference
    _, current_pose = movenet_inference.analyze_frame(frames_buffer[-1])
    _, previous_pose = movenet_inference.analyze_frame(frames_buffer[-2])

    # Object Detection Inference
    detected_object = object_tracker.process_frames_and_get_annotations(frames_buffer[-1])

    # 3D Classification Inference
    class_name = three_dim_inference.predict_class(frames_buffer)

    return current_pose, previous_pose, detected_object, class_name


def load_to_object_model(object_inference: list) -> Object:
    """Load an object inference to an Object model."""
    best_confidence = max(object_inference, key=lambda x: x.get('confidence', 0))
    return Object(
        classs_id=best_confidence.get('class_id', -1),
        type=best_confidence.get('class', "unknown"),
        confidence=best_confidence.get('confidence', 0.0),
        x=best_confidence.get('x', 0.0),
        y=best_confidence.get('y', 0.0),
        width=best_confidence.get('width', 0.0),
        height=best_confidence.get('height', 0.0),
    )

def load_to_features_model(previous_pose, current_pose, object_inference):
    """Save features in a Features model"""
    return Features(
        objects=object_inference,
        joint_angles=extract_joint_angles(current_pose),
        movement_pattern=extract_movement_patterns(current_pose, previous_pose),
        speeds=calculate_speed(current_pose, previous_pose),
        body_alignment=extract_body_alignment(current_pose),
        stability=calculate_stability(current_pose, previous_pose),
    )


def get_suggestions(features: Features, class_name: str):
    """Get suggestions for a given class name and features."""
    _, suggestions = calculate_form_accuracy(features.model_dump(), class_name)
    return suggestions


def load_to_exercise_data_model(features: Features, suggestions: str, frame_index: str):
    """Save exercise data in an ExerciseData model"""
    return ExerciseData(
        frame=frame_index,
        suggestion=suggestions,
        features=features
    )


def format_date(date: str):
    """Format a date string"""
    exercise_datetime = date.split('.')[0]
    return exercise_datetime.replace(':', '-')

def save_progress(
    username: str,
    exercise_name: str,
    date: str,
    time_frame: str, 
    exercise_data: Dict[str, Any], 
    db: FirebaseDBHelper):
    """Save progress to the database."""
    db.set_data(path=f'progress/{username}/{exercise_name}/{date}', data=exercise_data, key=time_frame)
    