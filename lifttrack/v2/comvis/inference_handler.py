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
from lifttrack.utils.logging_config import setup_logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = setup_logger("inference_handler", "inference_handler.log")

def convert_byte_to_numpy(byte_data: bytes | Any) -> np.ndarray:
    """Convert a byte array or any other type to a numpy array."""
    try:
        if not isinstance(byte_data, bytes):
            logger.error(f"Expected byte array, got {type(byte_data)}")
            raise ValueError("Input must be a byte array or numpy array")
        return cv2.imdecode(np.frombuffer(byte_data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Failed to decode frame: {str(e)}")
        raise

def process_frame(frame: np.ndarray):
    """Process a single frame and return resized version in WebP format."""
    try:
        frame = cv2.resize(frame, (192, 192))
        _, buffer = cv2.imencode(".jpeg", frame, [cv2.IMWRITE_JPEG_OPTIMIZE, 85])
        return frame, buffer
    except Exception as e:
        logger.error(f"Failed to process frame: {str(e)}")
        raise

def perform_frame_analysis(frames_buffer: Any):
    """
    Perform parallel frame analysis on the input frames.
    """
    try:
        if len(frames_buffer) < 2:
            return None, None, None, None

        start_time = time.time()

        # Run models in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks to the executor
            class_name_future = executor.submit(three_dim_inference.predict_class, frames_buffer)
            current_pose_future = executor.submit(movenet_inference.analyze_frame, frames_buffer[-1])
            detected_object_future = executor.submit(object_tracker.process_frames_and_get_annotations, frames_buffer[-1])
            previous_pose_future = executor.submit(movenet_inference.analyze_frame, frames_buffer[-2])
            
            # Get results from futures
            class_name = class_name_future.result()
            _, current_pose = current_pose_future.result()
            detected_object = detected_object_future.result()
            _, previous_pose = previous_pose_future.result()

        # Log processing time
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        logger.debug(f"Frame processing time: {processing_time:.3f}s ({fps:.1f} FPS)")

        return current_pose, previous_pose, detected_object, class_name

    except Exception as e:
        logger.error(f"Failed to perform frame analysis: {str(e)}")
        raise


def load_to_object_model(object_inference: list) -> Object:
    """Load an object inference to an Object model."""
    try:
        # Handle empty inference case
        if not object_inference:
            return Object(
                classs_id=-1,
                type="unknown",
                confidence=0.0,
                x=0.0,
                y=0.0,
                width=0.0,
                height=0.0,
            )
            
        best_confidence = max(object_inference, key=lambda x: x.get('confidence', 0))
        return Object(
            classs_id=best_confidence.get('class_id', 0),
            type=best_confidence.get('class', 'barbell'),  # Provide default value
            confidence=best_confidence.get('confidence', 0.0),
            x=best_confidence.get('x', 0.0),
            y=best_confidence.get('y', 0.0),
            width=best_confidence.get('width', 0.0),
            height=best_confidence.get('height', 0.0),
        )
    except Exception as e:
        logger.error(f"Failed to load object inference: {str(e)}")
        raise
    
def load_to_features_model(previous_pose, current_pose, object_inference, class_name):
    """Save features in a Features model"""
    try:
        if not isinstance(current_pose, dict):
            raise TypeError("current_pose must be a dictionary")
        if not isinstance(previous_pose, dict):
            raise TypeError("previous_pose must be a dictionary")
        if not isinstance(object_inference, Object):
            raise TypeError("object_inference must be an Object base model")
        
        object_inference = object_inference.model_dump()
        joint_angles = extract_joint_angles(current_pose)
        movement_pattern = extract_movement_patterns(current_pose, previous_pose)
        speeds = calculate_speed(movement_pattern)
        body_alignment = extract_body_alignment(current_pose)
        stability = calculate_stability(current_pose, previous_pose)
            
        return Features(
            objects=object_inference if isinstance(object_inference, dict) else {},
            joint_angles=joint_angles,
            movement_pattern=class_name,
            speeds=speeds,
            body_alignment=body_alignment,
            stability=stability,
        )
    except Exception as e:
        logger.error(f"Failed to load features: {str(e)}")
        raise

def get_suggestions(features: Features, class_name: str):
    """Get suggestions for a given class name and features."""
    try:
        _features = features.model_dump()
        if not isinstance(_features, dict):
            logger.error("features must be a dictionary")
            raise TypeError("features must be a dictionary")
        
        # Normalize the class_name to match the format expected by calculate_form_accuracy
        normalized_class_name = class_name.lower().replace(" ", "_")
        
        accuracy, suggestions = calculate_form_accuracy(_features, normalized_class_name)
        logger.info(f"Form accuracy: {accuracy}, Suggestions: {suggestions}")
        # Join suggestions list into a single string, or return a default message if empty
        return (accuracy, " ".join(suggestions) if suggestions else "Form looks good! Keep it up!")
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e.__traceback__.tb_lineno}")
        # Return a default message instead of None when an exception occurs
        return "Unable to analyze form at this time. Please continue your exercise."

def load_to_exercise_data_model(features: Features, suggestions: str, frame_index: str, frame_id: str = None):
    """Save exercise data in an ExerciseData model"""
    try:
        return ExerciseData(
            frame=frame_index,
            suggestion=suggestions,
            features=features,
            frame_id=frame_id
        )
    except Exception as e:
        logger.error(f"Failed to load exercise data: {str(e)}")
        raise


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
    try:
        db.set_data(
            path=f'progress/{username}/{exercise_name.lower()}/{date}',
            data=exercise_data, 
            key=time_frame)
    except Exception as e:
        logger.error(f"Failed to save progress: {str(e)}")
        raise

def create_yuv420_frame(y_plane_bytes, u_plane_bytes, v_plane_bytes, width, height):
    """Create a BGR frame from YUV planes"""
    try:        
        # Convert Y plane to numpy array
        y = np.frombuffer(y_plane_bytes, dtype=np.uint8).reshape((height, width))
        
        # Create a grayscale image from Y plane as a fallback
        gray_image = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
        
        # Rotate the grayscale image if needed
        if width > height:  # If image is in landscape but should be portrait
            gray_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)
        
        y_size = width * height
        uv_ratio = len(u_plane_bytes) / y_size
        
        try:
            if 0.49 <= uv_ratio <= 0.51:
                # Calculate dimensions for U and V planes
                uv_width = width // 2
                uv_height = height // 2
                
                # Reshape U and V planes
                u = np.frombuffer(u_plane_bytes, dtype=np.uint8)[:uv_width * uv_height].reshape((uv_height, uv_width))
                v = np.frombuffer(v_plane_bytes, dtype=np.uint8)[:uv_width * uv_height].reshape((uv_height, uv_width))
                
                # Try NV12 format conversion first
                try:
                    # Create NV12 format (YUV420sp)
                    nv12 = np.zeros((height * 3 // 2, width), dtype=np.uint8)
                    nv12[0:height, :] = y
                    
                    # Create and copy interleaved UV plane
                    uv = np.zeros((height//2, width), dtype=np.uint8)
                    for i in range(height//2):
                        for j in range(width//2):
                            uv[i, j*2] = u[i, j]
                            uv[i, j*2+1] = v[i, j]
                    nv12[height:, :] = uv
                    
                    # Convert to BGR
                    nv12 = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
                    return cv2.rotate(nv12, cv2.ROTATE_90_CLOCKWISE) if width > height else nv12
                    
                except Exception as e:
                    logger.error(f"Error in NV12 conversion: {str(e)}")
                    
                    # Try I420 format as fallback
                    try:
                        # Create I420 format (YUV420p)
                        i420 = np.zeros((height * 3 // 2, width), dtype=np.uint8)
                        i420[0:height, :] = y
                        i420[height:height+height//4, :width] = cv2.resize(u, (width, height//4))
                        i420[height+height//4:, :width] = cv2.resize(v, (width, height//4))
                        
                        # Convert to BGR
                        i420 = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
                        return cv2.rotate(nv12, cv2.ROTATE_90_CLOCKWISE) if width > height else i420
                        
                    except Exception as e2:
                        logger.error(f"Error in I420 conversion: {str(e2)}")
                        
                        # Try direct conversion as last resort
                        try:
                            # Resize U and V to full resolution
                            u_full = cv2.resize(u, (width, height))
                            v_full = cv2.resize(v, (width, height))
                            
                            # Convert to float32 and adjust range
                            y_float = y.astype(np.float32)
                            u_float = (u_full.astype(np.float32) - 128) * 0.872
                            v_float = (v_full.astype(np.float32) - 128) * 1.230
                            
                            # Convert to BGR using matrix multiplication
                            b = y_float + 2.032 * u_float
                            g = y_float - 0.395 * u_float - 0.581 * v_float
                            r = y_float + 1.140 * v_float
                            
                            # Clip values and convert back to uint8
                            b = np.clip(b, 0, 255).astype(np.uint8)
                            g = np.clip(g, 0, 255).astype(np.uint8)
                            r = np.clip(r, 0, 255).astype(np.uint8)
                            
                            # Merge channels
                            nv12 = cv2.merge([b, g, r])
                            
                            # Rotate the BGR image if needed
                            if width > height:  # If image is in landscape but should be portrait
                                nv12 = cv2.rotate(nv12, cv2.ROTATE_90_CLOCKWISE)
                            logger.info("Successfully converted YUV to BGR using direct conversion")
                            return nv12
                        except Exception as e3:
                            logger.error(f"Error in direct conversion: {str(e3)}")
                            return cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE) if width > height else gray_image
            
            logger.warning("Could not convert YUV data, returning grayscale image")
            return cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE) if width > height else gray_image
            
        except Exception as e:
            logger.error(f"Error in YUV conversion: {str(e)}")
            return cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE) if width > height else gray_image
            
    except Exception as e:
        logger.error(f"Error creating YUV frame: {str(e)}")
        # Create a blank image with correct orientation
        if width > height:
            return np.zeros((width, height, 3), dtype=np.uint8)  # Swapped dimensions for portrait
        return np.zeros((height, width, 3), dtype=np.uint8)