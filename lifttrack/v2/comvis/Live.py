import numpy as np
import cv2
import asyncio
import websockets
import tensorflow as tf
from lifttrack import config
from lifttrack.utils.logging_config import setup_logger
from lifttrack.v2.comvis.analyze_features import analyze_annotations
from lifttrack.v2.comvis.utils import resize_to_128x128

CLASS_NAMES = {
    0: "benchpress",
    1: "deadlift",
    2: "romanian_deadlift",
    3: "shoulder_press",
}

logger = setup_logger("analyzer-v2", "comvis.log")


class ThreeDimInference:
    def __init__(self):
        try:
            self.__model = tf.keras.models.load_model(config.get('CNN', 'lifttrack_cnn'), compile=False)  # Don't load optimizer
            logger.info("Analyzer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading primary model: {e}")
            try:
                fallback_path = config.get('CNN', 'lifttrack_cnn_lite')
                self.__model = tf.keras.models.load_model(fallback_path, compile=False)
                logger.info(f"Analyzer model loaded successfully from fallback path: {fallback_path}")
            except Exception as e:
                logger.error(f"Error loading fallback model: {e}")
                raise RuntimeError("Failed to load both primary and fallback models")
            
    def prepare_frames_for_input(self, frame_list, num_frames=30):
        """
        Prepare frames for model input by resizing and stacking them.
        
        Args:
            frame_list: List of frames to process
            num_frames: Number of frames expected (default: 30)
        
        Returns:
            numpy array with shape (num_frames, height, width, 3)
        """
        if len(frame_list) == 0 or not isinstance(frame_list[0], np.ndarray):
            raise ValueError("frame_list must be a non-empty list of numpy arrays")

        # Handle temporal dimension
        if len(frame_list) != num_frames:
            if len(frame_list) > num_frames:
                frame_list = frame_list[:num_frames]
            else:
                last_frame = frame_list[-1]
                padding = [last_frame] * (num_frames - len(frame_list))
                frame_list = np.array(list(frame_list) + padding)  # Convert to numpy array after padding
        
        frames_resized = []
        for frame in frame_list:
            # Convert grayscale to RGB if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            # Convert BGR to RGB if needed (OpenCV uses BGR by default)
            elif frame.shape[-1] == 3 and not np.array_equal(frame[:,:,0], frame[:,:,2]):  # Check if BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Handle other cases (like RGBA)
            elif frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                
            # Resize to 128x128
            frame_resized = resize_to_128x128(frame)
            frames_resized.append(frame_resized)
        
        # Stack frames
        frames_stack = np.stack(frames_resized, axis=0)  # Shape: (num_frames, 128, 128, 3)
        
        # Verify final shape
        expected_shape = (num_frames, 128, 128, 3)
        if frames_stack.shape != expected_shape:
            raise ValueError(f"Unexpected output dimensions. Expected {expected_shape}, got {frames_stack.shape}")
        
        return frames_stack

    def predict_class(self, frame_list):
        frames_input = self.prepare_frames_for_input(frame_list)  # Only one value to unpack
        frames_input_batch = np.expand_dims(frames_input, axis=0)
        predictions = self.__model.predict(frames_input_batch)
        logger.info(f"Predictions: {predictions}")
        predicted_class_index = tf.compat.v1.argmax(predictions[0]).numpy()
        logger.info(f"Predicted class index: {predicted_class_index}")
        return CLASS_NAMES[predicted_class_index]

    def provide_form_suggestions(self, predicted_class_name, features):
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


    async def analyze_frames(self, annotations, object_tracker):
        analysis_results = []
        max_frames = 1800
        
        for frame_index, annotation in enumerate(annotations[:max_frames]):
            try:
                if frame_index % 30 == 0:  # Still analyzing every 30th frame
                    predicted_class_index = annotation['predicted_class_index']
                    predicted_class_name = CLASS_NAMES.get(predicted_class_index)
                    
                    if predicted_class_name is None:
                        raise ValueError(f"Invalid class index: {predicted_class_index}")
                    
                    # Extract features from the annotation using analyze_features
                    features, _ = analyze_annotations([annotation], object_tracker)
                    
                    # Ensure features are structured correctly for provide_form_suggestions
                    suggestions = self.provide_form_suggestions(predicted_class_name, features[0])
                    
                    analysis_results.append({
                        'frame_index': frame_index,
                        'class_name': predicted_class_name,
                        'suggestions': suggestions
                    })
            except Exception as e:
                print(f"Error processing frame {frame_index}: {str(e)}")
                continue
        
        return {
            'frames': analysis_results,
            'total_frames': min(len(annotations), max_frames)
        }
