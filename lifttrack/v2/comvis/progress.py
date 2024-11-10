import cv2
import numpy as np
import base64
from io import BytesIO
from lifttrack.v2.comvis.Live import predicted_class_name
import websocket

# Function to calculate form accuracy and provide suggestions
def calculate_form_accuracy(features, predicted_class_name):
    """
    Calculate form accuracy based on extracted features for a given exercise class.
    
    Args:
    - features: Dictionary containing extracted features (angles, movement, speed, alignment, stability).
    - predicted_class_name: The name of the predicted exercise class (e.g., 'barbell_benchpress').
    
    Returns:
    - accuracy: A numerical accuracy score (between 0 and 1).
    - suggestions: A list of suggestions for improving form.
    """
    accuracy = 1.0  # Start with 100% accuracy
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

    # Add other classes as needed

    return accuracy, suggestions

# Function to convert an image to base64 format
def img_to_base64(image):
    """
    Converts an image (numpy array) to base64.
    
    Args:
    - image: The image to be converted.
    
    Returns:
    - base64 string representing the image.
    """
    _, buffer = cv2.imencode('.jpg', image)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64

async def frame_by_frame_analysis(features, final_annotated_frame, class_names, websocket):
    """
    Analyzes and sends the frame-by-frame replay with form accuracy and suggestions via WebSocket.
    Buffers 30 frames before sending them as a batch.
    
    Args:
    - features: List of features for each frame from analyze_annotations
    - final_annotated_frame: The final annotated frame from analyze_annotations
    - class_names: Dictionary mapping class indices to class names
    - websocket: The WebSocket connection to send messages
    """
    # Initialize message buffer
    message_buffer = []
    BUFFER_SIZE = 30

    for frame_index, feature in enumerate(features):
        frame_path = feature['frame_path']
        frame = cv2.imread(frame_path)
        
        if frame is None:
            continue
        
        # Extract relevant features for form analysis
        analysis_features = {
            'angles': feature['joint_angles'],
            'speed': feature['speeds'],
            'alignment': feature['body_alignment'],
            'stability': {'core_stability': feature['stability']}
        }

        # Get the predicted class name
        predicted_class_index = feature.get('predicted_class_index', 0)  # Default to 0 if not present
        predicted_class_name = class_names.get(predicted_class_index, "unknown")

        # Calculate form accuracy and provide suggestions
        accuracy, suggestions = calculate_form_accuracy(analysis_features, predicted_class_name)

        # Display the frame with keypoints (if available in final_annotated_frame)
        frame_with_keypoints = frame
        if final_annotated_frame is not None:
            frame_with_keypoints = final_annotated_frame

        # Convert the frame to base64
        frame_b64 = img_to_base64(frame_with_keypoints)

        # Prepare the message
        message = {
            'frame_b64': frame_b64,
            'accuracy': accuracy,
            'suggestions': suggestions,
            'frame_index': frame_index,
            'predicted_class_name': predicted_class_name
        }

        # Add message to buffer
        message_buffer.append(message)

        # When buffer reaches BUFFER_SIZE, send the batch
        if len(message_buffer) >= BUFFER_SIZE:
            await websocket.send_json({
                'batch': message_buffer,
                'batch_size': len(message_buffer),
                'total_frames': len(features)
            })
            # Clear the buffer
            message_buffer = []

    # Send any remaining messages in the buffer
    if message_buffer:
        await websocket.send_json({
            'batch': message_buffer,
            'batch_size': len(message_buffer),
            'total_frames': len(features)
        })