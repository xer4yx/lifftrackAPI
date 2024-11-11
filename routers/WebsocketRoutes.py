import asyncio
import base64
from fastapi import WebSocket, APIRouter
import cv2
import numpy as np
import tensorflow as tf

from lifttrack.v2.comvis.Live import (
    resize_to_128x128,
    predict_class,
    provide_form_suggestions,
    prepare_frames_for_input,
)
from lifttrack.v2.comvis.Movenet import analyze_frame  # This handles 192x192 internally
from lifttrack.v2.comvis.object_track import process_frames_and_get_annotations
from lifttrack.v2.comvis.features import extract_joint_angles, extract_movement_patterns, calculate_speed, extract_body_alignment, calculate_stability
from lifttrack.v2.comvis.analyze_features import analyze_annotations
from lifttrack.v2.comvis.progress import frame_by_frame_analysis
from lifttrack import config

router = APIRouter()

# Load the Live.py model once when the router starts
model = tf.keras.models.load_model(config.get('CNN', 'path'), compile=False)  # Don't load optimizer

@router.websocket("/v2/ws-tracking")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer_size = 30
    features_buffer = []
    frames_buffer_128 = []  # For Live.py predictions
    previous_keypoints = None
    class_names = {
        0: "benchpress",
        1: "deadlift",
        2: "romanian_deadlift",
        3: "shoulder_press",
}

    try:
        while True:
            # Receive and decode image
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            original_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if original_frame is None:
                await websocket.send_text("Failed to decode image.")
                continue

            try:
                # Pipeline 1: Live.py (128x128) for exercise classification
                frame_128 = resize_to_128x128(original_frame)
                frames_buffer_128.append(frame_128)

                current_prediction = None
                
                # Ensure that you are only processing the correct number of frames
                if len(frames_buffer_128) >= buffer_size:
                    # Prepare frames for input
                    frames_input_128, frames_input_192 = prepare_frames_for_input(frames_buffer_128[:buffer_size])
                    # Process the frames
                    predicted_class = predict_class(model, frames_buffer_128)
                    current_prediction = class_names[predicted_class]
                    frames_buffer_128 = frames_buffer_128[buffer_size:]  # Clear buffer after prediction

                # Pipeline 2: MoveNet (192x192) for pose estimation and analysis
                annotated_frame, keypoints = analyze_frame(original_frame)  # Handles 192x192 resize internally
                
                # Extract features from pose
                features = {
                    'joint_angles': extract_joint_angles(keypoints),
                    'body_alignment': extract_body_alignment(keypoints),
                }
                
                if previous_keypoints:
                    movement_patterns = extract_movement_patterns(keypoints, previous_keypoints)
                    features.update({
                        'movement_patterns': movement_patterns,
                        'speeds': calculate_speed(movement_patterns),
                        'stability': calculate_stability(keypoints, previous_keypoints)
                    })
                
                previous_keypoints = keypoints
                
                # Object detection and feature analysis
                frame_path = f"temp_frame_{len(features_buffer)}.jpg"
                cv2.imwrite(frame_path, original_frame)
                annotations, annotated_frame = process_frames_and_get_annotations(frame_path, analyze_frame)
                
                # Add to features buffer
                features_buffer.append({
                    'features': features,
                    'annotations': annotations[0] if annotations else None,
                    'frame': annotated_frame,
                    'prediction': current_prediction  # Include current prediction if available
                })

                # Process complete buffer
                if len(features_buffer) >= buffer_size:
                    # Analyze features
                    analyzed_features, final_frame = analyze_annotations(features_buffer[:buffer_size])  # Use only the first 30 frames
                    features_buffer = features_buffer[buffer_size:]  # Clear processed frames from the buffer
                    
                    # Get form suggestions if we have a prediction
                    form_suggestions = []
                    if current_prediction:
                        form_suggestions = provide_form_suggestions(
                            current_prediction,
                            analyzed_features
                        )
                    
                    # Get progress analysis
                    progress_suggestions = frame_by_frame_analysis(
                        analyzed_features,
                        final_frame,
                        class_names,
                        websocket.url_for_path("/")
                    )
                    
                    # Send combined results
                    await websocket.send_json({
                        'predicted_class': current_prediction,
                        'form_suggestions': form_suggestions,
                        'progress_suggestions': progress_suggestions,
                        'frame_count': len(features_buffer)
                    })
                    
                    # Clear buffers
                    features_buffer = []

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                await websocket.send_text(f"Error processing frame: {str(e)}")
                # Reset all buffers when an error occurs
                frames_buffer_128 = []
                features_buffer = []
                previous_keypoints = None
                continue

    except Exception as e:
        print(f"WebSocket connection closed: {e}")
        await websocket.close()