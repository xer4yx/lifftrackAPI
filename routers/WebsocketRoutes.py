import asyncio
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import tensorflow as tf

from lifttrack.v2.comvis.Live import (
    model,
    class_names,
    resize_to_128x128,
    predict_class,
    provide_form_suggestions,
    prepare_frames_for_input,
)

from lifttrack.v2.comvis.Movenet import analyze_frame  # This handles 192x192 internally
from lifttrack.v2.comvis.object_track import process_frames_and_get_annotations
from lifttrack.v2.comvis.features import extract_joint_angles, extract_movement_patterns, calculate_speed, extract_body_alignment, calculate_stability
from lifttrack.v2.comvis.analyze_features import analyze_annotations
from lifttrack.v2.comvis.progress import calculate_form_accuracy
from lifttrack import config

router = APIRouter()

@router.websocket("/v2/ws-tracking")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0
    frames_buffer = []
    
    try:
        while True:
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Immediately send back the original frame bytes
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_json({
                'frame': frame_bytes,
                'type': 'frame'
            })
            
            # Add frame to buffer for analysis
            frames_buffer.append(frame)
            frame_count += 1

            # Process analysis every 30th frame
            if frame_count % 30 == 0:
                # 1. Get MoveNet and Roboflow inference
                _, keypoints = analyze_frame(frame)
                roboflow_results = process_frames_and_get_annotations(frame, analyze_frame)[0]
                predictions = roboflow_results['predictions']

                # 2. Get predicted class from Live.py
                predicted_class_name = predict_class(model, frames_buffer[-30:])

                # 3. Extract features
                features = {
                    'joint_angles': extract_joint_angles(keypoints),
                    'speeds': calculate_speed(extract_movement_patterns(keypoints, keypoints)),
                    'body_alignment': extract_body_alignment(keypoints),
                    'stability': calculate_stability(keypoints, keypoints),
                    'object_detections': predictions
                }

                # 4. Calculate accuracy and get suggestions
                accuracy, suggestions = calculate_form_accuracy(features, predicted_class_name)

                # Send analysis results
                await websocket.send_json({
                    'type': 'analysis',
                    'accuracy': accuracy,
                    'suggestions': suggestions,
                    'predicted_class': predicted_class_name
                })

                # Clear buffer
                frames_buffer = []

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {str(e)}")
        if not websocket.client_state.DISCONNECTED:
            await websocket.close()