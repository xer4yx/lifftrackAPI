import asyncio
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import cv2
import numpy as np
import tensorflow as tf
from lifttrack.utils.logging_config import setup_logger

from lifttrack.v2.comvis.Live import (
    model,
    class_names,
    resize_to_128x128,
    predict_class,
    provide_form_suggestions,
    prepare_frames_for_input,
)

logger = setup_logger("router", "lifttrack_websocket.log")

from lifttrack.v2.comvis.Movenet import analyze_frame  # This handles 192x192 internally
from lifttrack.v2.comvis.object_track import process_frames_and_get_annotations
from lifttrack.v2.comvis.features import extract_joint_angles, extract_movement_patterns, calculate_speed, extract_body_alignment, calculate_stability
from lifttrack.v2.comvis.analyze_features import analyze_annotations
from lifttrack.v2.comvis.progress import calculate_form_accuracy
from lifttrack import config

from datetime import datetime
from lifttrack.models import ExerciseData, Features
from lifttrack.dbhandler.rtdbHelper import rtdb

router = APIRouter()

@router.websocket("/v2/ws-tracking")
async def websocket_endpoint(
    websocket: WebSocket,
    username: str = Query(..., description="Username of the person exercising"),
    exercise_name: str = Query(..., description="Name of the exercise being performed")
    ):
    await websocket.accept()
    
    frames_buffer = []  # Initialize buffer
    
    try:
        if not username or not exercise_name:
            raise ValueError("Both username and exercise_name are required")
            
        logger.info(f"Starting tracking session for user: {username}, exercise: {exercise_name}")
    except Exception as e:
        logger.error(f"Failed to get initialization data: {str(e)}")
        frames_buffer.clear()
        await websocket.close()
        return

    frame_count = 0
    max_frames = 1800
    last_frame_time = asyncio.get_event_loop().time()
    frame_timeout = 60.0
    
    try:
        while frame_count < max_frames:
            try:
                # Check for timeout
                current_time = asyncio.get_event_loop().time()
                if current_time - last_frame_time > frame_timeout:
                    logger.info("Frame reception timeout - no frames received for 5 seconds")
                    frames_buffer.clear()  # Clear buffer on timeout
                    await websocket.send_json({
                        'type': 'complete',
                        'message': 'Session ended - no frames received for 5 seconds',
                        'frame_count': frame_count
                    })
                    break

                # Receive frame with timeout
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_bytes(),
                        timeout=frame_timeout
                    )
                    last_frame_time = asyncio.get_event_loop().time()
                except asyncio.TimeoutError:
                    logger.info("Frame reception timeout")
                    frames_buffer.clear()  # Clear buffer on timeout
                    break

                logger.info(f"Received frame data of length: {len(data)} bytes")

                # Process the received frame data
                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # Check if the frame is empty
                if frame is None or frame.size == 0:
                    logger.error("Received an empty frame, skipping encoding.")
                    continue  # Skip the rest of the loop if the frame is empty
                
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
                logger.info(f"Frame count: {frame_count}")

                # Process analysis every 30th frame
                if frame_count % 30 == 0:
                    try:
                        # 1. Get MoveNet and Roboflow inference
                        _, keypoints = analyze_frame(frame)
                        logger.info(f"Got MoveNet results")
                        
                        roboflow_results = process_frames_and_get_annotations(frame)
                        logger.info(f"Got Roboflow results")
                        
                        if not roboflow_results:
                            predictions = []
                        else:
                            predictions = roboflow_results.get('predictions', [])

                        # 2. Get predicted class
                        predicted_class_name = predict_class(model, frames_buffer[-30:])
                        logger.info(f"Got predicted class")

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
                        logger.info(f"Got form accuracy and suggestions")

                        # Create Features object
                        features_obj = Features(
                            objects=str(predictions),
                            joint_angles=features['joint_angles'],
                            movement_pattern=predicted_class_name,
                            speeds=features['speeds'],
                            body_alignment=features['body_alignment'],
                            stability=features['stability']
                        )

                        # Create ExerciseData object without explicit date parameter
                        exercise_data = ExerciseData(
                            suggestion=suggestions[0] if suggestions else "No suggestions",
                            features=features_obj,
                            frame=f"frame_{frame_count}"
                        )

                        # Store in database
                        rtdb.put_progress(
                            username=username,
                            exercise_name=exercise_name.lower(),
                            exercise_data=exercise_data
                        )

                        # Send analysis results
                        await websocket.send_json({
                            'suggestions': suggestions,
                        })

                        frames_buffer.clear()  # Clear buffer after successful analysis
                        logger.info(f"Cleared buffer and stored progress data")
                    except Exception as e:
                        logger.error(f"Error during analysis: {str(e)}")
                        frames_buffer.clear()  # Clear buffer on analysis error
                        continue

            except WebSocketDisconnect:
                logger.info(f"Client disconnected after processing {frame_count} frames")
                frames_buffer.clear()  # Clear buffer on disconnect
                break
            except Exception as e:
                logger.error(f"Error in frame processing: {str(e)}")
                frames_buffer.clear()  # Clear buffer on error
                continue

        # Session completion
        if not websocket.client_state.DISCONNECTED:
            frames_buffer.clear()  # Clear buffer before closing
            await websocket.send_json({
                'type': 'complete',
                'message': 'Session ended successfully',
                'frame_count': frame_count,
                'reason': 'max_frames_reached' if frame_count >= max_frames else 'timeout'
            })
            await websocket.close()

    except WebSocketDisconnect:
        logger.info(f"Client disconnected after processing {frame_count} frames")
        frames_buffer.clear()  # Clear buffer on disconnect
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        frames_buffer.clear()  # Clear buffer on error
        if not websocket.client_state.DISCONNECTED:
            await websocket.send_json({
                'type': 'error',
                'message': str(e),
                'frame_count': frame_count
            })
            await websocket.close()
    finally:
        # Ensure buffer is cleared even if we hit an unexpected error
        frames_buffer.clear()
        logger.info("WebSocket connection closed and buffer cleared")