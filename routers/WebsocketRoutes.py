import asyncio
import base64
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Request
from lifttrack.auth import get_current_user  # Import the get_current_user function
from lifttrack.dbhandler.rtdbHelper import rtdb  # Import your RTDBHelper instance
import cv2
import numpy as np
import tensorflow as tf
from lifttrack.utils.logging_config import setup_logger
from lifttrack.models import User

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
from lifttrack.dbhandler.rtdbHelper import rtdb  # Import your Firebase DB handler

router = APIRouter()

@router.websocket("/v2/ws-tracking")
async def websocket_endpoint(websocket: WebSocket, current_user: User = Depends(get_current_user), request: Request):
    await websocket.accept()
    
    username = current_user.username

    # Example of using RTDBHelper to get user progress
    user_progress = rtdb.get_progress(username)
    if user_progress:
        logger.info(f"Retrieved progress for user {username}: {user_progress}")
    else:
        logger.warning(f"No progress data found for user {username}")
    
    frame_count = 0
    frames_buffer = []
    max_frames = 1800  # Maximum number of frames to process (60 seconds at 30fps)
    last_frame_time = asyncio.get_event_loop().time()
    frame_timeout = 60.0  # 5 seconds timeout for receiving frames
    
    try:
        while frame_count < max_frames:
            try:
                # Add timeout for receiving frames
                current_time = asyncio.get_event_loop().time()
                if current_time - last_frame_time > frame_timeout:
                    logger.info("Frame reception timeout - no frames received for 5 seconds")
                    await websocket.send_json({
                        'type': 'complete',
                        'message': 'Session ended - no frames received for 5 seconds',
                        'frame_count': frame_count
                    })
                    break

                # Set timeout for receiving the next frame
                data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=frame_timeout
                )
                
                # Decode the received bytes to a string and parse as JSON
                decoded_data = data.decode('utf-8')  # Decode bytes to string
                json_data = json.loads(decoded_data)  # Parse string as JSON
                exercise = json_data.get('exercise')  # Access the 'exercise' field
                current_date = json_data.get('date')  # Access the 'date' field
                
                logger.info(f"Received exercise: {exercise}, date: {current_date}")

                last_frame_time = asyncio.get_event_loop().time()
                
            except asyncio.TimeoutError:
                logger.info("Frame reception timeout")
                await websocket.send_json({
                    'type': 'complete',
                    'message': 'Session ended - timeout',
                    'frame_count': frame_count
                })
                break
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
                
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

                    # Construct the progress data
                    progress_data = {
                        "username": current_user.username,  # Assuming you have access to current_user
                        "exercise": {
                            "squat": [
                                {
                                    "date": current_date,  # Use the date from the received data
                                    "suggestion": suggestions,  # Use the suggestions from analysis
                                    "features": features,  # Use the extracted features
                                    "frame": frame_bytes  # Use the frame bytes
                                }
                            ]
                        }
                    }

                    # Send progress data to Firebase
                    rtdb.put_progress(current_user.username, "squat", progress_data)  # Adjust as necessary

                    # Send analysis results
                    await websocket.send_json({
                        'type': 'analysis',
                        'accuracy': accuracy,
                        'suggestions': suggestions,
                        'predicted_class': predicted_class_name,
                        'frame_count': frame_count,
                        'max_frames': max_frames
                    })
                    logger.info(f"Sent analysis results")
                    frames_buffer = []
                    logger.info(f"Cleared buffer")
                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    continue

        # Only send completion if we haven't disconnected
        if not websocket.client_state.DISCONNECTED:
            await websocket.send_json({
                'type': 'complete',
                'message': 'Session ended successfully',
                'frame_count': frame_count,
                'reason': 'max_frames_reached' if frame_count >= max_frames else 'timeout'
            })
            logger.info(f"Session ended with {frame_count} frames processed")
            await websocket.close()

    except WebSocketDisconnect:
        logger.info(f"Client disconnected after processing {frame_count} frames")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if not websocket.client_state.DISCONNECTED:
            await websocket.send_json({
                'type': 'error',
                'message': str(e),
                'frame_count': frame_count
            })
            await websocket.close()