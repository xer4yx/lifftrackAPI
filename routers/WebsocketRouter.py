import asyncio
import base64
import cv2
import numpy as np
from typing import Annotated
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from .management import ConnectionManager

from lifttrack.comvis import websocket_process_frames
from lifttrack.v2.comvis import (
    movenet_inference, 
    object_tracker, 
    three_dim_inference
)
from lifttrack.v2.comvis.features import (
    extract_joint_angles, 
    extract_movement_patterns, 
    calculate_speed, 
    extract_body_alignment, 
    calculate_stability
)
from lifttrack.v2.comvis.progress import calculate_form_accuracy


from infrastructure import get_admin_firebase_db
from infrastructure.database import DatabaseFactory
from .management import ConnectionManager
from utilities.monitoring.factory import MonitoringFactory


from lifttrack.models import ExerciseData, Features, Object

frame_timeout = 30  # Default 60 second timeout
max_frames = 1800   # Default max frames (1 minute at 30fps)

router = APIRouter()
logger = MonitoringFactory.get_logger("websocket")
connection_manager = ConnectionManager()

# API Endpoint [Frame Operations]
@router.websocket(path="/ws-tracking")  # Mobile version
async def websocket_inference(
    websocket: WebSocket,
    user_id: str = Query(..., description="Unique identifier for the user")
):
    """
    Endpoint for the WebSocket video feed. The server receives frames from the client that's formatted as a base64
    bytes. The server decodes the frame in a thread pool, processes the frame to get inference and annotations, and
    encodes it back to the client as bytes back.

    Args:
        websocket: WebSocket object.
    """
    try:
        await connection_manager.connect(websocket, user_id)
        connection_open = True
        model = None
    
        while connection_open:
            try:
                # Handle both bytes and dict messages
                frame_data = await websocket.receive()

                # Check if it's a close message
                if frame_data["type"] == "websocket.close":
                    connection_open = False
                    break

                # Get frame bytes either from direct bytes or from dict
                frame_byte = frame_data.get("bytes")
                if not isinstance(frame_byte, bytes):
                    break

                # Process frame
                (annotated_frame, _) = await asyncio.get_event_loop().run_in_executor(
                    None, websocket_process_frames, model, frame_byte
                )

                # Encode and send response
                encoded, buffer = cv2.imencode(".jpeg", annotated_frame)
                if not encoded:
                    raise WebSocketDisconnect

                return_bytes = base64.b64decode(buffer.tobytes())
                await websocket.send_bytes(return_bytes)

            except WebSocketDisconnect:
                connection_open = False
            except Exception as e:
                logger.exception(f"WebSocket error: {e}")
                connection_open = False
    finally:
        await connection_manager.disconnect(user_id)


@router.websocket(path="/v2/ws-tracking")
async def websocket_endpoint(
    websocket: WebSocket,
    db: Annotated[DatabaseFactory, Depends(get_admin_firebase_db)],
    username: str = Query(..., description="Username of the person exercising"),
    exercise_name: str = Query(..., description="Name of the exercise being performed")
):
    await websocket.accept()
    
    try:
        await connection_manager.connect(websocket, username)
        
        if not username or not exercise_name:
            raise ValueError("Both username and exercise_name are required")
            
        logger.info(f"Starting tracking session for user: {username}, exercise: {exercise_name}")
        
        frame_count = 0
        max_frames = 1800
        last_frame_time = asyncio.get_event_loop().time()
        frame_timeout = 60.0
        
        previous_frame = 29
        frame_buffer_size = 30  # Only keep last 30 frames for analysis
        frames_buffer = []
        last_analysis_time = asyncio.get_event_loop().time()
        analysis_interval = 1.0  # Perform analysis every 1 second instead of every 30 frames
        
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

                # Process the received frame data more efficiently
                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None or frame.size == 0:
                    continue

                # Resize frame before processing to reduce memory usage
                frame = cv2.resize(frame, (192, 192))  # Adjust resolution as needed
                
                # Use more efficient encoding parameters
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Add frame to buffer with size limit
                frames_buffer.append(frame)
                if len(frames_buffer) > frame_buffer_size:
                    frames_buffer.pop(0)  # Remove oldest frame
                frame_count += 1
                
                if len(frames_buffer) == previous_frame:
                    _, prev_keypoints = movenet_inference.analyze_frame(frames_buffer[-1])

                # Process analysis based on time interval instead of frame count
                if current_time - last_analysis_time >= analysis_interval and len(frames_buffer) >= frame_buffer_size:
                    try:
                        # Process latest frame only for MoveNet
                        _, curr_keypoints = movenet_inference.analyze_frame(frames_buffer[-1])
                        
                        # Process latest frame for object detection
                        object_inference = object_tracker.process_frames_and_get_annotations(frames_buffer[-1])
                        
                        # Use only recent frames for movement classification
                        predicted_class_name = three_dim_inference.predict_class(frames_buffer)
                        
                        object_predictions = object_inference.get('predictions', [])
                        
                        if not object_predictions:
                            logger.warning(f"Object detection failed. Only contains: {object_predictions}")
                            object_predictions = {}
                        else:
                            # Sort predictions by confidence and get the highest confidence prediction
                            object_predictions = max(object_predictions, key=lambda x: x.get('confidence', 0))
                            logger.info(f"Selected prediction with confidence: {object_predictions.get('confidence')}")

                        # Create features dict more efficiently
                        features = {
                            'joint_angles': extract_joint_angles(curr_keypoints),
                            'speeds': calculate_speed(extract_movement_patterns(curr_keypoints, prev_keypoints)),
                            'body_alignment': extract_body_alignment(curr_keypoints),
                            'stability': calculate_stability(curr_keypoints, prev_keypoints),
                            'object_detections': object_predictions if isinstance(object_predictions, dict) else {}
                        }

                        # 4. Calculate accuracy and get suggestions
                        _, suggestions = calculate_form_accuracy(features, predicted_class_name)
                        
                        # Create Object model with proper field mapping
                        object_base_model = Object(
                            x=object_predictions.get('x', 0),
                            y=object_predictions.get('y', 0),
                            width=object_predictions.get('width', 0),
                            height=object_predictions.get('height', 0),
                            confidence=object_predictions.get('confidence', 0),
                            classs_id=object_predictions.get('class_id', 0),
                            **{'class': object_predictions.get('class', '')}  # Use unpacking for the 'class' field
                        )

                        # Create Features object with empty objects if detection failed
                        features_base_model = Features(
                            objects=object_base_model,
                            joint_angles=features['joint_angles'],
                            movement_pattern=predicted_class_name,
                            speeds=features['speeds'],
                            body_alignment=features['body_alignment'],
                            stability=features['stability']
                        )

                        # Create ExerciseData object
                        exercise_data_base_model = ExerciseData(
                            suggestion=suggestions[0] if suggestions else "No suggestions",
                            features=features_base_model,
                            frame=f"frame_{frame_count}"
                        )
                        
                        # Format the date and time to be Firebase-safe
                        exercise_datetime = exercise_data_base_model.date.split('.')[0]  # Remove microseconds
                        safe_datetime = exercise_datetime.replace(':', '-')
                        exercise_data_dict = exercise_data_base_model.model_dump()
                        
                        # Calculate seconds from frame count (at 30fps)
                        seconds = frame_count // 30
                        
                        # Store under: progress/username/exercise_name/datetime/second_X
                        time_key = f"second_{seconds}"
                        user_id = db.set(
                            path=f'progress/{username}/{exercise_name.lower()}/{safe_datetime}', 
                            data=exercise_data_dict,
                            key=time_key
                        )

                        # Send analysis results
                        await websocket.send_json({
                            'suggestions': suggestions,
                        })

                        last_analysis_time = current_time
                        frames_buffer.clear()  # Clear buffer after analysis
                        
                    except Exception as e:
                        logger.error(f"Error during analysis: {str(e)}")
                        continue

                # Send frame response immediately without waiting for analysis
                await websocket.send_json({
                    'frame': base64.b64encode(buffer).decode('utf-8'),
                    'type': 'frame'
                })

                # Update the buffer with the processed frame
                connection_manager.update_buffer(username, frame)
                connection_manager.increment_frame_count(username)

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
        await connection_manager.disconnect(username)
