import asyncio
import base64
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends

from lifttrack.utils.logging_config import setup_logger
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
from lifttrack.v2.dbhelper import get_db
from lifttrack.v2.dbhelper.admin_rtdb import FirebaseDBHelper

from lifttrack.models import Exercise, ExerciseData, Features, Object
from lifttrack.dbhandler.rest_rtdb import rtdb

router = APIRouter()
logger = setup_logger("websocket", "protocols.log")


# API Endpoint [Frame Operations]
@router.websocket("/ws-tracking")  # Mobile version
async def websocket_inference(websocket: WebSocket):
    """
    Endpoint for the WebSocket video feed. The server receives frames from the client that's formatted as a base64
    bytes. The server decodes the frame in a thread pool, processes the frame to get inference and annotations, and
    encodes it back to the client as bytes back.

    Args:
        websocket: WebSocket object.
    """
    await websocket.accept()
    connection_open = True
    model = None
    while connection_open:
        try:
            # Expected format from client side is:
            # {"type": "websocket.receive", "text": data}
            frame_data = await websocket.receive()

            if frame_data["type"] == "websocket.close":
                connection_open = False
                break

            # frame_byte = base64.b64decode(frame_data["bytes"])
            frame_byte = frame_data.get("bytes")

            if not isinstance(frame_byte, bytes):
                await websocket.close()
                break

            # Process frame in thread pool to avoid blocking
            (annotated_frame, features) = await asyncio.get_event_loop().run_in_executor(
                None, websocket_process_frames, model, frame_byte
            )

            # Encode and send result
            encoded, buffer = cv2.imencode(".jpeg", annotated_frame)

            if not encoded:
                raise WebSocketDisconnect

            return_bytes = base64.b64decode(buffer.tobytes())

            # Expected return to the client side is:
            # {"type": "websocket.send", "bytes": data}
            await websocket.send_bytes(return_bytes)
        except WebSocketDisconnect:
            connection_open = False
        except Exception as e:
            logger.exception(f"WebSocket error: {e}")
            connection_open = False
        finally:
            if not connection_open:
                await websocket.close()


@router.websocket("/v2/ws-tracking")
async def websocket_endpoint(
    websocket: WebSocket,
    username: str = Query(..., description="Username of the person exercising"),
    exercise_name: str = Query(..., description="Name of the exercise being performed"),
    db: FirebaseDBHelper = Depends(get_db)
    ):
    connection_active = False
    try:
        await websocket.accept()
        connection_active = True
        
        frames_buffer = []  # Initialize buffer
        
        if not username or not exercise_name:
            raise ValueError("Both username and exercise_name are required")
            
        logger.info(f"Starting tracking session for user: {username}, exercise: {exercise_name}")
    except Exception as e:
        logger.error(f"Failed to get initialization data: {str(e)}")
        frames_buffer.clear()
        if connection_active:
            await websocket.close()
        return

    frame_count = 0
    max_frames = 1800
    last_frame_time = asyncio.get_event_loop().time()
    frame_timeout = 60.0
    
    previous_frame = 29
    frame_buffer_size = 30  # Only keep last 30 frames for analysis
    frames_buffer = []
    last_analysis_time = asyncio.get_event_loop().time()
    analysis_interval = 1.0  # Perform analysis every 1 second instead of every 30 frames
    
    try:
        while frame_count < max_frames and connection_active:
            try:
                # Check if connection is still active before processing
                if not websocket.client:  # Check if client is None, which means disconnected
                    logger.info("WebSocket disconnected, stopping frame processing")
                    connection_active = False
                    break

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
                    logger.warning("Failed to decode frame")
                    continue

                # Resize frame before processing to reduce memory usage
                frame = cv2.resize(frame, (192, 192))  # Adjust resolution as needed
                
                # Encode frame to JPEG before sending back
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Add frame to buffer with size limit
                frames_buffer.append(frame)
                if len(frames_buffer) > frame_buffer_size:
                    frames_buffer.pop(0)  # Remove oldest frame
                frame_count += 1
                
                # Process analysis based on time interval instead of frame count
                if current_time - last_analysis_time >= analysis_interval and len(frames_buffer) >= frame_buffer_size:
                    try:
                        # Process latest frame for MoveNet
                        _, curr_keypoints = movenet_inference.analyze_frame(frames_buffer[-1])
                        
                        if frame_count > 1:  # Only get previous keypoints if we have more than one frame
                            _, prev_keypoints = movenet_inference.analyze_frame(frames_buffer[-2])
                        else:
                            prev_keypoints = curr_keypoints  # Use current keypoints as previous for first frame
                        
                        # Process latest frame for object detection
                        object_inference = object_tracker.process_frames_and_get_annotations(frames_buffer[-1])
                        
                        # Use only recent frames for movement classification
                        predicted_class_name = three_dim_inference.predict_class(frames_buffer)
                        
                        
                        if not object_inference:
                            logger.warning("Object detection failed. Using default unknown object")
                            object_predictions = Object(
                                classs_id=-1,
                                type="unknown",
                                confidence=0.0,
                                x=0.0,
                                y=0.0,
                                width=0.0,
                                height=0.0
                            )
                        else:
                            best_pred = max(object_inference, 
                                         key=lambda x: x.get('confidence', 0))
                            
                            object_predictions = Object(
                                classs_id=best_pred.get('class_id', -1),
                                type=best_pred.get('class', 'unknown'),
                                confidence=best_pred.get('confidence', 0.0),
                                x=best_pred.get('x', 0.0),
                                y=best_pred.get('y', 0.0),
                                width=best_pred.get('width', 0.0),
                                height=best_pred.get('height', 0.0)
                            )

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
                        object_base_model = object_predictions.copy()

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
                        
                        # Create Exercise object and set the specific exercise data
                        # exercise = Exercise()
                        # exercise.set_exercise_data(exercise_name, exercise_data)

                        # Store in database
                        # rtdb.put_progress(
                        #     username=username,
                        #     exercise=exercise
                        # )
                        
                        # Format the date and time to be Firebase-safe
                        exercise_datetime = exercise_data_base_model.date.split('.')[0]  # Remove microseconds
                        safe_datetime = exercise_datetime.replace(':', '-')
                        exercise_data_dict = exercise_data_base_model.model_dump()
                        
                        # Calculate seconds from frame count (at 30fps)
                        seconds = frame_count // 30
                        
                        # Store under: progress/username/exercise_name/datetime/second_X
                        time_key = f"second_{seconds}"
                        user_id = db.set_data(
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

                # Only send frame if connection is still active
                if connection_active and not websocket.client_state.DISCONNECTED:
                    try:
                        await websocket.send_json({
                            'frame': base64.b64encode(buffer).decode('utf-8'),
                            'type': 'frame'
                        })
                    except RuntimeError as e:
                        logger.warning(f"Connection closed during send: {str(e)}")
                        connection_active = False
                        break
                    except WebSocketDisconnect:
                        logger.warning("WebSocket disconnected while sending frame")
                        connection_active = False
                        break

            except WebSocketDisconnect:
                logger.info(f"Client disconnected after processing {frame_count} frames")
                connection_active = False
                frames_buffer.clear()
                break
            except Exception as e:
                logger.error(f"Error in frame processing: {str(e)}")
                frames_buffer.clear()
                if not isinstance(e, WebSocketDisconnect):  # Only continue if it's not a disconnect
                    continue
                connection_active = False
                break

        # Session completion - only try to send if connection is still active
        if connection_active and not websocket.client_state.DISCONNECTED:
            frames_buffer.clear()
            try:
                await websocket.send_json({
                    'type': 'complete',
                    'message': 'Session ended successfully',
                    'frame_count': frame_count,
                    'reason': 'max_frames_reached' if frame_count >= max_frames else 'timeout'
                })
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect) as e:
                logger.warning(f"Connection already closed during completion: {str(e)}")

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
