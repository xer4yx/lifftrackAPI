import asyncio
import base64
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from lifttrack.utils.logging_config import setup_logger
from lifttrack.comvis import websocket_process_frames
from lifttrack.v2.comvis.Live import model, predict_class
from lifttrack.v2.comvis.Movenet import analyze_frame  # This handles 192x192 internally
from lifttrack.v2.comvis.object_track import process_frames_and_get_annotations
from lifttrack.v2.comvis.features import extract_joint_angles, extract_movement_patterns, calculate_speed, extract_body_alignment, calculate_stability
from lifttrack.v2.comvis.progress import calculate_form_accuracy

from lifttrack.models import Exercise, ExerciseData, Features
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

                # Process analysis every 30th frame
                if frame_count % 30 == 0:
                    try:
                        logger.info(f"Frame count: {frame_count}")
                        # 1. Get MoveNet and Roboflow inference
                        _, keypoints = analyze_frame(frame)
                        roboflow_results = process_frames_and_get_annotations(frame)
                        
                        # if not roboflow_results:
                        #     predictions = []
                        # else:
                        #     predictions = roboflow_results.get('predictions', [])
                        
                        predictions = {}
                        if roboflow_results:
                            predictions = {prediction for prediction in roboflow_results}

                        # 2. Get predicted class
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

                        # Create Features object
                        features_obj = Features(
                            objects=predictions,
                            joint_angles=features['joint_angles'],
                            movement_pattern=predicted_class_name,
                            speeds=features['speeds'],
                            body_alignment=features['body_alignment'],
                            stability=features['stability']
                        )

                        # Create ExerciseData object
                        exercise_data = ExerciseData(
                            suggestion=suggestions[0] if suggestions else "No suggestions",
                            features=features_obj,
                            frame=f"frame_{frame_count}"
                        )
                        
                        # Create Exercise object and set the specific exercise data
                        exercise = Exercise()
                        exercise.set_exercise_data(exercise_name, exercise_data)

                        # Store in database
                        rtdb.put_progress(
                            username=username,
                            exercise=exercise
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