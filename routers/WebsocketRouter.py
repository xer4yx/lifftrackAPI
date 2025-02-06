import asyncio
import base64
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, Query, Depends, status
from fastapi.websockets import WebSocketState, WebSocketDisconnect

from lifttrack.utils.logging_config import setup_logger
from lifttrack.comvis import websocket_process_frames
from lifttrack.v2.comvis.inference_handler import *
from lifttrack.v2.dbhelper import get_db
from lifttrack.v2.dbhelper.admin_rtdb import FirebaseDBHelper
from lifttrack.dbhandler.rest_rtdb import rtdb
import ws_endpoint

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
    await websocket.accept()
    connection_active = True
        
    logger.info(f"Starting tracking session for user: {username}, exercise: {exercise_name}")

    frame_count = 0
    last_frame_time = asyncio.get_event_loop().time()
    frames_buffer = []
    last_analysis_time = asyncio.get_event_loop().time()
    
    try:
        while frame_count < ws_endpoint.MAX_FRAMES and connection_active:
            try:
                # Check if connection is still active before processing
                if not websocket.client:  # Check if client is None, which means disconnected
                    logger.info("WebSocket disconnected, stopping frame processing")
                    connection_active = False
                    break

                # Check for timeout
                current_time = asyncio.get_event_loop().time()
                if current_time - last_frame_time > ws_endpoint.FRAME_TIMEOUT:
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
                        timeout=ws_endpoint.FRAME_TIMEOUT
                    )
                    last_frame_time = asyncio.get_event_loop().time()
                except asyncio.TimeoutError:
                    logger.info("Frame reception timeout")
                    frames_buffer.clear()  # Clear buffer on timeout
                    break

                # Process the received frame data more efficiently
                frame = convert_byte_to_numpy(data)
                if frame is None or frame.size == 0:
                    logger.warning("Failed to decode frame")
                    continue

                frame, buffer =  process_frame(frame)
                
                # Add frame to buffer with size limit
                frames_buffer.append(frame)
                if len(frames_buffer) > ws_endpoint.FRAME_BUFFER_SIZE:
                    frames_buffer.pop(0)  # Remove oldest frame
                frame_count += 1
                
                # Process analysis based on time interval instead of frame count
                if current_time - last_analysis_time >= ws_endpoint.ANALYSIS_INTERVAL and len(frames_buffer) >= ws_endpoint.FRAME_BUFFER_SIZE:
                    try:
                        curr_keypoints, prev_keypoints, object_inference, predicted_class_name = perform_frame_analysis(frames_buffer)

                        object_predictions = load_to_object_model(object_inference)
                        features = load_to_features_model(prev_keypoints, curr_keypoints, object_predictions)

                        suggestions = get_suggestions(features, predicted_class_name)

                        exercise_data_base_model = load_to_exercise_data_model(features, suggestions, f"frame_{frame_count}")
                    
                        safe_datetime = format_date(exercise_data_base_model.date)
                        exercise_data_dict = exercise_data_base_model.model_dump()
                        
                        seconds = frame_count // 30 # Calculate seconds from frame count (at 30fps)
                        save_progress(
                            username=username, 
                            exercise_name=exercise_name, 
                            date=safe_datetime, 
                            time_frame=f"second_{seconds}", 
                            exercise_data=exercise_data_dict, 
                            db=db)

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
                    'reason': 'max_frames_reached' if frame_count >= ws_endpoint.MAX_FRAMES else 'timeout'
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
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.send_json({
                'type': 'error',
                'message': str(e),
                'frame_count': frame_count
            })
            await websocket.close()
    finally:
        frames_buffer.clear()
        logger.info("WebSocket connection closed and buffer cleared")
