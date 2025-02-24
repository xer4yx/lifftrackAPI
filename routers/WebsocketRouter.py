import asyncio
import base64
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.encoders import jsonable_encoder

from lifttrack.utils.logging_config import setup_logger
from lifttrack.comvis import websocket_process_frames
from lifttrack.v2.comvis.inference_handler import *
from lifttrack.v2.dbhelper import get_db
from lifttrack.v2.dbhelper.admin_rtdb import FirebaseDBHelper

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
    frame_count = 0
    max_frames = 1800
    frame_buffer_size = 30
    frames_buffer = []
    last_analysis_time = asyncio.get_event_loop().time()
    analysis_interval = 1.0
    
    try:
        while frame_count <= max_frames and connection_active:
            data = await websocket.receive_bytes()

            # Check for completion signal
            if data == b'COMPLETED':
                logger.info(f"Received completion signal from client")
                await websocket.send_json({'status': 'COMPLETED_ACK'})

            frame = convert_byte_to_numpy(data)
            if frame is None or frame.size == 0:
                logger.warning("Failed to decode frame")
                continue

            frame, buffer =  process_frame(frame)
            frames_buffer.append(frame)
            if len(frames_buffer) > frame_buffer_size:
                frames_buffer.pop(0)
            frame_count += 1
            
            current_time = asyncio.get_event_loop().time()
            if current_time - last_analysis_time >= analysis_interval and len(frames_buffer) >= frame_buffer_size:
                try:
                    curr_keypoints, prev_keypoints, \
                    object_inference, predicted_class_name = perform_frame_analysis(frames_buffer=frames_buffer)

                    object_predictions = load_to_object_model(object_inference)
                    features = load_to_features_model(
                        previous_pose=prev_keypoints, 
                        current_pose=curr_keypoints, 
                        object_inference=object_predictions, 
                        class_name=predicted_class_name)

                    suggestions = get_suggestions(
                        features=features, 
                        class_name=predicted_class_name)

                    exercise_data_base_model = load_to_exercise_data_model(
                        features=features, 
                        suggestions=suggestions, 
                        frame_index=f"frame_{frame_count}")
                
                    safe_datetime = format_date(date=exercise_data_base_model.date)
                    
                    save_progress(
                        username=username, 
                        exercise_name=exercise_name, 
                        date=safe_datetime, 
                        time_frame=f"second_{frame_count // 30}", 
                        exercise_data=jsonable_encoder(exercise_data_base_model),
                        db=db)

                    # Send analysis results
                    await websocket.send_json({'suggestions': suggestions})

                    last_analysis_time = current_time
                    frames_buffer.clear()  # Clear buffer after analysis
                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    continue

            if connection_active and not websocket.client_state.DISCONNECTED:
                    await websocket.send_json({
                        'frame': base64.b64encode(buffer).decode('utf-8'),
                        'type': 'frame'
                    })
    except WebSocketDisconnect:
        logger.info(f"Client status: {websocket.client_state.name}")
        logger.info(f"Client disconnected after processing {frame_count} frames")
        frames_buffer.clear()
        
