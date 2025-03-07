import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, status
from fastapi.encoders import jsonable_encoder
from lifttrack.auth import validate_token
from lifttrack.v2.comvis.inference_handler import *
from lifttrack.v2.dbhelper import get_db
from lifttrack.v2.dbhelper.admin_rtdb import FirebaseDBHelper
from ..manager import ConnectionManager

router = APIRouter()
manager = ConnectionManager()

@router.websocket("/exercise-tracking")
async def track_exercise(
    websocket: WebSocket,
    username: str = Query(...),
    exercise_name: str = Query(...),
    token: str = Query(...),
    db: FirebaseDBHelper = Depends(get_db)
    ):
    if not validate_token(token, username):
        logger.error(f"Invalid token for user: {username}")
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Token is invalid or has expired."
        )
        return
    
    frame_count = 0
    max_frames = 1800
    analysis_interval = 1.0
    frame_buffer_size = 30
    frames_buffer = []
    last_analysis_time = asyncio.get_event_loop().time()
    
    await websocket.accept()
    logger.info(f"{username} state: {websocket.client_state.name}")
    connection_alive = True
    try:
        while connection_alive:
            byte_data = await websocket.receive_bytes()

            try:
                # Parse the header
                header_length = int.from_bytes(byte_data[0:4], byteorder='big')
                header_bytes = byte_data[4:4+header_length]
                header_json = header_bytes.decode('utf-8')
                header = json.loads(header_json)
                
                # Extract image dimensions and format
                width = header['width']
                height = header['height']
                format = header['format']
                y_size = header['ySize']
                u_size = header['uSize']
                v_size = header['vSize']
                
                # Log the image information
                logger.info(f"Image format: {format} frame: {width}x{height}, Y={y_size}, U={u_size}, V={v_size}")
                
                # Extract the YUV planes
                data_start = 4 + header_length
                y_plane = byte_data[data_start:data_start+y_size]
                u_plane = byte_data[data_start+y_size:data_start+y_size+u_size]
                v_plane = byte_data[data_start+y_size+u_size:data_start+y_size+u_size+v_size]
                
                # Create a custom YUV420 frame
                frame = create_yuv420_frame(y_plane, u_plane, v_plane, width, height)
                
                if frame is None or frame.size == 0:
                    logger.warning(f"Failed to create frame from {username}")
                    continue
                
                # Log before processing
                logger.info(f"Processing frame {frame_count}")
                
                frame, buffer = process_frame(frame)
                
                # Log after processing
                logger.info(f"Processed frame shape: {frame.shape}")
                
                frames_buffer.append(frame)
                if len(frames_buffer) > frame_buffer_size:
                    frames_buffer.pop(0)
                frame_count += 1
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
            
            current_time = asyncio.get_event_loop().time()
            if current_time - last_analysis_time >= analysis_interval and len(frames_buffer) == frame_buffer_size:
                current_keypoints, previous_keypoints, object_inference, predicted_class_name = perform_frame_analysis(frames_buffer)
                object_predictions = load_to_object_model(object_inference)
                features = load_to_features_model(previous_keypoints, current_keypoints, object_predictions, predicted_class_name)

                suggestions = get_suggestions(features, predicted_class_name)

                exercise_data_base_model = load_to_exercise_data_model(
                    features, 
                    suggestions, 
                    f"frame_{frame_count}")
                
                save_progress(
                    username=username,
                    exercise_name=exercise_name,
                    date=format_date(exercise_data_base_model.date),
                    time_frame=f"second_{frame_count}",
                    exercise_data=jsonable_encoder(exercise_data_base_model),
                    db=db)
                
                last_analysis_time = current_time
                await asyncio.sleep(0.01)
                
                await manager.send_json_message(
                    data={'suggestions': suggestions})
    except WebSocketDisconnect:
        logger.info(f"Client disconnected after processing {frame_count} frames")
        logger.info(f"Client state: {websocket.client_state.name}")
        connection_alive = False
        frames_buffer.clear()