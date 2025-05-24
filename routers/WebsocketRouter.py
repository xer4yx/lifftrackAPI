import asyncio
import urllib.parse
import base64
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, status
from fastapi.encoders import jsonable_encoder
from datetime import datetime

from lifttrack.utils.logging_config import setup_logger
from lifttrack.v2.comvis.inference_handler import *
from lifttrack.v2.dbhelper import get_db
from lifttrack.v2.dbhelper.admin_rtdb import FirebaseDBHelper

router = APIRouter()
logger = setup_logger("websocket", "protocols.log")

# Create a thread pool executor for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=4)



# Helper function to perform analysis in a separate thread
def perform_analysis(frames_buffer, username, exercise_name, db_exercise_name, frame_count, db, websocket: WebSocket):
    try:
        curr_keypoints, prev_keypoints, object_inference, predicted_class_name = perform_frame_analysis(
            frames_buffer=frames_buffer,
            shared_resource=websocket
        )

        object_predictions = load_to_object_model(object_inference)
        features = load_to_features_model(
            previous_pose=prev_keypoints,
            current_pose=curr_keypoints,
            object_inference=object_predictions,
            class_name=predicted_class_name
        )

        accuracy, suggestions = get_suggestions(
            features=features,
            class_name=exercise_name
        )

        exercise_data_base_model = load_to_exercise_data_model(
            features=features,
            suggestions=suggestions,
            frame_index=f"frame_{frame_count}"
        )

        safe_datetime = format_date(date=exercise_data_base_model.date)

        # Save to database (not batched as requested)
        save_progress(
            username=username,
            exercise_name=db_exercise_name,
            date=safe_datetime,
            time_frame=f"second_{frame_count // 30}",
            exercise_data=jsonable_encoder(exercise_data_base_model),
            db=db
        )
        
        return suggestions, accuracy
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return None
    
    
# Helper function to run analysis in thread pool and handle correctly as a coroutine
async def run_analysis_in_thread(frames_buffer, username, exercise_name, db_exercise_name, frame_count, db, websocket: WebSocket):
    try:
        # Run the analysis in a thread pool
        return await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            perform_analysis,
            frames_buffer,
            username,
            exercise_name,
            db_exercise_name,
            frame_count,
            db,
            websocket
        )
    except Exception as e:
        logger.error(f"Analysis thread error: {str(e)}")
        return None

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
    last_analysis_time = time.time()
    analysis_interval = 1.0
    
    # Pre-process exercise name once
    exercise_name = urllib.parse.unquote(string=exercise_name).lower().replace(" ", "_")
    valid_exercise_names = set(["bench_press", "benchpress", "deadlift", "romanian_deadlift", "rdl", "shoulder_press", "overhead_press"])
    if exercise_name not in valid_exercise_names:
        logger.warning(f"Unsupported exercise type: {exercise_name}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=f"Unsupported exercise type: {exercise_name}")
        return
    
    # Pre-calculate formatted exercise name for database
    db_exercise_name = exercise_name.replace("_", " ")
    
    try:
        while frame_count <= max_frames and connection_active:
            data = await websocket.receive_bytes()

            # Check for completion signal
            if data == b'COMPLETED':
                logger.info(f"Received completion signal from client")
                await websocket.send_json({'status': 'COMPLETED_ACK'})
                continue  # Skip the rest of the loop

            # Process frame in thread pool to avoid blocking the event loop
            frame = await asyncio.get_event_loop().run_in_executor(
                thread_pool, convert_byte_to_numpy, data
            )
            
            if frame is None or frame.size == 0:
                logger.warning("Failed to decode frame")
                continue

            # Process frame in thread pool
            frame_and_buffer = await asyncio.get_event_loop().run_in_executor(
                thread_pool, process_frame, frame
            )
            
            if frame_and_buffer:
                frame, buffer = frame_and_buffer
                frames_buffer.append(frame)
                if len(frames_buffer) > frame_buffer_size:
                    frames_buffer.pop(0)
                frame_count += 1
            else:
                continue
            
            current_time = time.time()
            # Only analyze when we have enough frames and enough time has passed
            if current_time - last_analysis_time >= analysis_interval and len(frames_buffer) >= frame_buffer_size:
                try:
                    # Make a copy of frames buffer to avoid race conditions
                    frames_to_analyze = frames_buffer.copy()
                    
                    # Run the analysis in a thread pool and get results
                    accuracy,suggestions = await run_analysis_in_thread(
                        frames_to_analyze,
                        username,
                        exercise_name,
                        db_exercise_name,
                        frame_count,
                        db,
                        websocket
                    )
                    
                    if suggestions:
                        # Send analysis results
                        await websocket.send_json({'suggestions': suggestions, 'accuracy': str(accuracy)})
                    
                    last_analysis_time = current_time
                    frames_buffer.clear()  # Clear buffer after analysis
                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    frames_buffer.clear()

            # Always send the current frame back to client
            if connection_active and not websocket.client_state.DISCONNECTED:
                # Generate frame_id in the format: {exercise name}_frame_{frame number}_{date}
                safe_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                frame_id = f"{exercise_name}_frame_{frame_count}_{safe_datetime}"
                
                # Encode the frame and send it with the frame_id
                encoded_frame = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_json({
                    'type': 'frame',
                    'frame': encoded_frame,
                    'frame_id': frame_id,
                })
    except WebSocketDisconnect:
        logger.info(f"Client status: {websocket.client_state.name}")
        logger.info(f"Client disconnected after processing {frame_count} frames")
        frames_buffer.clear()
