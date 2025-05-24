import asyncio
import json
import concurrent.futures
import urllib
from functools import partial
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, status
from fastapi.encoders import jsonable_encoder
from lifttrack.auth import validate_token
from lifttrack.v2.comvis.inference_handler import *
from lifttrack.v2.dbhelper import get_db
from lifttrack.v2.dbhelper.admin_rtdb import FirebaseDBHelper
from ..manager import ConnectionManager

router = APIRouter()
manager = ConnectionManager()

# Create a thread pool executor for CPU-bound tasks
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Create more efficient frame processing functions
def process_frame_batch(frames):
    """Process multiple frames in parallel"""
    processed_frames = []
    for frame in frames:
        processed_frame, _ = process_frame(frame)
        processed_frames.append(processed_frame)
    return processed_frames

async def process_frame_async(frame):
    """Asynchronous wrapper for frame processing"""
    loop = asyncio.get_running_loop()
    processed_frame, buffer = await loop.run_in_executor(thread_pool, process_frame, frame)
    return processed_frame

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
    
    await websocket.accept()
    logger.info(f"{username} state: {websocket.client_state.name}")
    
    # Initialize variables
    frame_count = 0
    max_frames = 1800
    frame_buffer_size = 30
    frames_buffer = []
    pending_analyses = set()
    connection_active = True
    
    # Track which seconds we've already saved
    saved_seconds = set()
    
    # Pre-process exercise name once
    exercise_name = urllib.parse.unquote(string=exercise_name).lower().replace(" ", "_")
    
    # Precompile frame parsing functions
    async def parse_frame(byte_data):
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
            
            # Extract the YUV planes
            data_start = 4 + header_length
            y_plane = byte_data[data_start:data_start+y_size]
            u_plane = byte_data[data_start+y_size:data_start+y_size+u_size]
            v_plane = byte_data[data_start+y_size+u_size:data_start+y_size+u_size+v_size]
            
            # Create a custom YUV420 frame
            frame = create_yuv420_frame(y_plane, u_plane, v_plane, width, height)
            
            if frame is None or frame.size == 0:
                return None
                
            return frame
        except Exception as e:
            logger.error(f"Error parsing frame: {str(e)}")
            return None

    async def analyze_buffer(frames_buffer, current_frame_count, second_number, websocket: WebSocket):
        """Analyze the buffer of frames and send results"""
        loop = asyncio.get_running_loop()
        
        try:
            # Run analysis in thread pool
            current_keypoints, previous_keypoints, object_inference, predicted_class_name = await loop.run_in_executor(
                thread_pool, perform_frame_analysis, frames_buffer.copy(), websocket)
            
            # Process results
            object_predictions = load_to_object_model(object_inference)
            features = load_to_features_model(previous_keypoints, current_keypoints, object_predictions, predicted_class_name)
            accuracy, suggestions = get_suggestions(features, exercise_name)
    
            exercise_data_base_model = load_to_exercise_data_model(
                features, 
                suggestions, 
                f"frame_{current_frame_count}")
            
            # Use exact second number for the time frame key
            time_frame = f"second_{second_number}"
            
            # Database operations - continue even if websocket is closed
            await loop.run_in_executor(
                thread_pool,
                partial(
                    save_progress,
                    username=username,
                    exercise_name=exercise_name.replace("_", " "),
                    date=format_date(exercise_data_base_model.date),
                    time_frame=time_frame,
                    exercise_data=jsonable_encoder(exercise_data_base_model),
                    db=db
                )
            )
            logger.info(f"Saved analysis for {time_frame} at frame {current_frame_count}")
            
            # Only send message to client if connection is still active
            if connection_active and websocket.client_state.name == "CONNECTED":
                try:
                    await websocket.send_json({'suggestions': suggestions, 'accuracy': str(accuracy)})
                except RuntimeError as e:
                    logger.warning(f"Could not send analysis results: {str(e)}")
                    # Don't raise the exception, just log it
            
            return suggestions, accuracy
        except Exception as e:
            logger.error(f"Analysis task failed: {str(e)}")
            return None, None
    
    try:
        while connection_active and frame_count < max_frames:
            # Receive frame data
            byte_data = await websocket.receive_bytes()
            
            # Check for completion signal
            if byte_data == b'COMPLETED':
                logger.info(f"Received completion signal from client")
                await websocket.send_json({'status': 'COMPLETED_ACK'})
                continue  # Skip the rest of the loop
            
            # Parse frame (non-blocking)
            frame = await parse_frame(byte_data)
            if frame is not None:
                # Process frame asynchronously
                processed_frame = await process_frame_async(frame)
                
                # Add to buffer
                frames_buffer.append(processed_frame)
                if len(frames_buffer) > frame_buffer_size:
                    frames_buffer.pop(0)
                
                # Increment frame count
                frame_count += 1
                
                # Calculate the current second based on frame count
                current_second = frame_count // 30
                
                # Check if we need to save this second and haven't saved it yet
                if frame_count % 30 == 0 and current_second not in saved_seconds:
                    # Mark this second as saved
                    saved_seconds.add(current_second)
                    
                    # Schedule analysis and don't wait for it
                    analysis_task = asyncio.create_task(
                        analyze_buffer(frames_buffer.copy(), frame_count, current_second, websocket)
                    )
                    
                    # Properly track the task
                    pending_analyses.add(analysis_task)
                    
                    # Use a proper callback that captures the task
                    def task_done_callback(completed_task):
                        pending_analyses.discard(completed_task)
                        if completed_task.exception():
                            logger.error(f"Analysis task failed: {completed_task.exception()}")
                    
                    analysis_task.add_done_callback(task_done_callback)
                
                # Return feedback to client about frame processing
                try:
                    await websocket.send_json({
                        'status': 'processing',
                        'frame_count': frame_count,
                        'current_second': current_second
                    })
                except RuntimeError as e:
                    logger.warning(f"Could not send processing status: {str(e)}")
                    connection_active = False
                    break
            
            # Short yield to allow other tasks to run
            await asyncio.sleep(0.001)
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected after processing {frame_count} frames")
        logger.info(f"Client state: {websocket.client_state.name}")
        connection_active = False
    except Exception as e:
        logger.error(f"Error in websocket handler: {str(e)}")
        connection_active = False
    finally:
        # Mark connection as inactive to prevent new messages
        connection_active = False
        
        # Wait for pending analyses to complete but don't send messages
        if pending_analyses:
            try:
                # Set a timeout to prevent hanging forever
                done, pending = await asyncio.wait(pending_analyses, timeout=3.0)
                
                # Log any incomplete tasks
                if pending:
                    logger.warning(f"{len(pending)} analysis tasks did not complete in time")
            except Exception as e:
                logger.error(f"Error waiting for pending analyses: {str(e)}")
        
        frames_buffer.clear()