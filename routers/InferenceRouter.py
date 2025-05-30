import cv2
import numpy as np
import base64
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, Request
from fastapi.responses import StreamingResponse
from typing import List
from pydantic import BaseModel
from datetime import datetime

from lifttrack.v2.dbhelper import get_db
from lifttrack.v2.dbhelper.admin_rtdb import FirebaseDBHelper
from lifttrack.utils.logging_config import setup_logger
from lifttrack.v2.comvis.features import (
    extract_joint_angles, 
    extract_movement_patterns, 
    calculate_speed, 
    extract_body_alignment, 
    calculate_stability
)
from lifttrack.v2.comvis.progress import calculate_form_accuracy
from lifttrack.models import Exercise, ExerciseData, Features, Object

from core.interface import InferenceInterface
from infrastructure.database import FirebaseAdmin
from infrastructure.di import get_firebase_admin

router = APIRouter()

class FrameData(BaseModel):
    frame: str  # base64 encoded frame
    username: str
    exercise_name: str

class InferenceResponse(BaseModel):
    keypoints: dict
    object_detection: dict
    movement_class: str
    frame_processed: bool
    
logger = setup_logger("inference", "inference.log")

@router.post("/inference", response_model=InferenceResponse, deprecated=True)
async def inference_endpoint(
    request: Request,
    video_file: UploadFile,  # Expecting a video file
    username: str = Query(...),
    exercise_name: str = Query(...),
    db: FirebaseAdmin = Depends(get_firebase_admin)
):
    try:
        if not username or not exercise_name:
            raise HTTPException(status_code=400, detail="Both username and exercise_name are required")
        
        logger.info(f"Processing video for user: {username}, exercise: {exercise_name}")

        # Read the video file
        video_data = await video_file.read()
        video_array = np.frombuffer(video_data, np.uint8)
        video = cv2.imdecode(video_array, cv2.IMREAD_COLOR)

        if video is None:
            raise HTTPException(status_code=400, detail="Failed to decode video")

        # Initialize variables for processing
        frame_count = 0
        processed_frames = []
        current_time = datetime.now()
        timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")

        # Process each frame in the video
        while True:
            # Get the inference services from the request
            inference_services = request.app.state.inference_services
            movenet_inference: InferenceInterface = inference_services.get('posenet')
            object_tracker: InferenceInterface = inference_services.get('object_tracker')
            three_dim_inference: InferenceInterface = inference_services.get('videoaction')
            
            ret, frame = video.read()
            if not ret:
                break  # Exit if no more frames

            # Resize frame
            frame = cv2.resize(frame, (192, 192))

            # Only process every 30th frame (1 second at 30 fps)
            if frame_count % 30 == 0:
                # Process frame with MoveNet
                _, curr_keypoints = await movenet_inference.infer_async(frame)

                # Process frame with object detection
                object_inference = await object_tracker.infer_async(frame)
                if not object_inference:
                    object_inference = {
                        "class_id": -1,
                        "type": "unknown",
                        "confidence": 0.0
                    }

                # Process movement classification
                frames_buffer = [frame]  # For single frame analysis
                predicted_class_name = await three_dim_inference.infer_async(frames_buffer)

                # Create features dict
                features = {
                    'joint_angles': extract_joint_angles(curr_keypoints),
                    'speeds': calculate_speed(extract_movement_patterns(curr_keypoints, curr_keypoints)),
                    'body_alignment': extract_body_alignment(curr_keypoints),
                    'stability': calculate_stability(curr_keypoints, curr_keypoints),
                    'object_detections': object_inference if isinstance(object_inference, dict) else {}
                }

                # Calculate accuracy and get suggestions
                _, suggestions = calculate_form_accuracy(features, predicted_class_name)

                # Calculate the current second
                second = frame_count // 30  # Every 30 frames corresponds to 1 second

                # Create ExerciseData object
                exercise_data_base_model = ExerciseData(
                    suggestion=suggestions[0] if suggestions else "No suggestions",
                    features=features,
                    frame=f"frame_{frame_count}"  # Use frame number for naming
                )

                # Store under: progress/username/exercise_name/datetime/second_X
                exercise_datetime = exercise_data_base_model.date.split('.')[0]  # Remove microseconds
                safe_datetime = exercise_datetime.replace(':', '-')
                exercise_data_dict = exercise_data_base_model.model_dump()
                time_key = f"second_{second}"  # Format time_key as second_{second}
                user_id = await db.set_data(
                    key=f'progress/{username}/{exercise_name.lower()}/{safe_datetime}/{time_key}', 
                    value=exercise_data_dict
                )

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                processed_frames.append(base64.b64encode(buffer).decode('utf-8'))  # Store processed frame

            frame_count += 1

        return {
            'processed_frames': processed_frames,
            'suggestions': suggestions,
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))