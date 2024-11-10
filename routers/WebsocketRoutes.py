import asyncio
import base64
from fastapi import WebSocket, APIRouter
from fastapi.responses import HTMLResponse
import cv2
import numpy as np

from lifttrack.v2.comvis.Live import resize_to_128x128
from lifttrack.v2.comvis.Movenet import analyze_frame
from lifttrack.v2.comvis.object_track import process_frames_and_get_annotations
from lifttrack.v2.comvis.features import extract_features_from_annotations
from lifttrack.v2.comvis.analyze_features import analyze_annotations
from lifttrack.v2.comvis.progress import frame_by_frame_analysis

router = APIRouter()

@router.websocket("/v2/ws-tracking")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer_size = 30
    message_buffer = []
    features_buffer = []
    class_names = {
        0: "barbell_benchpress",
        1: "barbell_deadlift",
        2: "barbell_rdl",
        3: "barbell_shoulderpress", 
        4: "dumbbell_benchpress",  
        5: "dumbbell_deadlift",
        6: "dumbbell_shoulderpress", 
    }

    try:
        while True:
            # Receive image bytes from the client
            data = await websocket.receive_bytes()
            
            # Decode the image
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_text("Failed to decode image.")
                continue

            # Step 1: Process with Live.py (resize to 128x128)
            resized_128 = resize_to_128x128(frame)
            # Perform inference (assuming a function `infer_live` exists)
            # Replace `infer_live` with the actual inference function from Live.py
            # live_prediction = infer_live(resized_128)

            # Step 2: Process with Movenet.py (resize to 192x192)
            annotated_frame, keypoints = analyze_frame(frame)
            
            # Step 3: Object Tracking using object_track.py
            # Assuming `process_frames_and_get_annotations` processes single frames
            # For single frame, wrap it in a list
            annotations, _ = process_frames_and_get_annotations([annotated_frame], keypoints)
            
            if not annotations:
                await websocket.send_text("No annotations found.")
                continue
            
            # Step 4: Feature Extraction using features.py
            # Assuming `extract_features_from_annotations` processes annotations
            features, _ = extract_features_from_annotations(annotations)
            
            if not features:
                await websocket.send_text("No features extracted.")
                continue
            
            # Step 5: Analyze Features using analyze_features.py
            analyzed_features, final_annotated_frame = analyze_annotations(features)
            
            if not analyzed_features:
                await websocket.send_text("No analyzed features.")
                continue
            
            # Step 6: Progress Handling using progress.py
            # Assuming `frame_by_frame_analysis` processes features and returns suggestions
            # Here, we accumulate the features_buffer
            features_buffer.append(analyzed_features)
            
            # When buffer reaches 30 frames, process and send
            if len(features_buffer) >= buffer_size:
                # Flatten the list if necessary
                all_features = [feature for sublist in features_buffer for feature in sublist]
                
                # Perform progress analysis
                suggestions = frame_by_frame_analysis(all_features, final_annotated_frame, class_names, websocket)
                
                # Clear the buffer
                features_buffer = []
                
                # Send the suggestions back to the client
                await websocket.send_json({
                    'suggestions': suggestions
                })

    except Exception as e:
        await websocket.close()
        print(f"WebSocket connection closed: {e}")