import cv2
import os
from inference_sdk import InferenceHTTPClient
from lifttrack.v2.comvis.Movenet import analyze_frame

# Initialize the Roboflow Inference Client
client = InferenceHTTPClient(
    api_url="http://localhost:9001",  # URL for the Roboflow Inference Client
    api_key="TiK2P0kPcpeVnssORWRV",  # Your API Key
)

# Set your Roboflow project details
project_id = "lifttrack"
model_version = 4

# Function to draw bounding boxes on the frame with scaling
def draw_bounding_boxes(frame, predictions, original_size):
    original_height, original_width, _ = original_size.shape

    for pred in predictions:
        x1, y1, x2, y2 = pred['x'], pred['y'], pred['x'] + pred['width'], pred['y'] + pred['height']
        confidence = pred['confidence']
        label = pred['class']

        # Scale the bounding box coordinates
        x1_scaled = int(x1)
        y1_scaled = int(y1)
        x2_scaled = int(x2)
        y2_scaled = int(y2)

        # Draw the scaled bounding box if above confidence threshold
        if confidence > 0.5:
            frame = cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)  # Green box
            frame = cv2.putText(frame, f"{label} {confidence:.2f}", (x1_scaled, y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Function to draw keypoints on the frame
def draw_keypoints(frame, keypoints):
    for point, (x, y, confidence) in keypoints.items():
        if confidence > 0.5:  # Draw only confident keypoints
            frame = cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dots for keypoints
    return frame

# Main function to process frames and retrieve annotations for both keypoints and objects
def process_frames_and_get_annotations(frame):
    try:
        # Run Roboflow inference for object detection
        roboflow_results = client.infer(frame, model_id=f"{project_id}/{model_version}")  # Send the image directly
        
        return roboflow_results
        
    except Exception as e:
        print(f"Error during object detection: {e}")
        return None 