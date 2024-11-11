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
    original_height, original_width, _ = original_size

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
def process_frames_and_get_annotations(frames_directory, analyze_frame):
    annotations = []  # List to store the combined annotations
    annotated_frame = None  # Initialize annotated_frame to avoid reference before assignment

    # Walk through the frames directory
    for root, dirs, files in os.walk(frames_directory):
        for file in sorted(files):
            if file.endswith('.jpg'):
                frame_path = os.path.join(root, file)
                print(f"Processing frame: {file}")
                frame = cv2.imread(frame_path)

                if frame is None:
                    print(f"Error reading frame: {frame_path}")
                    continue

                # Analyze the frame to get annotated_frame and keypoints (from MoveNet)
                try:
                    annotated_frame, keypoints = analyze_frame(frame)
                except Exception as e:
                    print(f"Error analyzing frame {frame_path}: {e}")
                    continue

                # Draw keypoints on the frame
                annotated_frame = draw_keypoints(annotated_frame, keypoints)

                # Run Roboflow inference for object detection
                try:
                    roboflow_results = client.infer(frame_path, model_id=f"{project_id}/{model_version}")
                except Exception as e:
                    print(f"Error during RoboFlow inference for {frame_path}: {e}")
                    continue

                # Draw object annotations on the frame with bounding boxes
                annotated_frame = draw_bounding_boxes(annotated_frame, roboflow_results['predictions'], frame.shape)

                # Combine keypoints and object predictions
                frame_info = {
                    'frame_path': frame_path,
                    'objects': roboflow_results['predictions'],  # Object detection results from Roboflow
                    'keypoints': keypoints,                      # Keypoint data from MoveNet
                    'image_info': roboflow_results['image']      # Additional image info from Roboflow
                }

                # Append the frame's combined annotation info to the annotations list
                annotations.append(frame_info)

                # Optionally display or save the annotated frame with both keypoints and objects drawn
                # cv2.imshow("Annotated Frame", annotated_frame)
                # cv2.waitKey(1)

    # Return combined annotations and annotated frames
    return annotations, annotated_frame 