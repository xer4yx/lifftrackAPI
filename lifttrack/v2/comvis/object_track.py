import cv2
import os
from inference_sdk import InferenceHTTPClient

from lifttrack import config
from lifttrack.comvis.tensor import check_docker_container_status
from lifttrack.utils.logging_config import setup_logger
from lifttrack.models import Object

container_logger = setup_logger("docker-container", "container.log")
comvis_logger = setup_logger("roboflow-v2", "comvis.log")
    

class ObjectTracker:
    def __init__(self) -> None:
        # Initialize the Roboflow Inference Client
        # check_docker_container_status(container_logger, config.get(section="Docker", option="CONTAINER_NAME"))
        self.__client = InferenceHTTPClient(
            api_url="http://localhost:9001",  # URL for the Roboflow Inference Client
            api_key="TiK2P0kPcpeVnssORWRV",  # Your API Key
        )
        self.__project_id = config.get(section="Roboflow", option="ROBOFLOW_PROJECT_ID")
        self.__model_version = int(config.get(section="Roboflow", option="ROBOFLOW_MODEL_VER"))
        
    # Function to draw bounding boxes on the frame with scaling
    def draw_bounding_boxes(self, frame, predictions, original_size):
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
    def draw_keypoints(self, frame, keypoints):
        for point, (x, y, confidence) in keypoints.items():
            if confidence > 0.5:  # Draw only confident keypoints
                frame = cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dots for keypoints
        return frame
    
    def process_frames_and_get_annotations(self, frame):
        try:
            # Run Roboflow inference for object detection
            roboflow_results = self.__client.infer(frame, model_id=f"{self.__project_id}/{self.__model_version}")
            comvis_logger.info(f"Received Roboflow Inference: {roboflow_results}")
            
            if not roboflow_results or not isinstance(roboflow_results, dict):
                return []  # Return empty dict for invalid results
            
            predictions = [
                Object(
                    x=pred.get('x', 0),
                    y=pred.get('y', 0),
                    width=pred.get('width', 0),
                    height=pred.get('height', 0),
                    confidence=pred.get('confidence', 0),
                    classs_id=pred.get('class_id', 0),
                    **{'class': pred.get('class', '')}
                    ).model_dump() for pred in roboflow_results['predictions']]
            return predictions
            
        except Exception as e:
            comvis_logger.error(f"Error during object detection: {e}")
            return []  # Return empty dict on error
