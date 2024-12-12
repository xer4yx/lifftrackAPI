import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from lifttrack import config
from lifttrack.v2.comvis.utils import resize_to_192x192
from lifttrack.utils.logging_config import setup_logger

logger = setup_logger("movenet-v2", "comvis.log")

# Initialize dictionary for keypoints
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}


class MovenetInference:
    def __init__(self):
        self.__movenet_model = hub.load(config.get('TensorHub', 'model'))
        self.__movenet = self.__movenet_model.signatures['serving_default']
        # List to store annotations (keypoints) for later use
        self.__annotations_list = []
        
    def process_keypoints(self, keypoints, image_shape):
        y, x, _ = image_shape
        shaped_keypoints = {}

        for name, index in KEYPOINT_DICT.items():
            ky, kx, kp_conf = keypoints[index]
            cx, cy = int(kx * x), int(ky * y)
            shaped_keypoints[name] = (cx, cy, float(kp_conf))

        return shaped_keypoints

    def draw_keypoints_on_frame(self, frame, keypoints):
        for name, (x, y, confidence) in keypoints.items():
            if confidence > 0.5:  # Only draw if confidence is above 0.5
                # Draw keypoint as a small circle
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # Green dot
                # Optionally, add text label
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        return frame

    def analyze_frame(self, frame):
        # Resize the frame using your resize function
        resized_frame = resize_to_192x192(frame)

        # Prepare the image for MoveNet (add batch dimension and resize with padding)
        img = tf.image.resize_with_pad(tf.expand_dims(resized_frame, axis=0), 192, 192)
        input_img = tf.cast(img, dtype=tf.int32)
        
        # MoveNet inference
        results = self.__movenet(input_img)
        keypoints = results['output_0'].numpy()[0, 0, :, :3]
        print(f"Received Movenet Keypoints: {keypoints}")  # Extract keypoints

        # Process keypoints for annotations
        shaped_keypoints = self.process_keypoints(keypoints, frame.shape)

        # Save the annotations to the list
        self.__annotations_list.append(shaped_keypoints)

        # Draw keypoints on the frame
        annotated_frame = self.draw_keypoints_on_frame(frame.copy(), shaped_keypoints)

        return annotated_frame, shaped_keypoints
