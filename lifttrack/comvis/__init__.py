from queue import Queue
import threading

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from lifttrack import cv2
from lifttrack.comvis.tensor import MoveNetInference, RoboflowInference
from lifttrack.utils import draw_prediction

movenet = MoveNetInference()
roboflow = RoboflowInference()

frame_queue = Queue(maxsize=30)
result_queue = Queue(maxsize=30)


def run_inference(frame):
    """Run both MoveNet and Roboflow inference"""

    # MoveNet inference
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.int32)
    keypoints = movenet.run_keypoint_inference(input_img)

    # Roboflow inference
    try:
        roboflow_results = roboflow.run_object_inference(frame)
    except Exception as e:
        print(f"Error during Roboflow inference: {e}")
        roboflow_results = {'predictions': [], 'image': {}}

    # Combine results
    frame_annotation = {
        'keypoints': keypoints,
        'objects': roboflow_results['predictions'],
        'image_info': roboflow_results['image']
    }

    return frame_annotation


def extract_features(self, annotation):
    """Extract features from a single annotation"""
    if not isinstance(annotation, dict):
        print(f"Expected a dictionary for annotation, got {type(annotation)}: {annotation}")
        return np.zeros(9, dtype=np.float32)

    keypoints = annotation.get('keypoints', {})
    objects = annotation.get('objects', [])

    if not keypoints:
        return np.zeros(9, dtype=np.float32)

    elbow_angle = self.calculate_angle(keypoints.get('left_shoulder', [0, 0]), keypoints.get('left_elbow', [0, 0]),
                                       keypoints.get('left_wrist', [0, 0]))
    knee_angle = self.calculate_angle(keypoints.get('left_hip', [0, 0]), keypoints.get('left_knee', [0, 0]),
                                      keypoints.get(' left_ankle', [0, 0]))
    hip_angle = self.calculate_angle(keypoints.get('left_shoulder', [0, 0]), keypoints.get('left_hip', [0, 0]),
                                     keypoints.get('left_knee', [0, 0]))
    spine_vertical = self.calculate_angle(keypoints.get('left_shoulder', [0, 0]), keypoints.get('left_hip', [0, 0]),
                                          [keypoints.get('left_hip', [0, 0])[0], 0])

    weight_position = next(
        (self.calculate_distance([obj.get('x', 0), obj.get('y', 0)], keypoints.get('left_shoulder', [0, 0])) for obj
         in objects if obj.get('class') in ['barbell', 'dumbbell']), 0.0)

    shoulder_symmetry = abs(keypoints.get('left_shoulder', [0, 0])[1] - keypoints.get('right_shoulder', [0, 0])[1])
    stability = np.mean(
        [self.calculate_distance(keypoints.get('left_shoulder', [0, 0]), keypoints.get('right_shoulder', [0, 0])),
         self.calculate_distance(keypoints.get('left_hip', [0, 0]), keypoints.get('right_hip', [0, 0]))])
    bar_path_deviation = np.std(
        [pos.get('x', 0) for pos in objects if pos.get('class') in ['barbell', 'dumbbell']]) if objects else 0.0

    return np.array([
        elbow_angle / 180.0,
        knee_angle / 180.0,
        hip_angle / 180.0,
        spine_vertical / 90.0,
        weight_position / 100.0,
        shoulder_symmetry / 50.0,
        stability / 50.0,
        bar_path_deviation / 50.0,
        1.0  # Reserved for additional features
    ], dtype=np.float32)


def websocket_process_frames(frame_data):
    # Decode the frame data
    np_frame = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    height, width, _ = frame.shape

    # Prepare input for model
    input_image = cv2.resize(frame, (192, 192))
    input_image = tf.expand_dims(input_image, axis=0)
    input_image = tf.cast(input_image, dtype=tf.int32)

    # Run inference
    keypoints = movenet.run_keypoint_inference(input_image)

    # Draw predictions
    annotated_frame = draw_prediction(
        frame,
        keypoints,
        output_image_height=height
    )

    print(annotated_frame)

    return annotated_frame


# def process_frames():
#     while True:
#         frame = frame_queue.get()
#
#         # Decode the frame data
#         np_frame = np.frombuffer(frame, np.uint8)
#         frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
#
#         # Prepare input for model
#         input_image = cv2.resize(frame, (192, 192))
#         input_image = tf.expand_dims(input_image, axis=0)
#         input_image = tf.cast(input_image, dtype=tf.int32)
#
#         # Run inference
#         keypoints = movenet.run_inference(input_image)
#
#         # Draw predictions
#         annotated_frame = draw_prediction(
#             frame,
#             keypoints,
#             output_image_height=480
#         )
#
#         result_queue.put(annotated_frame)
#         frame_queue.task_done()
#
#
# def generate_frames():
#     cap = cv2.VideoCapture(0)
#
#     try:
#         while True:
#             success, frame = cap.read()
#
#             if not success:
#                 break
#
#             if not frame_queue.full():
#                 frame_queue.put(frame.tobytes())
#
#             if not result_queue.empty():
#                 annotated_frame = result_queue.get()
#
#                 # Encode frame for streaming
#                 ret, buffer = cv2.imencode('.jpg', annotated_frame)
#                 frame_bytes = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#
#     finally:
#         cap.release()
#
#
# # Start the processing thread
# process_thread = threading.Thread(target=process_frames, daemon=True)
# process_thread.start()