import io
from queue import Queue

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from numpy import ndarray

from lifttrack import cv2, Mat
from lifttrack.comvis.Live import ExerciseFormAnalyzer, class_names
from lifttrack.comvis.tensor import MoveNetInference, RoboflowInference
from lifttrack.utils import draw_prediction

frame_queue = Queue(maxsize=30)
result_queue = Queue(maxsize=30)

def run_inference(movenet, roboflow, frame: Mat | ndarray):
    """Run both MoveNet and Roboflow inference"""
    
    # MoveNet inference - properly preprocess the frame
    # First convert to float32 (0-255 range), then cast to int32
    input_img = tf.cast(frame, dtype=tf.float32)
    input_img = tf.expand_dims(input_img, axis=0)
    input_img = tf.image.resize_with_pad(
        input_img,
        target_height=192,
        target_width=192
    )
    input_img = tf.cast(input_img, dtype=tf.int32)
    
    # Run MoveNet inference
    keypoints = movenet.run_keypoint_inference(input_img)

    # Roboflow inference
    try:
        roboflow_results = roboflow.run_object_inference(frame)
    except Exception as e:
        print(f"Error during Roboflow inference: {e}")
        roboflow_results = {'predictions': [], 'image': {}}

    # Combine results
    frame_annotation = {
        'keypoints': keypoints[0],  # Extract from batch dimension
        'objects': roboflow_results['predictions'],
        'image_info': roboflow_results['image']
    }

    return frame_annotation

def extract_features(annotation):
    """Extract features from a single annotation"""
    if not isinstance(annotation, dict):
        print(f"Expected a dictionary for annotation, got {type(annotation)}: {annotation}")
        return np.zeros(9, dtype=np.float32)

    # Get keypoints tensor and convert to numpy
    keypoints_tensor = annotation.get('keypoints')
    if keypoints_tensor is None:
        return np.zeros(9, dtype=np.float32)
    
    # Convert tensor to numpy and create a dictionary mapping
    # Take first element from batch dimension [0]
    keypoints_array = keypoints_tensor.numpy()[0]  # Now shape is (17, 3) [y, x, confidence]
    keypoint_mapping = {
        'left_shoulder': keypoints_array[5],  # Indices based on MoveNet output
        'right_shoulder': keypoints_array[6],
        'left_elbow': keypoints_array[7],
        'left_wrist': keypoints_array[9],
        'left_hip': keypoints_array[11],
        'right_hip': keypoints_array[12],
        'left_knee': keypoints_array[13],
        'left_ankle': keypoints_array[15]
    }

    # Convert the format to match what the rest of the function expects
    keypoints = {k: [v[1], v[0]] for k, v in keypoint_mapping.items()}  # Swap x,y coordinates
    objects = annotation.get('objects', [])

    def calculate_angle(p1, p2, p3):
        vector1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        vector2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
            return 0.0
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def calculate_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    elbow_angle = calculate_angle(
        p1=keypoints.get('left_shoulder', [0, 0]),
        p2=keypoints.get('left_elbow', [0, 0]),
        p3=keypoints.get('left_wrist', [0, 0])
    )
    knee_angle = calculate_angle(
        p1=keypoints.get('left_hip', [0, 0]),
        p2=keypoints.get('left_knee', [0, 0]),
        p3=keypoints.get(' left_ankle', [0, 0])
    )
    hip_angle = calculate_angle(
        p1=keypoints.get('left_shoulder', [0, 0]),
        p2=keypoints.get('left_hip', [0, 0]),
        p3=keypoints.get('left_knee', [0, 0])
    )
    spine_vertical = calculate_angle(
        p1=keypoints.get('left_shoulder', [0, 0]),
        p2=keypoints.get('left_hip', [0, 0]),
        p3=[keypoints.get('left_hip', [0, 0])[0], 0]
    )

    weight_position = next(
        (calculate_distance([obj.get('x', 0), obj.get('y', 0)], keypoints.get('left_shoulder', [0, 0])) for obj
         in objects if obj.get('class') in ['barbell', 'dumbbell']), 0.0)

    shoulder_symmetry = abs(keypoints.get('left_shoulder', [0, 0])[1] - keypoints.get('right_shoulder', [0, 0])[1])
    stability = np.mean(
        [calculate_distance(keypoints.get('left_shoulder', [0, 0]), keypoints.get('right_shoulder', [0, 0])),
         calculate_distance(keypoints.get('left_hip', [0, 0]), keypoints.get('right_hip', [0, 0]))])
    bar_path_deviation = np.std(
        [pos.get('x', 0) for pos in objects if pos.get('class') in ['barbell', 'dumbbell']]) if objects else 0.0

    return np.array([
        elbow_angle / 180.00,
        knee_angle / 180.0,
        hip_angle / 180.0,
        spine_vertical / 90.0,
        weight_position / 100.0,
        shoulder_symmetry / 50.0,
        stability / 50.0,
        bar_path_deviation / 50.0,
        1.0  # Reserved for additional features
    ], dtype=np.float32)

def websocket_process_frames(analyzer, frame_data: bytes | io.BytesIO):
    """
    Process a single frame for inference.

    Args:
        frame_data: Camera frame sent as a bytes.

    Returns:
        Annotated frame with inference results.
    """
    try:
        # Convert bytes to numpy array properly
        np_arr = np.frombuffer(frame_data, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None or frame.size == 0:
            raise ValueError("Invalid image data or empty frame")

        height, width, _ = frame.shape
        print(frame.shape)

        # Prepare input for model (use the original frame, not resized yet)
        input_image = tf.image.resize(
            images=tf.expand_dims(frame, axis=0),
            size=(192, 192)
        )
        input_image = tf.cast(input_image, dtype=tf.int32)
        print(input_image.shape)

        # Run inference
        inference = run_inference(frame=frame)  # Use original frame for object detection
        print(inference)

        # Extract features
        features = extract_features(annotation=inference)
        print(features)

        # Process frame for CNN
        processed_frame = analyzer.process_frame_for_cnn(input_image)
        print(processed_frame.shape)

        # Add frame and features to the buffer
        analyzer.add_to_buffer(processed_frame, features)
        print(analyzer.buffer_index)

        # Run inference if buffer is full
        prediction = None
        if analyzer.buffer_index == 0:
            frames, feature_data = analyzer.get_buffer_for_prediction()

            # Ensure inputs match model's expected shapes
            frames = tf.convert_to_tensor(frames, dtype=tf.float32)
            feature_data = tf.convert_to_tensor(feature_data, dtype=tf.float32)

            # Create a data dictionary with all required inputs
            input_data = {
                'input_layer_8': frames,  # Update this key to match your model's input layer name
                'input_layer_9': feature_data  # Update this key to match your model's input layer name
            }

            # Make prediction
            with tf.device('/CPU:0'):
                prediction = analyzer.model.predict(input_data, verbose=0)
                
            print(prediction)

        # Draw predictions
        annotated_frame = draw_prediction(
            image=frame,
            keypoints_with_scores=inference['keypoints'],
            output_image_height=height
        )

        if prediction is None:
            prediction_data = {
                "prediction": None,
                "predicted_class": None,
                "class_name": "Unknown"
            }
        else:
            prediction_data = {
                "prediction": prediction.tolist(),
                "predicted_class": int(np.argmax(prediction[0])),
                "class_name": class_names.get(int(np.argmax(prediction[0])), "Unknown")
            }

        print("Inference successful.")
        return annotated_frame, prediction_data

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None, None

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
