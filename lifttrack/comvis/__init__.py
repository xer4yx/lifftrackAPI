from queue import Queue
import threading

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from lifttrack import cv2
from lifttrack.comvis.tensor import MoveNetHelper
from lifttrack.utils import draw_prediction

movenet = MoveNetHelper()

frame_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)


def websocket_process_frames(frame_data):
    # Decode the frame data
    np_frame = np.frombuffer(frame_data, np.uint8)
    print(np_frame)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    print(frame.shape)
    height, width, _ = frame.shape

    # Prepare input for model
    input_image = cv2.resize(frame, (192, 192))
    input_image = tf.expand_dims(input_image, axis=0)
    input_image = tf.cast(input_image, dtype=tf.int32)

    # Run inference
    keypoints = movenet.run_inference(input_image)

    # Draw predictions
    annotated_frame = draw_prediction(
        frame,
        keypoints,
        output_image_height=height
    )

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