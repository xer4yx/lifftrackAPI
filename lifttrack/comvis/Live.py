import os
import time
import cv2
import tensorflow as tf
import numpy as np

# Define a mapping of class indices to class names
class_names = {
    0: "barbell_benchpress",
    1: "barbell_deadlift",
    2: "barbell_shoulderpress",
    3: "dumbbell_benchpress",
    4: "dumbbell_deadlift",
    5: "dumbbell_shoulderpress",
    6: "rdl_barbell"
}


class ExerciseFormAnalyzer:
    def __init__(self, model_path='model/exercise_model_20241031_153926.keras', input_shape=(112, 112),
                 max_video_length=30):
        self.__start_time = time.time()
        self.model = tf.keras.models.load_model(model_path)
        self.__end_time = time.time()
        print(f"{self.__class__.__name__} model loaded in {self.__end_time - self.__start_time:.2f} seconds")

        self.input_shape = input_shape
        self.max_video_length = max_video_length

        # Initialize buffers with proper shapes
        self.original_frame_buffer = np.zeros((max_video_length, *input_shape, 3), dtype=np.float16)
        self.feature_buffer = np.zeros((max_video_length, 9), dtype=np.float16)
        self.buffer_index = 0

    def process_frame_for_cnn(self, frame):
        """Prepare frame for CNN processing"""
        processed_frame = cv2.resize(frame, self.input_shape)
        processed_frame = processed_frame.astype(np.float32) / 255.0
        return processed_frame.astype(np.float16)

    def add_to_buffer(self, frame, features):
        """Add frame and features to circular buffer"""
        self.original_frame_buffer[self.buffer_index] = frame
        self.feature_buffer[self.buffer_index] = features
        self.buffer_index = (self.buffer_index + 1) % self.max_video_length

    def get_buffer_for_prediction(self):
        """Get properly arranged buffer for model prediction"""
        # Rearrange buffer to maintain temporal order
        frames = np.roll(self.original_frame_buffer, -self.buffer_index, axis=0)
        features = np.roll(self.feature_buffer, -self.buffer_index, axis=0)

        # Ensure that we are only using the filled part of the buffer
        num_frames = np.count_nonzero(frames[:, 0, 0, 0])  # Count valid frames
        frames = frames[:num_frames]  # Adjust frames shape
        features = features[:num_frames]  # Adjust features shape

        # Add batch dimension
        frames = np.expand_dims(frames, axis=0)  # Shape: (1, num_frames, height, width, channels)
        features = np.expand_dims(features, axis=0)  # Shape: (1, num_frames, 9)

        # Ensure that the number of frames does not exceed 30
        if frames.shape[1] < 30:
            # If there are fewer than 30 frames, pad with zeros
            padding = np.zeros((1, 30 - frames.shape[1], 112, 112, 3), dtype=np.float16)
            frames = np.concatenate([frames, padding], axis=1)

        if features.shape[1] < 30:
            # If there are fewer than 30 feature sets, pad with zeros
            padding = np.zeros((1, 30 - features.shape[1], 9), dtype=np.float16)
            features = np.concatenate([features, padding], axis=1)

        return frames, features

    def run_inference(self, frame):
        """Run CNN inference"""
        # Process frames
        processed_frame = self.process_frame_for_cnn(frame)

        # Add to buffer
        self.add_to_buffer(processed_frame, None)

        # Only predict if buffer is full
        if self.buffer_index == 0:
            try:
                frames, feature_data = self.get_buffer_for_prediction()

                # Ensure inputs match model's expected shapes
                frames = tf.convert_to_tensor(frames,
                                              dtype=tf.float32)  # Shape: (1, num_frames, height, width, channels)
                feature_data = tf.convert_to_tensor(feature_data, dtype=tf.float32)  # Shape: (1, num_frames, 9)

                # Create a data dictionary with all required inputs
                input_data = {
                    'input_layer_8': frames,  # Update this key to match your model's input layer name
                    'input_layer_9': feature_data  # Update this key to match your model's input layer name
                }

                # Make prediction
                with tf.device('/CPU:0'):  # Force CPU execution for consistent behavior
                    output = self.model.predict(input_data, verbose=0)

                return output
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                return None

        return None

    def run(self):
        """Main loop for video capture and analysis"""
        cap = cv2.VideoCapture(0)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Ensure frames directory exists
            os.makedirs("../../frames", exist_ok=True)
            frame_path = f"frames/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)

            # Process frame
            output = self.run_inference(frame, frame_path)
            if output is not None:
                print(f"Frame {frame_count} prediction:", output)

            # Display frame
            cv2.imshow('Exercise Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
