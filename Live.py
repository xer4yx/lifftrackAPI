import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.models import load_model
from inference_sdk import InferenceHTTPClient

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

def print_prediction(prediction):
    """
    Prints the class name of the predicted class based on the model's output.

    Parameters:
    - prediction: A numpy array or list containing the predicted probabilities for each class.
    """
    # Convert prediction to a numpy array if it isn't already
    prediction = np.array(prediction)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)

    # Get the class name from the mapping
    predicted_class_name = class_names[predicted_class_index]

    # Print the result
    print(f"Predicted class index: {predicted_class_index}, Class name: {predicted_class_name}")

class ExerciseFormAnalyzer:
    def __init__(self, model_path='model/exercise_model_20241027_163105.keras', input_shape=(112, 112), max_video_length=30):
        self.model = load_model(model_path)
        self.input_shape = input_shape
        self.max_video_length = max_video_length
        
        # Initialize buffers with proper shapes
        self.original_frame_buffer = np.zeros((max_video_length, *input_shape, 3), dtype=np.float16)
        self.feature_buffer = np.zeros((max_video_length, 9), dtype=np.float16)
        self.buffer_index = 0
        
        # Load MoveNet
        self.movenet_model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.movenet = self.movenet_model.signatures['serving_default']
        
        # Initialize Roboflow
        self.roboflow_client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key="TiK2P0kPcpeVnssORWRV"
        )
        
        self.project_id = "lifttrack"
        self.model_version = 4
        
        # Keypoint dictionary
        self.KEYPOINT_DICT = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }

    def calculate_angle(self, p1, p2, p3):
        """Calculate the angle formed by three points."""
        vector1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        vector2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
            return 0.0
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def calculate_distance(self, p1, p2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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

    def extract_features(self, annotation):
        """Extract features from a single annotation"""
        if not isinstance(annotation, dict):
            print(f"Expected a dictionary for annotation, got {type(annotation)}: {annotation}")
            return np.zeros(9, dtype=np.float32)

        keypoints = annotation.get('keypoints', {})
        objects = annotation.get('objects', [])

        if not keypoints:
            return np.zeros(9, dtype=np.float32)

        elbow_angle = self.calculate_angle(keypoints.get('left_shoulder', [0, 0]), keypoints.get('left_elbow', [0, 0]), keypoints.get('left_wrist', [0, 0]))
        knee_angle = self.calculate_angle(keypoints.get('left_hip', [0, 0]), keypoints.get('left_knee', [0, 0]), keypoints.get(' left_ankle', [0, 0]))
        hip_angle = self.calculate_angle(keypoints.get('left_shoulder', [0, 0]), keypoints.get('left_hip', [0, 0]), keypoints.get('left_knee', [0, 0]))
        spine_vertical = self.calculate_angle(keypoints.get('left_shoulder', [0, 0]), keypoints.get('left_hip', [0, 0]), [keypoints.get('left_hip', [0, 0])[0], 0])

        weight_position = next((self.calculate_distance([obj.get('x', 0), obj.get('y', 0)], keypoints.get('left_shoulder', [0, 0])) for obj in objects if obj.get('class') in ['barbell', 'dumbbell']), 0.0)

        shoulder_symmetry = abs(keypoints.get('left_shoulder', [0, 0])[1] - keypoints.get('right_shoulder', [0, 0])[1])
        stability = np.mean([self.calculate_distance(keypoints.get('left_shoulder', [0, 0]), keypoints.get('right_shoulder', [0, 0])),
                            self.calculate_distance(keypoints.get('left_hip', [0, 0]), keypoints.get('right_hip', [0, 0]))])
        bar_path_deviation = np.std([pos.get('x', 0) for pos in objects if pos.get('class') in ['barbell', 'dumbbell']]) if objects else 0.0

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

    def run_inference(self, frame, frame_path):
        """Run both MoveNet and Roboflow inference"""
        # MoveNet inference
        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
        input_img = tf.cast(img, dtype=tf.int32)
        results = self.movenet(input_img)
        keypoints = results['output_0'].numpy()[0, 0, :, :3]
        
        # Process keypoints
        shaped_keypoints = self.process_keypoints(keypoints, frame.shape)
        
        # Roboflow inference
        try:
            roboflow_results = self.roboflow_client.infer(
                frame_path, 
                model_id=f"{self.project_id}/{self.model_version}"
            )
        except Exception as e:
            print(f"Error during Roboflow inference: {e}")
            roboflow_results = {'predictions': [], 'image': {}}

        # Combine results
        frame_annotation = {
            'keypoints': shaped_keypoints,
            'objects': roboflow_results['predictions'],
            'image_info': roboflow_results['image']
        }
        
        return frame_annotation

    def process_keypoints(self, keypoints, image_shape):
        """Process keypoints into a structured format"""
        y, x, _ = image_shape
        shaped_keypoints = {}
        for name, index in self.KEYPOINT_DICT.items():
            ky, kx, kp_conf = keypoints[index]
            cx, cy = int(kx * x), int(ky * y)
            shaped_keypoints[name] = (cx, cy, float(kp_conf))
        return shaped_keypoints

    def analyze_frame(self, frame, frame_path):
        """Process a single frame and return predictions if buffer is full"""
        # Process frame
        processed_frame = self.process_frame_for_cnn(frame)
        frame_annotation = self.run_inference(frame, frame_path)
        
        # Pass the single frame annotation directly
        features = self.extract_features(frame_annotation)

        # Add to buffer
        self.add_to_buffer(processed_frame, features)

        # Only predict if buffer is full
        if self.buffer_index == 0:
            try:
                frames, feature_data = self.get_buffer_for_prediction()

                # Ensure inputs match model's expected shapes
                frames = tf.convert_to_tensor(frames, dtype=tf.float32)  # Shape: (1, num_frames, height, width, channels)
                feature_data = tf.convert_to_tensor(feature_data, dtype=tf.float32)  # Shape: (1, num_frames, 9)

                # Create a data dictionary with all required inputs
                input_data = {
                    'input_layer_8': frames,  # Update this key to match your model's input layer name
                    'input_layer_9': feature_data  # Update this key to match your model's input layer name
                }

                # Make prediction
                with tf.device('/CPU:0'):  # Force CPU execution for consistent behavior
                    output = self.model.predict(input_data, verbose=0)

                # Print prediction
                print_prediction(output)

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
            os.makedirs("frames", exist_ok=True)
            frame_path = f"frames/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            
            # Process frame
            output = self.analyze_frame(frame, frame_path)
            if output is not None:
                print(f"Frame {frame_count} prediction:", output)
            
            # Display frame
            cv2.imshow('Exercise Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = ExerciseFormAnalyzer()
    analyzer.run()