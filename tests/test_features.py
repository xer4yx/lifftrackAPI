import cv2
import numpy as np
import matplotlib.pyplot as plt
from lifttrack.v2.comvis.Movenet import MovenetInference
from lifttrack.v2.comvis.features import extract_joint_angles

KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}


def test_movenet_with_joint_angles():
    # Initialize MovenetInference
    movenet = MovenetInference()
    
    # Load a sample image
    # Replace with your actual image path
    image_path = "./tests/images/seated-dumbbell-press-1.webp"
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert BGR to RGB (OpenCV loads as BGR, but we want RGB for display)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MovenetInference
    keypoints = movenet.get_predictions(frame_rgb)
    
    # Extract joint angles from the keypoints
    joint_angles = extract_joint_angles(keypoints)
    
    # Print the detected keypoints and calculated angles
    print("Detected Keypoints:")
    for key, value in keypoints.items():
        print(f"{key}: position={value[:2]}, confidence={value[2]:.2f}")
    
    print("\nCalculated Joint Angles:")
    for angle_name, angle_value in joint_angles.items():
        print(f"{angle_name}: {angle_value:.2f} degrees")
    
    plt.figure(figsize=(20, 20))
    for name, (x, y, confidence) in keypoints.items():    
        cv2.circle(frame_rgb, (x, y), 4, (0, 255, 0), -1)  # Green dot
        # Optionally, add text label
        cv2.putText(frame_rgb, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    plt.title("Pose Detection with Joint Angles")
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_movenet_with_joint_angles()