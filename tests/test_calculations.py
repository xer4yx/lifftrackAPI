import cv2
import numpy as np
import matplotlib.pyplot as plt
from lifttrack.v2.comvis.Movenet import MovenetInference
from lifttrack.v2.comvis.features import (
    extract_joint_angles, 
    visualize_angles,
    filter_low_confidence_keypoints,
    calculate_relative_distances,
    detect_form_issues
)

def test_feature_visualization(image_path, exercise_type="general"):
    """
    Test function to visualize feature calculations on an image.
    
    Args:
    - image_path: Path to the image file
    - exercise_type: Type of exercise for form detection (default: "general")
    
    Returns:
    - None (displays a plot with annotations)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Convert from BGR to RGB for correct display in matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize Movenet
    movenet = MovenetInference()
    
    # Get keypoints
    keypoints = movenet.get_predictions(image)
    
    # Filter keypoints by confidence
    filtered_keypoints = filter_low_confidence_keypoints(keypoints, threshold=0.2)
    
    # Extract features
    joint_angles = extract_joint_angles(filtered_keypoints)
    relative_distances = calculate_relative_distances(filtered_keypoints)
    
    # Detect form issues if exercise type is specified
    form_issues = detect_form_issues({"joint_angles": joint_angles}, exercise_type)
    
    # Visualize the angles on the image
    annotated_image = visualize_angles(image_rgb, filtered_keypoints, joint_angles)
    
    # Create a figure with two subplots side by side
    plt.figure(figsize=(18, 10))
    
    # Plot the original image with keypoints
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image with Keypoints")
    
    # Draw keypoints on the original image
    for name, (x, y, conf) in filtered_keypoints.items():
        if conf > 0.2:
            plt.plot(x, y, 'go', markersize=5)
            plt.text(x+5, y+5, name, fontsize=8, color='blue')
    
    # Plot the annotated image with angles
    plt.subplot(1, 2, 2)
    plt.imshow(annotated_image)
    plt.title("Annotated Image with Joint Angles")
    
    # Add text box with calculations
    calculation_text = "Joint Angles:\n"
    for name, angle in sorted(joint_angles.items()):
        calculation_text += f"{name}: {angle:.1f}°\n"
    
    calculation_text += "\nRelative Distances:\n"
    for name, dist in sorted(relative_distances.items()):
        calculation_text += f"{name}: {dist:.2f}\n"
    
    if form_issues:
        calculation_text += f"\nDetected Form Issues ({exercise_type}):\n"
        for issue, value in form_issues.items():
            calculation_text += f"- {issue.replace('_', ' ').title()}\n"
    
    # Add text box to the right of the plot
    plt.figtext(0.98, 0.5, calculation_text, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8),
               verticalalignment='center', 
               horizontalalignment='right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    test_feature_visualization("./tests/images/barbell-deadlift-1.jpg", exercise_type="deadlift")