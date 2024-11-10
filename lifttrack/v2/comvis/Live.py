import numpy as np
import cv2
import tensorflow as tf

# Load your trained model (replace 'path_to_model' with actual path)
model = tf.keras.models.load_model('path_to_model.h5')

# Class names mapping
class_names = {
    0: "barbell_benchpress",
    1: "barbell_deadlift",
    2: "barbell_rdl",
    3: "barbell_shoulderpress", 
    4: "dumbbell_benchpress",  
    5: "dumbbell_deadlift",
    6: "dumbbell_shoulderpress", 
}

def resize_to_128x128(input_array):
    """Resize a given numpy array to 128x128."""
    if isinstance(input_array, np.ndarray):
        resized_array = cv2.resize(input_array, (128, 128))
        return resized_array
    else:
        raise ValueError("Input must be a numpy array")

def infer_live(resized_image):
    """
    Perform inference using the resized 128x128 image.
    
    Args:
    - resized_image: Numpy array of shape (128, 128, 3)
    
    Returns:
    - predicted_class_index: Integer index of the predicted class
    """
    # Preprocess the image as per your model's requirements
    input_image = resized_image.astype('float32') / 255.0
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    return predicted_class_index
