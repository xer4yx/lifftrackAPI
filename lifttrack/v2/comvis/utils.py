import numpy as np
import cv2

def resize_to_128x128(input_array):
    """Resize a given numpy array to 128x128."""
    if isinstance(input_array, np.ndarray):
        return cv2.resize(input_array, (128, 128))
    else:
        raise ValueError("Input must be a numpy array")

def resize_to_192x192(input_array):
    """Resize a given numpy array to 192x192."""
    if isinstance(input_array, np.ndarray):
        return cv2.resize(input_array, (192, 192))
    else:
        raise ValueError("Input must be a numpy array")