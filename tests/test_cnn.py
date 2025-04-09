import cv2
import numpy as np
import sys
import os

# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import lifttrack
from lifttrack.v2.comvis import ThreeDimInference

three_dim_inference = ThreeDimInference()

def test_cnn():
    image = cv2.imread("./tests/images/seated-dumbbell-press-1.jpg")
    image = np.array(image)
    
    for _ in range(10):
        logits = three_dim_inference.predict_class(image)
        print(logits)

if __name__ == "__main__":
    test_cnn()