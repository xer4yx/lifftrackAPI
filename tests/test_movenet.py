import tensorflow_hub as hub

def test_pose_estimation():
    singlepose_lightning = 'https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-lightning/4'
    singlepose_thunder = 'https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder'
    movenet = hub.load(singlepose_lightning)
    print(movenet.signatures['serving_default'])
    
if __name__ == "__main__":
    test_pose_estimation()
