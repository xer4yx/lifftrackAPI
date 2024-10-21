import io
import time
import tensorflow as tf
import tensorflow_hub as hub


def cast_image(image):
    """
    Casts the image to 192x192 and int32 based from the model's input requirements.
    """
    if type(image) not in [bytes, io.BytesIO]:
        raise TypeError(f"Given image type in {type(image)} instead of bytes.")

    image = tf.cast(
        x=tf.image.resize_with_pad(
            image=image,
            target_height=192,
            target_width=192),
        dtype=tf.int32)
    return image


class MoveNetHelper:
    def __init__(self):
        start_time = time.time()
        model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.movenet = model.signatures['serving_default']
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds")

    def run_inference(self, input, inference_count=5):
        """
        Runs MoveNet model for Pose Estimation.
        """
        for _ in range(inference_count-1):
            inference = self.movenet(input)
        keypoints = inference['output_0']
        # Reshape keypoints to [17, 3] format
        return keypoints  # Returns shape [17, 3]