import io
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
        model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.movenet = model.signatures['serving_default']

    def run_inference(self, input):
        """
        Runs MoveNet model for Pose Estimation.
        """
        movenet_inference = self.movenet(input)
        keypoints = movenet_inference['output_0']

        return keypoints
