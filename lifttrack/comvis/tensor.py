import io
import time

from lifttrack import config
from lifttrack.comvis import tf, hub

from inference_sdk import InferenceHTTPClient

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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


class MoveNetInference:
    def __init__(self):
        self.__start_time = time.time()
        model = hub.load(config.get(section="TensorHub", option="model"))
        self.movenet = model.signatures[config.get(section="TensorHub", option="signature")]
        self.__end_time = time.time()
        print(f"{self.__class__.__name__} model loaded in {self.__end_time - self.__start_time:.2f} seconds")

    def run_keypoint_inference(self, input, inference_count=5):
        """
        Runs MoveNet model for Pose Estimation.
        """
        for _ in range(inference_count-1):
            inference = self.movenet(input)
        keypoints = inference[config.get(section="TensorHub", option="inference")]
        # Reshape keypoints to [17, 3] format
        return keypoints  # Returns shape [17, 3]


class RoboflowInference:
    def __init__(self):
        self.project_id = config.get(section="Roboflow", option="project_id")
        self.model_version = int(config.get(section="Roboflow", option="model_ver"))
        self.__start_time = time.time()
        self.roboflow_client = InferenceHTTPClient(
            api_url=config.get(section="Roboflow", option="api_url"),
            api_key=config.get(section="Roboflow", option="api_key")
        )
        self.__end_time = time.time()
        print(f"{self.__class__.__name__} model loaded in {self.__end_time - self.__start_time:.2f} seconds")

    def run_object_inference(self, frame):
        return self.roboflow_client.infer(
            inference_input=frame,
            model_id=f"{self.project_id}/{self.model_version}"
        )
