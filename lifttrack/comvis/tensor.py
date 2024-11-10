import warnings
from cryptography.utils import CryptographyDeprecationWarning

# Suppress CryptographyDeprecationWarning
warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)

import io
import docker
import time

from lifttrack import config, os
from lifttrack.comvis import tf, hub

from inference_sdk import InferenceHTTPClient


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


def check_docker_container_status(container_name: str = None):
    if container_name is None:
        raise ValueError("container_name cannot be None")

    retry_delay = 5
    docker_desktop_path = "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"
    max_wait_time = 300  # Maximum wait time of 5 minutes
    
    while True:
        try:
            # First check if Docker daemon is running
            try:
                client = docker.from_env()
                client.ping()  # Will raise exception if daemon isn't running
            except docker.errors.DockerException:
                print("Docker daemon not running. Attempting to start Docker Desktop...")
                # Start Docker Desktop on Windows 11
                os.system(f'start "" "{docker_desktop_path}"')
                print("Waiting for Docker Desktop to start...")
                
                # Incrementally wait for Docker Desktop to start
                wait_time = 0
                increment = 10  # Check every 10 seconds
                while wait_time < max_wait_time:
                    try:
                        client = docker.from_env()
                        client.ping()
                        print(f"Docker Desktop started successfully after {wait_time} seconds")
                        break
                    except docker.errors.DockerException:
                        time.sleep(increment)
                        wait_time += increment
                        print(f"Still waiting for Docker Desktop... ({wait_time} seconds)")
                
                if wait_time >= max_wait_time:
                    raise TimeoutError("Docker Desktop failed to start within 5 minutes")
                continue

            try:
                container = client.containers.get(container_name)
                if container.status == "running":
                    print(f"Docker Container {container_name} is already running")
                    break
                else:
                    container.start()
            except docker.errors.NotFound:
                # If container doesn't exist, create and start it
                container = client.containers.run(
                    "roboflow/roboflow-inference-server-gpu:latest",
                    name=container_name,
                    ports={'9001/tcp': 9001},
                    detach=True,
                    gpu=True,
                    network="host"
                )
            
            # Wait and verify container is running
            time.sleep(retry_delay)
            container.reload()
            if container.status == "running":
                print(f"Docker Container {container_name} started successfully")
                break
                
        except (docker.errors.DockerException,
                docker.errors.ImageNotFound, 
                docker.errors.APIError) as e:
            print(f"Warning: Docker container issue: {str(e)}")
            print("Retrying connection...")
            time.sleep(retry_delay)
            continue


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
        check_docker_container_status(config.get(section="Docker", option="container_name"))
        self.project_id = config.get(section="Roboflow", option="project_id")
        self.model_version = int(config.get(section="Roboflow", option="model_ver"))
        self.__start_time = time.time()
        host_ip = config.get(section="Server", option="host")
        self.roboflow_client = InferenceHTTPClient(
            api_url=f"http://{host_ip}:9001",
            api_key=config.get(section="Roboflow", option="api_key")
        )
        self.__end_time = time.time()
        print(f"{self.__class__.__name__} connected to server.")

    def run_object_inference(self, frame):
        return self.roboflow_client.infer(
            inference_input=frame,
            model_id=f"{self.project_id}/{self.model_version}"
        )
