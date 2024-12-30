import io
import docker
import time

from lifttrack import config
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


def check_docker_container_status(logger,container_name: str = None):
    """
    Checks and ensures the Roboflow inference container is running.
    For production cloud deployment.
    """
    if container_name is None:
        raise ValueError("container_name cannot be None")

    retry_delay = 5
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            client = docker.from_env()
            
            # Check if container exists and its status
            try:
                container = client.containers.get(container_name)
                
                # Check container health
                container.reload()
                if container.status == "running":
                    # Verify container is healthy
                    if container.attrs.get('State', {}).get('Health', {}).get('Status') == 'healthy':
                        logger.info(f"Container {container_name} is running and healthy")
                        return
                    else:
                        logger.warning(f"Container {container_name} is running but may not be healthy")
                        return
                
                # If container exists but not running, remove it and recreate
                container.remove(force=True)
                logger.info(f"Removed existing container {container_name}")
                
            except docker.errors.NotFound:
                pass  # Container doesn't exist, will create new one
            
            # Create new container with health check
            container = client.containers.run(
                "roboflow/roboflow-inference-server-gpu:latest",
                name=container_name,
                ports={'9001/tcp': 9001},
                detach=True,
                gpu=True,
                network="host",
                healthcheck={
                    "test": ["CMD", "curl", "-f", "http://localhost:9001/health"],
                    "interval": 30000000000,  # 30 seconds in nanoseconds
                    "timeout": 3000000000,    # 3 seconds in nanoseconds
                    "retries": 3
                },
                restart_policy={"Name": "unless-stopped"}
            )
            
            # Wait for container to be healthy
            logger.info(f"Waiting for container {container_name} to be ready...")
            time.sleep(retry_delay)
            container.reload()
            
            if container.status == "running":
                logger.info(f"Container {container_name} started successfully")
                return
                
        except docker.errors.DockerException as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise RuntimeError(f"Failed to start container after {max_retries} attempts: {str(e)}")
            logger.error(f"Docker error (attempt {retry_count}/{max_retries}): {str(e)}")
            time.sleep(retry_delay)
            continue


class MoveNetInference:
    def __init__(self):
        self.__start_time = time.time()
        model = hub.load(config.get(section="TensorHub", option="MOVENET_MODEL"))
        self.movenet = model.signatures[config.get(section="TensorHub", option="MOVENET_SERVING_DEFAULT")]
        self.__end_time = time.time()
        print(f"{self.__class__.__name__} model loaded in {self.__end_time - self.__start_time:.2f} seconds")

    def run_keypoint_inference(self, input, inference_count=5):
        """
        Runs MoveNet model for Pose Estimation.
        """
        for _ in range(inference_count-1):
            inference = self.movenet(input)
        keypoints = inference[config.get(section="TensorHub", option="MOVENET_OUTPUT_BLOCK")]
        # Reshape keypoints to [17, 3] format
        return keypoints  # Returns shape [17, 3]


class RoboflowInference:
    def __init__(self):
        # check_docker_container_status(config.get(section="Docker", option="container_name"))
        self.project_id = config.get(section="Roboflow", option="ROBOFLOW_PROJECT_ID")
        self.model_version = int(config.get(section="Roboflow", option="ROBOFLOW_MODEL_VER"))
        self.__start_time = time.time()
        host_ip = config.get(section="Server", option="LOCAL_SERVER_HOST")
        self.roboflow_client = InferenceHTTPClient(
            api_url=f"http://{host_ip}:9001",
            api_key=config.get(section="Roboflow", option="ROBOFLOW_API_KEY")
        )
        self.__end_time = time.time()
        print(f"{self.__class__.__name__} connected to server.")

    def run_object_inference(self, frame):
        return self.roboflow_client.infer(
            inference_input=frame,
            model_id=f"{self.project_id}/{self.model_version}"
        )
