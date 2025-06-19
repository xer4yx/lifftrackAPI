from typing import Dict, Any, Optional

from core.interface import InferenceInterface
from infrastructure.inference.roboflow_inference import RoboflowInferenceService
from infrastructure.inference.movenet_inference import PoseNetInferenceService
from infrastructure.inference.video_action_inference import VideoActionInferenceService


class InferenceServiceFactory:
    """
    Factory for creating inference service instances.
    This follows the factory pattern to create appropriate inference service implementations.
    """

    @staticmethod
    def create_inference_service(
        service_type: str, config: Optional[Dict[str, Any]] = None
    ) -> InferenceInterface:
        """
        Create an inference service instance based on type.

        Args:
            service_type: Type of inference service to create
            config: Optional configuration parameters

        Returns:
            An instance of an InferenceInterface implementation

        Raises:
            ValueError: If an unsupported service type is requested
        """
        if config is None:
            config = {}

        if service_type.lower() == "roboflow":
            return RoboflowInferenceService(
                api_url=config.get("api_url"),
                api_key=config.get("api_key"),
                project_id=config.get("project_id"),
                model_version=config.get("model_version"),
                max_workers=config.get("max_workers", 4),
            )
        elif service_type.lower() in ["posenet", "movenet"]:
            return PoseNetInferenceService(
                model_url=config.get("model_url"),
                confidence_threshold=config.get("confidence_threshold", 0.1),
                max_workers=config.get("max_workers", 4),
                enable_gpu=config.get("enable_gpu", True),
            )
        elif service_type.lower() in ["videoaction", "action3d", "exercise"]:
            return VideoActionInferenceService(
                model_path=config.get("model_path"),
                fallback_model_path=config.get("fallback_model_path"),
                num_frames=config.get("num_frames", 30),
                max_workers=config.get("max_workers", 4),
                enable_gpu=config.get("enable_gpu", True),
            )
        else:
            raise ValueError(f"Unsupported inference service type: {service_type}")
