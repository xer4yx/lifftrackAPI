from fastapi import Request, WebSocket
from typing import Optional

from core.usecase import InferenceUseCase


async def get_inference_usecase(websocket: WebSocket) -> InferenceUseCase:
    """
    Get an instance of InferenceUseCase with all required inference services.

    Args:
        websocket: FastAPI websocket object

    Returns:
        InferenceUseCase instance with all required inference services
    """
    # Get the inference services from app state
    inference_services = websocket.app.state.inference_services

    # Create and return the InferenceUseCase with reduced workers for 4-core CPU
    # Use only 1 worker to prevent thread saturation since we're doing sequential processing
    inference_usecase = InferenceUseCase(
        object_detection_service=inference_services.get("roboflow"),
        pose_estimation_service=inference_services.get("posenet"),
        action_recognition_service=inference_services.get("videoaction"),
        max_workers=1,  # Reduce from default 4 to prevent thread saturation
    )

    return inference_usecase
