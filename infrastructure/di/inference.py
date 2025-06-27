from fastapi import Request
from core.interface import InferenceInterface


async def get_posenet_service(request: Request) -> InferenceInterface:
    """
    Dependency for PoseNet/MoveNet inference service that properly manages lifecycle.

    This is a dependency that returns the PoseNet service from the request app state.

    Args:
        request: The FastAPI request object

    Returns:
        An instance of InferenceInterface configured for PoseNet
    """
    return request.app.state.inference_services["posenet"]


async def get_roboflow_service(request: Request) -> InferenceInterface:
    """
    Dependency for Roboflow inference service that properly manages lifecycle.

    This is a dependency that returns the Roboflow service from the request app state.

    Args:
        request: The FastAPI request object

    Returns:
        An instance of InferenceInterface configured for Roboflow
    """
    return request.app.state.inference_services["roboflow"]


async def get_video_action_service(request: Request) -> InferenceInterface:
    """
    Dependency for Video Action inference service that properly manages lifecycle.

    This is a dependency that returns the Video Action service from the request app state.

    Args:
        request: The FastAPI request object

    Returns:
        An instance of InferenceInterface configured for Video Action
    """
    return request.app.state.inference_services["videoaction"]
