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
