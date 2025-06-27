from typing_extensions import Annotated
from fastapi import APIRouter, Depends, status, Response, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Optional, Annotated

from infrastructure.database import FirebaseAdmin
from interface.di.auth_service_di import get_current_user
from lifttrack.models import User
from lifttrack.utils.logging_config import setup_logger

from infrastructure.di import get_firebase_admin

logger = setup_logger("metrics_routes", "router.log")

metrics_router = APIRouter(
    prefix="/metrics",
    tags=["v1-metrics"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication failed"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
    },
)

limiter = Limiter(key_func=get_remote_address)



@metrics_router.get("/{username}/{exercise}", status_code=status.HTTP_200_OK)
@limiter.limit("30/minute")
async def get_metrics(
    username: str,
    exercise: str,
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: FirebaseAdmin = Depends(get_firebase_admin),
):
    """
    Endpoint to get metrics for a user.
    """
    try:
        if username != current_user.username:
            logger.error(f"User {username} is not authorized to access metrics for {current_user.username}")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not authorized to access this resource")
        key = f"metrics/{username}/{exercise.replace("_", " ")}"
        metrics = await db.get_data(key)
        if metrics is None:
            logger.error(f"Metrics not found for {username}/{exercise}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics not found")
        logger.info(f"Metrics found for {username}. Exercise type: {exercise}")
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics for {username}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@metrics_router.get("/{username}", status_code=status.HTTP_200_OK)
@limiter.limit("30/minute")
async def get_all_metrics(
    username: str,
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: FirebaseAdmin = Depends(get_firebase_admin),
):
    """
    Endpoint to get all metrics for a user.
    """
    try:
        if username != current_user.username:
            logger.error(f"User {username} is not authorized to access metrics for {current_user.username}")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not authorized to access this resource")
        key = f"metrics/{username}"
        metrics = await db.get_data(key)
        if metrics is None:
            logger.error(f"Metrics not found for {username}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics not found")
        logger.info(f"Metrics found for {username}")
        return metrics
    except Exception as e:
        logger.error(f"Error getting all metrics for {username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))