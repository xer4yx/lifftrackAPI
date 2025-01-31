from pydantic import ValidationError
from typing import Optional, Annotated

from lifttrack import network_logger
from lifttrack.models import User, ExerciseData
from lifttrack.auth import get_current_user
from lifttrack.dbhandler.rest_rtdb import rtdb
from lifttrack.utils.logging_config import setup_logger, log_network_io

from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.util import get_remote_address

# Configure logging
logger = setup_logger("progress_routes", "router.log")

# Initialize router
router = APIRouter(
    prefix="/progress",
    tags=["progress"],
    responses={404: {"description": "Not found"}}
)

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)

@router.put("/{username}/{exercise}")
@limiter.limit("30/minute")
async def create_progress(
    username: str, 
    exercise: str, 
    exercise_data: ExerciseData,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    Endpoint to create or append exercise progress data.
    """
    try:
        reponse = None
        if username != current_user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify other user's progress"
            )

        rtdb.put_progress(username, exercise, exercise_data)
        
        response = JSONResponse(
            content={"msg": "Progress saved successfully"},
            status_code=status.HTTP_201_CREATED
        )
        return response
    except ValidationError as vale:
        response = JSONResponse(
            content={"msg": str(vale)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        return response
    except HTTPException as httpe:
        response = JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
        return response
    except Exception as e:
        response = JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        logger.exception(f"Error in create_progress: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )

@router.get("/{username}")
@limiter.limit("30/minute")
async def get_progress(
    username: str,
    request: Request,
    exercise: Annotated[Optional[str], Query(description="Exercise name")] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Endpoint to retrieve progress data.
    """
    try:
        response = None
        if username != current_user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot access other user's progress"
            )

        progress = rtdb.get_progress(username, exercise)
        if progress is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Progress data not found"
            )
            
        response = JSONResponse(
            content=progress,
            status_code=status.HTTP_200_OK
        )
        return response
    except HTTPException as httpe:
        response = JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
        return response
    except Exception as e:
        response = JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        logger.exception(f"Error in get_progress: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )

@router.delete("/{username}")
@limiter.limit("10/minute")
async def delete_progress(
    username: str,
    request: Request,
    exercise: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Endpoint to delete progress data.
    """
    try:
        response = None
        if username != current_user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete other user's progress"
            )

        deleted = rtdb.delete_progress(username, exercise)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Progress deletion failed"
            )
            
        response = JSONResponse(
            content={"msg": "Progress deleted successfully"},
            status_code=status.HTTP_200_OK
        )
        return response
    except HTTPException as httpe:
        response = JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
        return response
    except Exception as e:
        response = JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        logger.exception(f"Error in delete_progress: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )
