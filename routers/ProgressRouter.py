from pydantic import ValidationError
from typing import Optional, Annotated
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Query, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from lifttrack import network_logger
from lifttrack.models import User, ExerciseData
from lifttrack.auth import get_current_user
from lifttrack.dbhandler.rest_rtdb import RTDBHelper
from lifttrack.utils.logging_config import setup_logger, log_network_io
from .manager import HTTPConnectionPool

from infrastructure.database import FirebaseREST
from infrastructure.di import get_firebase_rest

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
    response: Response,
    current_user: User = Depends(get_current_user)
):
    """
    Endpoint to create or append exercise progress data.
    """
    try:
        if username != current_user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify other user's progress"
            )
        async with HTTPConnectionPool.get_session() as session:
            async with RTDBHelper(session) as rtdb:
                await rtdb.put_progress(username, exercise, exercise_data)
            
                return JSONResponse(
                    content={"msg": "Progress saved successfully"},
                    status_code=status.HTTP_201_CREATED
                )
    except ValidationError as vale:
        return JSONResponse(
            content={"msg": str(vale)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in create_progress: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
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
    response: Response,
    exercise: Annotated[Optional[str], Query(description="Exercise name")] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Endpoint to retrieve progress data.
    """
    try:
        if username != current_user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot access other user's progress"
            )
            
        # Use HTTPConnectionPool directly instead of db dependency
        async with HTTPConnectionPool.get_session() as session:
            async with RTDBHelper(session) as rtdb:
                # Construct the key based on whether exercise is provided
                key = f"progress/{username}"
                if exercise:
                    key += f"/{exercise}"
                    
                progress = await rtdb.get_progress(username, exercise)
                
                # Handle case when progress data is not found
                if progress is None:
                    return JSONResponse(
                        content={},  # Return empty object instead of null
                        status_code=status.HTTP_200_OK
                    )
                
                return JSONResponse(
                    content=progress,
                    status_code=status.HTTP_200_OK
                )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in get_progress: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
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
    request: Request,
    response: Response,
    username: str,
    exercise: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Endpoint to delete progress data.
    """
    try:
        if username != current_user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete other user's progress"
            )
        async with HTTPConnectionPool.get_session() as session:
            async with RTDBHelper(session) as rtdb:
                deleted = await rtdb.delete_progress(username, exercise)
                if not deleted:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Progress deletion failed"
                    )
                    
                return JSONResponse(
                    content={"msg": "Progress deleted successfully"},
                    status_code=status.HTTP_200_OK
                )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in delete_progress: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )
