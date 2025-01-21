from typing import List
from pydantic import ValidationError

from fastapi import APIRouter, Body, Depends, Request, Response, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from lifttrack import network_logger
from lifttrack.models import User
from lifttrack.auth import LiftTrackAuthenticator
from lifttrack.utils.logging_config import log_network_io

from core.services import UserService
from infrastructure.database import DatabaseFactory
from infrastructure import user_service_admin
from interfaces.api.schemas import UserCreateSchema, UserUpdateSchema
from interfaces.api import (
    RESPONSE_201, 
    RESPONSE_200, 
    RESPONSE_400, 
    RESPONSE_401, 
    RESPONSE_404, 
    RESPONSE_405, 
    RESPONSE_412, 
    RESPONSE_422, 
    RESPONSE_500, 
    RESPONSE_503
)
from utilities.monitoring.factory import MonitoringFactory

logger = MonitoringFactory.get_logger("v2-user-router")

router = APIRouter(
    prefix="/v2",
    tags=["v2-user"],
    responses={
        200: RESPONSE_200,
        201: RESPONSE_201,
        400: RESPONSE_400,
        401: RESPONSE_401,
        404: RESPONSE_404,
        405: RESPONSE_405,
        412: RESPONSE_412,
        422: RESPONSE_422,
        500: RESPONSE_500,
        503: RESPONSE_503
    }
)
limiter = Limiter(key_func=get_remote_address)

@router.post("/users", response_model=User, status_code=201)
@limiter.limit("5/minute")
async def create_user(
    request: Request, 
    response: Response,
    user_service: UserService = Depends(user_service_admin),
    user: UserCreateSchema = Body(..., example={
        "first_name": "John",
        "last_name": "Doe",
        "username": "johndoe",
        "email": "john.doe@example.com",
        "phone_number": "09123456789",
        "password": "Password123!"
    })
):
    """
        Create a new user in the Firebase database.
        
        Args:
            user (User): User creation details
            db (FirebaseDBHelper): Database connection
        
        Returns:
            User: Created user details
    """
    try:
        result = await user_service.create_user(
            username=user.username,
            email=user.email,
            password=user.password,
            first_name=user.first_name,
            last_name=user.last_name,
            phone_number=user.phone_number
        )
        
        if not result:
            logger.error(f"User creation failed: {result}")
            return JSONResponse(
                content={"msg": "User creation failed."},
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return JSONResponse(
            content={"msg": "User created successfully"},
            status_code=status.HTTP_201_CREATED
        )
    except ValidationError as ve:
        logger.error(f"Validation error in create_user: {str(ve)}")
        return JSONResponse(
            content={"msg": str(ve)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except Exception as e:
        logger.exception(f"Error in create_user: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url.path, 
            method=request.method, 
            response_status=response.status_code
        )


@router.get("/users/{username}", response_model=List[User])
@limiter.limit("20/minute")
async def get_users(
    request: Request, 
    response: Response,
    username: str,
    user_service: UserService = Depends(user_service_admin)
):
    """
    Retrieve users with optional age filtering.
    
    Args:
        is_authenticated (Optional[bool]): Filter by authentication status
        is_deleted (Optional[bool]): Filter by deletion status
    
    Returns:
        List[User]: List of user details
    """
    try:
        user_data = await user_service.get_user(username)
        
        if not user_data:
            logger.error(f"User not found: {user_data}")
            return JSONResponse(
                content={"msg": "User not found"},
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        return JSONResponse(
            content=user_data,
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        logger.exception(f"Error in get_users: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url.path, 
            method=request.method, 
            response_status=response.status_code
        )


@router.put("/users/{username}")
async def update_user(
    request: Request, 
    response: Response,
    username: str, 
    update_data: UserUpdateSchema,
    user_service: UserService = Depends(user_service_admin)
):
    """
    Update an existing user's information.
    
    Args:
        user_id (str): ID of user to update
        update_data (User): New user information
        db (FirebaseDBHelper): Database connection
    
    Returns:
        dict: Update confirmation
    """
    try:
        # Check if user exists
        existing_user = await user_service.get_user(username)
        if not existing_user:
            logger.error(f"User not found: {existing_user}")
            return JSONResponse(
                content={"msg": "User not found"},
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Get only the fields that were provided in the update request
        update_fields = update_data.model_dump(
            exclude_unset=True,
            exclude_none=True
        )
        
        if not update_fields:
            return JSONResponse(
                content={"msg": "No fields to update"},
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Merge update data with existing user data
        merged_data = {**existing_user, **update_fields}
        
        # Create entity from merged data
        user_entity = UserUpdateSchema(**merged_data).to_entity()
        
        # Perform update
        success = await user_service.update_user(
            username=username,
            user_data=user_entity
        )
        
        if not success:
            logger.error(f"User update failed: {success}")
            return JSONResponse(
                content={"msg": "User update failed"},
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return JSONResponse(
            content={"msg": "User updated successfully"},
            status_code=status.HTTP_200_OK
        )
    except ValidationError as ve:
        logger.error(f"Validation error in update_user: {str(ve)}")
        return JSONResponse(
            content={"msg": str(ve)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except Exception as e:
        logger.exception(f"Error in update_user: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url.path, 
            method=request.method, 
            response_status=response.status_code
        )


@router.delete("/users/{username}")
async def delete_user(
    request: Request, 
    response: Response, 
    username: str,
    user_service: UserService = Depends(user_service_admin)
):
    """
    Delete a user from the database.
    
    Args:
        user_id (str): ID of user to delete
        db (FirebaseDBHelper): Database connection
    
    Returns:
        dict: Deletion confirmation
    """
    try:
        # Check if user exists
        existing_user = await user_service.get_user(username)
        if not existing_user:
            logger.error(f"User not found: {existing_user}")
            return JSONResponse(
                content={"msg": "User not found"},
                status_code=status.HTTP_404_NOT_FOUND
            )
            
        # Perform deletion
        success = await user_service.delete_user(username)
        
        if not success:
            logger.error(f"User deletion failed: {success}")
            return JSONResponse(
                content={"msg": "User deletion failed"},
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return JSONResponse(
            content={"msg": "User deleted successfully"},
            status_code=status.HTTP_200_OK
        )
    
    except Exception as e:
        logger.exception(f"Error in delete_user: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url.path, 
            method=request.method, 
            response_status=response.status_code
        )
