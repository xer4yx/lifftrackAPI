import uuid
from pydantic import ValidationError

from fastapi import APIRouter, Request, Response, status, Body
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.util import get_remote_address

from lifttrack import network_logger
from lifttrack.auth import LiftTrackAuthenticator
from lifttrack.models import User
from lifttrack.utils.logging_config import setup_logger, log_network_io

from core.services import UserService
from core.exceptions import DatabaseError
from infrastructure import get_rest_firebase_db
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

router = APIRouter(
    prefix="/user",
    tags=["user"],
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

logger = setup_logger("user_router", "router.log")
limiter = Limiter(key_func=get_remote_address)
authentication_service = LiftTrackAuthenticator(get_rest_firebase_db())
user_service = UserService(
    database=get_rest_firebase_db(),
    password_service=authentication_service, 
    input_validator=authentication_service
)

@router.put("/create", status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def create_user(
    request: Request, 
    response: Response, 
    user: UserCreateSchema = Body(..., example={
                "first_name": "John",
                "last_name": "Doe",
                "username": "johndoe",
                "email": "john.doe@example.com",
                "phone_number": "09123456789",
                "password": str(uuid.uuid4().hex)
            }
        )
    ):
    """
    Endpoint to create a new user in the Firebase Realtime Database.

    Args:
        user: User model that contains user data.
        request: FastAPI Request object.
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
            return JSONResponse(
                content={"msg": "User creation failed."},
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return JSONResponse(
            content={"msg": "User created"},
            status_code=status.HTTP_201_CREATED
        )
        
    except ValidationError as vale:
        return JSONResponse(
            content={"msg": str(vale)},
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


@router.get("/{username}", status_code=status.HTTP_200_OK)
@limiter.limit("20/minute")
async def get_user_data(request: Request, username: str, response: Response):
    """
    Endpoint to get user data from the Firebase Realtime Database.

    Args:
        username: Username of the target user.
        data: Specific data to retrieve. Defaults to None to get all user information.
        request: FastAPI Request object.
    """
    try:
        user_data = await user_service.get_user(username)

        if user_data is None:
            return JSONResponse(
                content={"msg": "User not found."},
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        return JSONResponse(
            content=user_data,
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        logger.exception(f"Error in get_user_data: {e}")
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

@router.put("/{username}", status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
async def update_user_data(
    username: str, 
    request: Request, 
    response: Response,
    user: UserUpdateSchema = Body(..., example={
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "phone_number": "09123456789",
    })
):
    """
    Endpoint to update user data in the database.

    Args:
        username: Username of the target user.
        user: User model containing fields to update.
        request: FastAPI Request object.
    
    Returns:
        JSONResponse with success/error message
    """
    try:
        # Convert Pydantic model to entity and update
        success = await user_service.update_user(
            username=username,
            user_data=user.to_entity()
        )

        if not success:
            return JSONResponse(
                content={"msg": "User update failed"},
                status_code=status.HTTP_400_BAD_REQUEST
            )

        return JSONResponse(
            content={"msg": "User updated successfully"},
            status_code=status.HTTP_200_OK
        )

    except ValidationError as ve:
        return JSONResponse(
            content={"msg": str(ve)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except DatabaseError as db_error:
        return JSONResponse(
            content={"msg": str(db_error)},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.exception(f"Error in update_user_data: {e}")
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
    


@router.put("/{username}/change-pass", status_code=status.HTTP_200_OK, deprecated=True)
@limiter.limit("5/minute")
async def change_password(username: str, user: User, request: Request, response: Response):
    """
    Endpoint to change user password.

    Args:
        username: Username of the target user.
        user: User model that contains user data.
        request: FastAPI Request object.
    """
    try:
        hashed_pass = await user_service.get_user_password(username)
        current_password = user.password  # Store current password
        
        if not hashed_pass or not await user_service.verify_password(current_password, hashed_pass):
            return JSONResponse(
                content={"msg": "Incorrect password."},
                status_code=status.HTTP_400_BAD_REQUEST
            )

        # Update user with new password
        user.password = await user_service.hash_password(user.password)  # Use password field for new password
        success = await user_service.update_user(username, user)
        
        if not success:
            return JSONResponse(
                content={"msg": "Password change failed."},
                status_code=status.HTTP_400_BAD_REQUEST
            )

        response = JSONResponse(
            content={"msg": "Password changed."},
            status_code=status.HTTP_200_OK
        )
        return response
    except DatabaseError as db_error:
        response = JSONResponse(
            content={"msg": str(db_error)},
            status_code=status.HTTP_400_BAD_REQUEST
        )
        return response
    except Exception as e:
        logger.exception(f"Error in change_password: {e}")
        response = JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )


@router.delete("/{username}", status_code=status.HTTP_200_OK)
@limiter.limit("5/minute")
async def delete_user(username: str, request: Request, response: Response):
    """
    Endpoint to delete a user from the Firebase Realtime Database.

    Args:
        username: Username of the target user.
        request: FastAPI Request object.
    """
    try:
        # deleted = rtdb.delete_data(username)
        
        success = await user_service.delete_user(username)
        
        if not success:
            raise DatabaseError("User deletion failed.")
            
        # remove_from_username_cache(username)

        response = JSONResponse(
            content={"msg": "User deleted."},
            status_code=status.HTTP_200_OK
        )
        return response
    except DatabaseError as db_error:
        response = JSONResponse(
            content={"msg": str(db_error)},
            status_code=status.HTTP_400_BAD_REQUEST
        )
        return response
    except Exception as e:
        # Handle unexpected exceptions
        response = JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        logger.exception(f"Error in delete_user: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )
