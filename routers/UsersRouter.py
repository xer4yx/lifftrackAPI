from typing import Optional
from pydantic import ValidationError

from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.util import get_remote_address

from lifttrack import network_logger
from lifttrack.auth import (
    check_password_update, 
    get_password_hash, 
    remove_from_username_cache, 
    add_to_username_cache,
    verify_password, 
    validate_input
)
from lifttrack.models import User
from lifttrack.dbhandler.rest_rtdb import rtdb
from lifttrack.utils.logging_config import setup_logger, log_network_io

router = APIRouter(
    prefix="/user",
    tags=["user"],
    responses={404: {"description": "Not found"}}
)
logger = setup_logger("user_router", "router.log")

limiter = Limiter(key_func=get_remote_address)

@router.put("/create")
@limiter.limit("5/minute")
async def create_user(
    request: Request, 
    user: User = Depends(lambda user: validate_input(user, is_update=False))
    ):
    """
    Endpoint to create a new user in the Firebase Realtime Database.

    Args:
        user: User model that contains user data.
        request: FastAPI Request object.
    """
    try:
        response = None
        user_data = {
            "id": user.id,
            "fname": user.fname,
            "lname": user.lname,
            "username": user.username,
            "phoneNum": user.phoneNum,
            "email": user.email,
            "password": get_password_hash(user.password),
            "pfp": user.pfp,
            "isAuthenticated": user.isAuthenticated,
            "isDeleted": user.isDeleted
        }

        # Attempt to add the user to the database
        rtdb.put_data(user_data)
        add_to_username_cache(user.username)

        reponse = JSONResponse(
            content={"msg": "User created."},
            status_code=status.HTTP_201_CREATED
        )
        return response
    except ValidationError as vale:
        # Handle errors raised by Pydantic
        response = JSONResponse(
            content={"msg": str(vale)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        return response
    except ValueError as ve:
        # Handle errors raised by RTDBHelper
        response = JSONResponse(
            content={"msg": str(ve)},
            status_code=status.HTTP_400_BAD_REQUEST
        )
        return reponse
    except HTTPException as httpe:
        # Handle FastAPI-specific HTTP exceptions
        response = JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
        return response
    except Exception as e:
        # Handle unexpected exceptions
        response = JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        logger.exception(f"Error in create_user: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )


@router.get("/{username}")
@limiter.limit("20/minute")
async def get_user_data(request: Request, username: str, data: Optional[str] = None):
    """
    Endpoint to get user data from the Firebase Realtime Database.

    Args:
        username: Username of the target user.
        data: Specific data to retrieve. Defaults to None to get all user information.
        request: FastAPI Request object.
    """
    try:
        response = None
        user_data = rtdb.get_data(username, data)

        if user_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found."
            )

        response = JSONResponse(
            content=user_data,
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
        # Handle unexpected exceptions
        response = JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        logger.exception(f"Error in get_user_data: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )

@router.put("/user/{username}")
@limiter.limit("10/minute")
async def update_user_data(
    username: str, 
    request: Request,
    update_data: tuple[User, bool] = Depends(check_password_update)
    ):
    """
    Endpoint to update user data in the database.

    Args:
        username: Username of the target user.
        user: User model that contains user data.
        request: FastAPI Request object.
    """
    try:
        response = None
        user, is_password_update = update_data
        
        if is_password_update:
            user.password = get_password_hash(user.password)

        snapshot = rtdb.update_data(username, user.model_dump())

        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User update failed."
            )

        response = JSONResponse(
            content={"msg": "User updated."},
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
        # Handle unexpected exceptions
        response = JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        logger.exception(f"Error in update_user_data: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )
    


@router.put("/user/{username}/change-pass")
@limiter.limit("5/minute")
async def change_password(username: str, user: User, request: Request):
    """
    Endpoint to change user password.

    Args:
        username: Username of the target user.
        user: User model that contains user data.
        request: FastAPI Request object.
    """
    try:
        response = None
        hashed_pass = rtdb.get_data(user.username, "password")

        if not hashed_pass or not verify_password(user.password, hashed_pass):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect password."
            )

        updated_user_data = {
            "id": user.id,
            "fname": user.fname,
            "lname": user.lname,
            "username": user.username,
            "phoneNum": user.phoneNum,
            "email": user.email,
            "password": get_password_hash(user.password),
            "pfp": user.pfp,
            "isAuthenticated": user.isAuthenticated,
            "isDeleted": user.isDeleted
        }

        snapshot = rtdb.update_data(user.username, updated_user_data)
        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password change failed."
            )

        response = JSONResponse(
            content={"msg": "Password changed."},
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
        # Handle unexpected exceptions
        response = JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        logger.exception(f"Error in change_password: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )


@router.delete("/user/{username}")
@limiter.limit("5/minute")
async def delete_user(username: str, request: Request):
    """
    Endpoint to delete a user from the Firebase Realtime Database.

    Args:
        username: Username of the target user.
        request: FastAPI Request object.
    """
    try:
        response = None
        deleted = rtdb.delete_data(username)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User deletion failed."
            )
            
        remove_from_username_cache(username)

        response = JSONResponse(
            content={"msg": "User deleted."},
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
