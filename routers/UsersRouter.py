from typing import Optional
import uuid
from pydantic import ValidationError

from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi.params import Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from slowapi import Limiter
from slowapi.util import get_remote_address

from lifttrack import network_logger
from lifttrack.auth import (
    check_password_update, 
    get_password_hash, 
    remove_from_username_cache, 
    add_to_username_cache,
    verify_password
)
from lifttrack.models import User
from lifttrack.dbhandler.rest_rtdb import RTDBHelper
from lifttrack.utils.logging_config import setup_logger, log_network_io
from .manager import HTTPConnectionPool

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
    response: Response, 
    user: User = Body(..., example={
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
        async with HTTPConnectionPool.get_session() as session:
            async with RTDBHelper(session) as rtdb:
                user.password = get_password_hash(user.password)
                await rtdb.put_data(user_data=jsonable_encoder(user))
                add_to_username_cache(user.username)

                return JSONResponse(
                    content={"msg": "User created."},
                    status_code=status.HTTP_201_CREATED
                )
    except ValidationError as vale:
        return JSONResponse(
            content={"msg": str(vale)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except ValueError as ve:
        return JSONResponse(
            content={"msg": str(ve)},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
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
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )


@router.get("/{username}")
@limiter.limit("20/minute")
async def get_user_data(
    request: Request, 
    response: Response,
    username: str, 
    data: Optional[str] = None):
    """
    Endpoint to get user data from the Firebase Realtime Database.

    Args:
        username: Username of the target user.
        data: Specific data to retrieve. Defaults to None to get all user information.
        request: FastAPI Request object.
    """
    try:
        async with HTTPConnectionPool.get_session() as session:
            async with RTDBHelper(session) as rtdb:
                user_data = await rtdb.get_data(username, data)

                if user_data is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="User not found."
                    )

                return JSONResponse(
                    content=user_data,
                    status_code=status.HTTP_200_OK
                )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
        
    except Exception as e:
        # Handle unexpected exceptions
        logger.exception(f"Error in get_user_data: {e}")
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

@router.put("/user/{username}")
@limiter.limit("10/minute")
async def update_user_data(
    username: str, 
    request: Request,
    response: Response,
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
        async with HTTPConnectionPool.get_session() as session:
            async with RTDBHelper(session) as rtdb:
                user, is_password_update = update_data
                
                if is_password_update:
                    user.password = get_password_hash(user.password)

                snapshot = await rtdb.update_data(username, user.model_dump())

                if not snapshot:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="User update failed."
                    )

                return JSONResponse(
                    content={"msg": "User updated."},
                    status_code=status.HTTP_200_OK
                )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        ) 
    except Exception as e:
        # Handle unexpected exceptions
        logger.exception(f"Error in update_user_data: {e}")
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


@router.put("/user/{username}/change-pass")
@limiter.limit("5/minute")
async def change_password(user: User, request: Request, response: Response):
    """
    Endpoint to change user password.

    Args:
        username: Username of the target user.
        user: User model that contains user data.
        request: FastAPI Request object.
    """
    try:
        async with HTTPConnectionPool.get_session() as session:
            async with RTDBHelper(session) as rtdb:
                hashed_pass = await rtdb.get_data(user.username, "password")

                if not hashed_pass or not verify_password(user.password, hashed_pass):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Incorrect password."
                    )
                
                user.password = get_password_hash(user.password)

                snapshot = rtdb.update_data(user.username, jsonable_encoder(user))
                if not snapshot:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Password change failed."
                    )

                return JSONResponse(
                    content={"msg": "Password changed."},
                    status_code=status.HTTP_200_OK
                )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
        
    except Exception as e:
        logger.exception(f"Error in change_password: {e}")
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


@router.delete("/user/{username}")
@limiter.limit("5/minute")
async def delete_user(username: str, request: Request, response: Response):
    """
    Endpoint to delete a user from the Firebase Realtime Database.

    Args:
        username: Username of the target user.
        request: FastAPI Request object.
    """
    try:
        async with HTTPConnectionPool.get_session() as session:
            async with RTDBHelper(session) as rtdb:
                deleted = await rtdb.delete_data(username)
                if not deleted:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="User deletion failed."
                    )
                    
                remove_from_username_cache(username)

                return JSONResponse(
                    content={"msg": "User deleted."},
                    status_code=status.HTTP_200_OK
                )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
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
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )
