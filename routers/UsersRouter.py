from typing import Optional
from pydantic import ValidationError

from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.util import get_remote_address

from lifttrack.auth import check_password_update, get_password_hash, verify_password, validate_input
from lifttrack.models import User
from lifttrack.dbhandler import rtdb
from lifttrack.utils.logging_config import setup_logger

router = APIRouter(
    prefix="/user",
    tags=["user"],
    responses={404: {"description": "Not found"}}
)
logger = setup_logger("user_router", ".log")

limiter = Limiter(key_func=get_remote_address)

@router.put("/create")
@limiter.limit("5/minute")
async def create_user(request: Request, user: User = Depends(validate_input)):
    """
    Endpoint to create a new user in the Firebase Realtime Database.

    Args:
        user: User model that contains user data.
        request: FastAPI Request object.
    """
    try:
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

        return JSONResponse(
            content={"msg": "User created."},
            status_code=status.HTTP_201_CREATED
        )
    except ValidationError as vale:
        # Handle errors raised by Pydantic
        return JSONResponse(
            content={"msg": str(vale)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except ValueError as ve:
        # Handle errors raised by RTDBHelper
        return JSONResponse(
            content={"msg": str(ve)},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    except HTTPException as httpe:
        # Handle FastAPI-specific HTTP exceptions
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in create_user: {e}")
        # Handle unexpected exceptions
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
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
        user_data = rtdb.get_data(username, data)

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
        logger.exception(f"Error in get_user_data: {e}")
        # Handle unexpected exceptions
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
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
        user, is_password_update = update_data
        
        if is_password_update:
            user.password = get_password_hash(user.password)

        snapshot = rtdb.update_data(username, user.model_dump())

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
        logger.exception(f"Error in update_user_data: {e}")
        # Handle unexpected exceptions
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
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
        # Handle unexpected exceptions
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
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
        deleted = rtdb.delete_data(username)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User deletion failed."
            )

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
        # Handle unexpected exceptions
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )