from typing import Optional

from lifttrack import timedelta, threading, asyncio, base64
from lifttrack.dbhandler.rtdbHelper import rtdb
from lifttrack.models import User, Token, AppInfo, LoginForm
from lifttrack.comvis import cv2, websocket_process_frames
from lifttrack.auth import (
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user,
    verify_password,
    get_password_hash,
    validate_input
)

from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocketDisconnect
from fastapi.security import OAuth2PasswordRequestForm

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from routers.WebsocketRoutes import router as websocket_router

import logging

from lifttrack.utils.logging_config import setup_logger

# Configure logging for main.py
logger = setup_logger("main", "lifttrack_main.log")

# Initialize FastAPI app
app = FastAPI()

# Include the websocket router
app.include_router(websocket_router)

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add SlowAPI middleware
app.add_middleware(SlowAPIMiddleware)

# CORS Configuration
server_origin = [
    'http://localhost:8000',
]

server_method = ["PUT", "GET", "DELETE"]

server_header = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=server_origin,
    allow_methods=server_method,
    allow_headers=server_header
)

latest_frame_lock = threading.Lock()
latest_frame = None


# API Endpoint [ROOT]
@app.get("/")
@limiter.limit("10/minute")  # Apply specific rate limit
async def read_root(request: Request):
    """Lifttrack API root endpoint."""
    try:
        return JSONResponse(
            content={"msg": "Welcome to LiftTrack!"},
            status_code=status.HTTP_200_OK
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in read_root: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# API Endpoint [About App]
@app.get("/app-info")
@limiter.limit("20/minute")
async def get_app_info(request: Request):
    """Endpoint to get information about the app."""
    try:
        # Construct the AppInfo object here or retrieve it from a source
        appinfo = AppInfo()
        
        return JSONResponse(
            content=appinfo,
            status_code=status.HTTP_200_OK
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in get_app_info: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# API Endpoint [Authentication Operations]
@app.post("/login")
@limiter.limit("10/minute")  # Limit login attempts
async def login(login_form: LoginForm, request: Request):
    """
    API endpoint for user login.

    Args:
        login_form: BaseModel that contains username and password.
        request: FastAPI Request object.
    """
    try:
        user_data = rtdb.get_data(login_form.username)

        if user_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        if not verify_password(login_form.password, user_data.get("password", "")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        return JSONResponse(
            content={"message": "Login successful", "success": True},
            status_code=status.HTTP_200_OK
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in login: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.post("/token", response_model=Token)
@limiter.limit("10/minute")  # Limit token requests
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Endpoint to get an access token.

    Args:
        form_data: OAuth2PasswordRequestForm that contains username and password.
        request: FastAPI Request object.
    """
    try:
        user = rtdb.get_data(form_data.username)

        if user is None or not verify_password(form_data.password, user.get("password", "")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        return JSONResponse(
            content={"access_token": access_token, "token_type": "bearer"},
            status_code=status.HTTP_200_OK
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in login_for_access_token: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.post("/logout")
@limiter.limit("10/minute")
async def logout(request: Request, current_user: User = Depends(get_current_user)):
    """
    Endpoint to logout user and invalidate their token.
    """
    try:
        # You might want to add the token to a blacklist here if implementing token revocation
        return JSONResponse(
            content={"msg": "Successfully logged out"},
            status_code=status.HTTP_200_OK
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in logout: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# API Endpoint [RTDB Operations]
@app.get("/users/me/")
@limiter.limit("30/minute")
async def read_users_me(request: Request, current_user: User = Depends(get_current_user)):
    """
    Endpoint to get the current user.

    Args:
        current_user: User model that contains user data.
        request: FastAPI Request object.
    """
    try:
        # Convert User model to dictionary
        user_dict = {
            "id": current_user.id,
            "fname": current_user.fname,
            "lname": current_user.lname,
            "username": current_user.username,
            "phoneNum": current_user.phoneNum,
            "email": current_user.email,
            "password": current_user.password,
            "pfp": current_user.pfp,
            "isAuthenticated": current_user.isAuthenticated,
            "isDeleted": current_user.isDeleted
        }
        return JSONResponse(
            content=user_dict,
            status_code=status.HTTP_200_OK
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail},
            status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in read_users_me: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.put("/user/create")
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


@app.get("/user/{username}")
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


@app.put("/user/{username}")
@limiter.limit("10/minute")
async def update_user_data(username: str, user: User, request: Request):
    """
    Endpoint to update user data in the database.

    Args:
        username: Username of the target user.
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

        snapshot = rtdb.update_data(username, user_data)

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


@app.put("/user/{username}/change-pass")
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


@app.delete("/user/{username}")
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


# API Endpoint [Frame Operations]
@app.websocket("/ws-tracking")  # Mobile version
async def websocket_inference(websocket: WebSocket):
    """
    Endpoint for the WebSocket video feed. The server receives frames from the client that's formatted as a base64
    bytes. The server decodes the frame in a thread pool, processes the frame to get inference and annotations, and
    encodes it back to the client as bytes back.

    Args:
        websocket: WebSocket object.
    """
    await websocket.accept()
    connection_open = True
    while connection_open:
        try:
            # Expected format from client side is:
            # {"type": "websocket.receive", "text": data}
            frame_data = await websocket.receive()

            if frame_data["type"] == "websocket.close":
                connection_open = False
                break

            # frame_byte = base64.b64decode(frame_data["bytes"])
            frame_byte = frame_data.get("bytes")

            if not isinstance(frame_byte, bytes):
                await websocket.close()
                break

            # Process frame in thread pool to avoid blocking
            (annotated_frame, features) = await asyncio.get_event_loop().run_in_executor(
                None, websocket_process_frames, frame_byte
            )

            # Encode and send result
            encoded, buffer = cv2.imencode(".jpeg", annotated_frame)

            if not encoded:
                raise WebSocketDisconnect

            return_bytes = base64.b64decode(buffer.tobytes())

            # Expected return to the client side is:
            # {"type": "websocket.send", "bytes": data}
            await websocket.send_bytes(return_bytes)
        except WebSocketDisconnect:
            connection_open = False
        except Exception as e:
            logger.exception(f"WebSocket error: {e}")
            connection_open = False
        finally:
            if not connection_open:
                await websocket.close()


# @app.get("/stream-tracking")
# async def video_feed():  # Web version
#     """
#     Endpoint for the video feed.
#     """
#     return StreamingResponse(
#         generate_frames(),
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )
