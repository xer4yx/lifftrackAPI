import threading

from lifttrack import timedelta, threading
from lifttrack.utils.logging_config import log_network_io, setup_logger
from lifttrack.dbhandler.rtdbHelper import rtdb
from lifttrack.models import User, Token, AppInfo, LoginForm
from lifttrack.auth import (
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user,
    verify_password
)


from fastapi import (
    FastAPI, 
    Depends, 
    HTTPException, 
    status,
    Request
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from routers.WebsocketRouter import router as websocket_router
from routers.ProgressRouter import router as progress_router
from lifttrack.utils.syslogger import log_cpu_and_mem_usage, start_resource_monitoring

# Initialize FastAPI app
app = FastAPI()

# Include the websocket router
app.include_router(progress_router)
app.include_router(websocket_router)

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Configuration
server_origin = ["*"]
server_method = ["PUT", "GET", "DELETE"]
server_header = ["*"]

# Modify CORS middleware to include additional security headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_origin,
    allow_methods=server_method,
    allow_headers=server_header,
    allow_credentials=True,  # Added for secure cookie handling
    expose_headers=["*"]
)
# Add SlowAPI middleware
app.add_middleware(SlowAPIMiddleware)
latest_frame_lock = threading.Lock()
latest_frame = None

# Initialize loggers
# Configure logging for main.py
logger = setup_logger("main", "lifttrack_main.log")
system_logger = setup_logger("system", "server_resource.log")
network_logger = setup_logger("network", "network.log")

# Start resource monitoring
start_resource_monitoring(system_logger, log_cpu_and_mem_usage, 60)

# API Endpoint [ROOT]
@app.get("/")
@limiter.limit("10/minute")  # Apply specific rate limit
async def read_root(request: Request):
    """Lifttrack API root endpoint."""
    try:
        response = JSONResponse(
            content={"msg": "Welcome to LiftTrack!"},
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
        logger.exception(f"Error in read_root: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )


# API Endpoint [About App]
@app.get("/app-info")
@limiter.limit("20/minute")
async def get_app_info(request: Request):
    """Endpoint to get information about the app."""
    try:
        # Construct the AppInfo object here or retrieve it from a source
        appinfo = AppInfo()
        
        response = JSONResponse(
            content=appinfo,
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
        logger.exception(f"Error in get_app_info: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
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
        response = None
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

        response = JSONResponse(
            content={"message": "Login successful", "success": True},
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
        logger.exception(f"Error in login: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
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
        response = None
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
        response = JSONResponse(
            content={"access_token": access_token, "token_type": "bearer"},
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
        logger.exception(f"Error in login_for_access_token: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )


@app.post("/logout")
@limiter.limit("10/minute")
async def logout(request: Request, current_user: User = Depends(get_current_user)):
    """
    Endpoint to logout user and invalidate their token.
    """
    try:
        response = None
        # You might want to add the token to a blacklist here if implementing token revocation
        response = JSONResponse(
            content={"msg": "Successfully logged out"},
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
        logger.exception(f"Error in logout: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
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
        response = None
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
        response = JSONResponse(
            content=user_dict,
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
        logger.exception(f"Error in read_users_me: {e}")
        return response
    finally:
        log_network_io(
            logger=network_logger, 
            endpoint=request.url, 
            method=request.method, 
            response_status=response.status_code
        )


# @app.get("/stream-tracking")
# async def video_feed():  # Web version
#     """
#     Endpoint for the video feed.
#     """
#     return StreamingResponse(
#         generate_frames(),
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )
