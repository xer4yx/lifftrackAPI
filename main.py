import threading
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.encoders import jsonable_encoder

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from lifespan import lifespan

from lifttrack import timedelta, threading, network_logger
from lifttrack.utils.logging_config import log_network_io, setup_logger
from lifttrack.utils.syslogger import log_cpu_and_mem_usage, start_resource_monitoring
from lifttrack.models import User, Token, AppInfo, LoginForm, AppUpdate
from lifttrack.auth import (
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user,
    verify_password,
)

from routers.manager import HTTPConnectionPool
from routers.UsersRouter import router as users_router
from routers.ProgressRouter import router as progress_router
from routers.WebsocketRouter import router as websocket_router
from routers.InferenceRouter import router as inference_router
from routers.v2.UsersRouter import router as v2_users_router
from routers.v2.WebsocketRouter import router as v2_websocket_router

from utils import AppSettings
from utils.cors_settings import CorsSettings

from infrastructure.database import FirebaseREST
from infrastructure.di import get_firebase_rest
from interface.routers import user_router as v3_user_router
from interface.routers import auth_router as v2_auth_router
from interface.ws import websocket_router_v3 as v3_websocket_router

app_settings = AppSettings()
cors_settings = CorsSettings()

app = FastAPI(title=app_settings.name, version=app_settings.version, lifespan=lifespan)

# v1 API Routers
app.include_router(users_router)
app.include_router(progress_router)
app.include_router(websocket_router)
app.include_router(inference_router)
app.include_router(v2_users_router)
app.include_router(v2_websocket_router)
app.include_router(v3_user_router)
app.include_router(v2_auth_router)
app.include_router(v3_websocket_router)

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Modify CORS middleware to include additional security headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_settings.allowed_origins,
    allow_methods=cors_settings.allowed_methods,
    allow_headers=cors_settings.allowed_headers,
    allow_credentials=cors_settings.allow_credentials,
    expose_headers=cors_settings.expose_headers,
)
# Add SlowAPI middleware
app.add_middleware(SlowAPIMiddleware)
latest_frame_lock = threading.Lock()
latest_frame = None

# Initialize loggers
# Configure logging for main.py
logger = setup_logger("main", "lifttrack_main.log")
system_logger = setup_logger("system", "server_resource.log")

# Start resource monitoring
start_resource_monitoring(system_logger, log_cpu_and_mem_usage, 60)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(path="./favicon.ico", media_type="image/x-icon")


# API Endpoint [ROOT]
@app.get("/")
@limiter.limit("10/minute")  # Apply specific rate limit
async def read_root(request: Request, response: Response):
    """Lifttrack API root endpoint."""
    try:
        return JSONResponse(
            content={"msg": "Welcome to LiftTrack!"}, status_code=status.HTTP_200_OK
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail}, status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in read_root: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        log_network_io(
            logger=network_logger,
            endpoint=request.url,
            method=request.method,
            response_status=response.status_code,
        )


# API Endpoint [About App]
@app.get("/app-info")
@limiter.limit("20/minute")
async def get_app_info(request: Request, response: Response, appinfo: AppInfo):
    """Endpoint to get information about the app."""
    try:
        return JSONResponse(
            content=jsonable_encoder(appinfo), status_code=status.HTTP_200_OK
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail}, status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in get_app_info: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        log_network_io(
            logger=network_logger,
            endpoint=request.url,
            method=request.method,
            response_status=response.status_code,
        )


@app.get("/app-update")
@limiter.limit("20/minute")
async def get_app_update(
    request: Request,
    response: Response,
    current_version: str = Query(...),
    db: FirebaseREST = Depends(get_firebase_rest),
):
    """Endpoint to check for app updates and get login messages."""
    try:
        # Get the latest version and messages from Firebase
        app_config = await db.get_data(key=f"app_config")

        # Create AppUpdate response
        app_update = AppUpdate(
            current_version=current_version,
            latest_version=app_config.get("latest_version", current_version),
            update_available=app_config.get("latest_version", current_version)
            != current_version,
            update_message=app_config.get("update_message", ""),
            download_url=app_config.get("download_url", ""),
            login_message=app_config.get("login_message", ""),
        )

        return JSONResponse(
            content=jsonable_encoder(app_update), status_code=status.HTTP_200_OK
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail}, status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in get_app_update: {e}")
        # Return default information if there's an error
        app_update = AppUpdate(
            current_version=current_version,
            latest_version=current_version,
            update_available=False,
            update_message="",
            download_url="",
            login_message="",
        )
        return JSONResponse(
            content=jsonable_encoder(app_update), status_code=status.HTTP_200_OK
        )
    finally:
        log_network_io(
            logger=network_logger,
            endpoint=request.url,
            method=request.method,
            response_status=response.status_code,
        )


# API Endpoint [Authentication Operations]
@app.post("/login")
@limiter.limit("3/minute")  # Limit login attempts
async def login(
    login_form: LoginForm,
    request: Request,
    response: Response,
    db: FirebaseREST = Depends(get_firebase_rest),
):
    """
    API endpoint for user login.

    Args:
        login_form: BaseModel that contains username and password.
        request: FastAPI Request object.
    """
    try:
        user_data = await db.get_data(key=f"users/{login_form.username}")

        if user_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        if not verify_password(login_form.password, user_data.get("password", "")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        return JSONResponse(
            content={"message": "Login successful", "success": True},
            status_code=status.HTTP_200_OK,
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail}, status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in login: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        log_network_io(
            logger=network_logger,
            endpoint=request.url,
            method=request.method,
            response_status=response.status_code,
        )


@app.post("/token", response_model=Token)
@limiter.limit("10/minute")  # Limit token requests
async def login_for_access_token(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: FirebaseREST = Depends(get_firebase_rest),
):
    """
    Endpoint to get an access token.

    Args:
        form_data: OAuth2PasswordRequestForm that contains username and password.
        request: FastAPI Request object.
    """
    try:
        user = await db.get_data(key=f"users/{form_data.username}")

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        if not verify_password(form_data.password, user.get("password", "")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        return JSONResponse(
            content={
                "access_token": access_token,
                "token_type": "bearer",
                "message": "Login successful",
                "success": True,
            },
            status_code=status.HTTP_200_OK,
        )
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail}, status_code=httpe.status_code
        )
    except Exception as e:
        logger.exception(f"Error in login_for_access_token: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        log_network_io(
            logger=network_logger,
            endpoint=request.url,
            method=request.method,
            response_status=response.status_code,
        )


@app.post("/logout")
@limiter.limit("10/minute")
async def logout(request: Request, response: Response):
    """
    Endpoint to logout user and invalidate their token.
    """
    try:
        # TODO: Add token revocation
        # You might want to add the token to a blacklist here if implementing token revocation
        return JSONResponse(
            content={"msg": "Successfully logged out"}, status_code=status.HTTP_200_OK
        )
    except Exception as e:
        logger.exception(f"Error in logout: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        log_network_io(
            logger=network_logger,
            endpoint=request.url,
            method=request.method,
            response_status=response.status_code,
        )


# API Endpoint [RTDB Operations]
@app.get("/users/me/")
@limiter.limit("30/minute")
async def read_users_me(
    request: Request, response: Response, current_user: User = Depends(get_current_user)
):
    """
    Endpoint to get the current user.

    Args:
        current_user: User model that contains user data.
        request: FastAPI Request object.
    """
    try:
        return JSONResponse(
            content=current_user.model_dump(), status_code=status.HTTP_200_OK
        )
    except Exception as e:
        logger.exception(f"Error in read_users_me: {e}")
        return JSONResponse(
            content={"msg": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        log_network_io(
            logger=network_logger,
            endpoint=request.url,
            method=request.method,
            response_status=response.status_code,
        )


@app.get("/ping")
@limiter.limit("10/minute")
async def ping(request: Request, response: Response):
    """Endpoint to check if the server is alive."""
    try:
        return JSONResponse(content={"status": "ok"}, status_code=status.HTTP_200_OK)
    except HTTPException as httpe:
        return JSONResponse(
            content={"msg": httpe.detail}, status_code=httpe.status_code
        )
    finally:
        log_network_io(
            logger=network_logger,
            endpoint=request.url,
            method=request.method,
            response_status=response.status_code,
        )
