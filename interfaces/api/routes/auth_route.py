from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from core.services.auth_service import AuthService
from core.services.auth_service import get_auth_service
from core.exceptions.auth import AuthenticationError
from interfaces.api.schemas import LoginRequest, LoginResponse, LogoutResponse
from interfaces.api.schemas import APIResponse
from interfaces.api.dependencies import get_current_user
from interfaces.api.constants import (
    MSG_INVALID_CREDENTIALS,
    ERR_AUTH_INVALID,
    MSG_UNAUTHORIZED
)
from utilities.monitoring import MonitoringFactory

logger = MonitoringFactory.get_logger("auth-api")

auth_router = APIRouter(prefix="/auth", tags=["authentication"])

@auth_router.post("/login", response_model=APIResponse[LoginResponse])
async def login(
    credentials: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    try:
        logger.info(f"Login attempt for user: {credentials.username}")
        login_result = await auth_service.authenticate_user(
            credentials.username,
            credentials.password
        )
        
        return APIResponse(
            status_code=status.HTTP_200_OK,
            message="Login successful",
            data=LoginResponse(
                access_token=login_result["access_token"],
                expires_at=login_result["expires_at"],
                user_id=login_result["user_id"]
            )
        )
        
    except AuthenticationError as e:
        logger.warning(f"Login failed for user {credentials.username}: {e}")
        return APIResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            message=MSG_INVALID_CREDENTIALS,
            data={"error_code": ERR_AUTH_INVALID}
        )
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@auth_router.post("/logout", response_model=APIResponse[LogoutResponse])
async def logout(
    current_user: dict = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    try:
        logger.info(f"Logout request for user: {current_user['sub']}")
        await auth_service.logout_user(current_user["sub"])
        
        return APIResponse(
            status_code=status.HTTP_200_OK,
            message="Logout successful",
            data=LogoutResponse()
        )
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# OAuth2 compatible token endpoint
@auth_router.post("/token")
async def login_oauth(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service)
):
    try:
        logger.info(f"OAuth token request for user: {form_data.username}")
        login_result = await auth_service.authenticate_user(
            form_data.username,
            form_data.password
        )
        
        return {
            "access_token": login_result["access_token"],
            "token_type": "bearer"
        }
        
    except AuthenticationError as e:
        logger.warning(f"OAuth token request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=MSG_INVALID_CREDENTIALS,
            headers={"WWW-Authenticate": "Bearer"},
        ) 