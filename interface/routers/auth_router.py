from fastapi import APIRouter, Depends, status, Response
from fastapi.security import OAuth2PasswordRequestForm

from core.entities.auth_entity import CredentialsEntity, TokenEntity
from core.usecase.auth_usecase import AuthUseCase
from infrastructure.auth.exceptions import (
    InvalidCredentialsError,
    ValidationError,
    InvalidPasswordError
)
from interface.di import get_auth_service, get_current_user


# Create auth router
auth_router = APIRouter(
    prefix="/v2/auth",
    tags=["v2-authentication"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication failed"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"}
    }
)


@auth_router.post("/token", response_model=TokenEntity)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_usecase: AuthUseCase = Depends(get_auth_service)
):
    """
    Authenticate and generate a JWT token.
    
    Uses OAuth2 password flow compatible with standard OAuth clients.
    """
    credentials = CredentialsEntity(
        username=form_data.username,
        password=form_data.password
    )
    
    success, token, user, error = await auth_usecase.login(credentials)
    
    if not success:
        raise InvalidCredentialsError(detail=error or "Invalid username or password")
        
    return token


@auth_router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    token: str = Depends(get_current_user),
    auth_usecase: AuthUseCase = Depends(get_auth_service)
):
    """
    Log out by invalidating the current token.
    
    Returns 204 No Content on success.
    """
    success = await auth_usecase.logout(token)
    
    if not success:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
    return None


@auth_router.post("/refresh", response_model=TokenEntity)
async def refresh_token(
    refresh_token: str,
    auth_usecase: AuthUseCase = Depends(get_auth_service)
):
    """
    Refresh an access token using a refresh token.
    
    Returns a new access token if successful.
    """
    success, token, error = await auth_usecase.refresh_token(refresh_token)
    
    if not success:
        raise InvalidCredentialsError(detail=error or "Invalid refresh token")
        
    return token


@auth_router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    old_password: str,
    new_password: str,
    user = Depends(get_current_user),
    auth_usecase: AuthUseCase = Depends(get_auth_service)
):
    """
    Change the current user's password.
    
    Requires old password verification and validates new password strength.
    """
    success, error = await auth_usecase.change_password(
        user.id, 
        old_password, 
        new_password
    )
    
    if not success:
        if "password" in error.lower() and "requirements" in error.lower():
            raise InvalidPasswordError(detail=error)
        else:
            raise ValidationError(detail=error)
            
    return None 