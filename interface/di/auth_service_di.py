from fastapi import Depends, HTTPException, status
from typing import Any, Dict

from core.entities import UserEntity
from core.interface import AuthenticationInterface
from core.usecase import AuthUseCase

from infrastructure.auth.exceptions import (
    InvalidCredentialsError,
    TokenInvalidError,
    UserNotFoundError,
)
from infrastructure.di import get_current_user_token, get_authenticator


def get_auth_service(
    auth_service: AuthenticationInterface = Depends(get_authenticator)
) -> AuthUseCase:
    """
    Get the authentication use case with its service dependency.
    """
    return AuthUseCase(auth_service=auth_service)

async def get_current_user(
    token: str = Depends(get_current_user_token),
    auth_usecase: AuthUseCase = Depends(get_auth_service)
) -> UserEntity:
    """
    Get the current authenticated user.
    
    This dependency validates the token and returns the associated user,
    or raises an appropriate exception if authentication fails.
    
    Args:
        token: The JWT token extracted from the request's Authorization header
        auth_usecase: The authentication use case
        
    Returns:
        The authenticated user entity
        
    Raises:
        HTTPException: If token validation fails or user is not found
    """
    try:
        valid, user, error = await auth_usecase.validate_token(token)
        
        if not valid:
            if "token" in error.lower():
                raise TokenInvalidError(detail=error)
            elif "user" in error.lower():
                raise UserNotFoundError(detail=error)
            else:
                raise InvalidCredentialsError(detail=error or "Invalid authentication credentials")
            
        return user
    except Exception as e:
        # Catch any unexpected errors and provide a clear message
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) 