from functools import lru_cache
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Any, Dict
import configparser

from lifttrack import config
from core.interface import TokenBlacklistRepository, AuthenticationInterface
from infrastructure.auth import InMemoryTokenBlacklistRepository, Authenticator
from infrastructure.repositories import UserRepository
from infrastructure.di.repositories import get_user_repository

@lru_cache
def get_auth_config() -> Dict[str, Any]:
    """
    Get authentication configuration from application config.
    Uses caching for better performance.
    Provides default values if configuration is missing.
    """
    try:
        return {
            'secret_key': config.get(section='Authentication', option='SECRET_KEY', fallback='default_secret_key_for_development_only'),
            'algorithm': config.get(section='Authentication', option='ALGORITHM', fallback='HS256'),
            'access_token_expire_minutes': int(config.get(section='Authentication', option='TTL', fallback='30'))
        }
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        # Log error and return default values for development environment
        import logging
        logging.error(f"Authentication config error: {str(e)}. Using default values.")
        return {
            'secret_key': 'default_secret_key_for_development_only',
            'algorithm': 'HS256',
            'access_token_expire_minutes': 30
        }


# Create a single instance for the application
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_oauth2_scheme() -> OAuth2PasswordBearer:
    """
    Get the OAuth2 password bearer scheme for token authentication.
    """
    return oauth2_scheme  # Return the global instance


def get_token_blacklist_repository() -> TokenBlacklistRepository:
    """
    Get the token blacklist repository implementation.
    
    For now, using the in-memory implementation.
    In production, this could be switched to a Redis or database implementation.
    """
    return InMemoryTokenBlacklistRepository()


async def get_current_user_token(token: str = Depends(oauth2_scheme)) -> str:
    """
    Get the current user's token from the request.
    
    Relies on FastAPI's dependency injection system to extract the token from the
    Authorization header. FastAPI will raise appropriate HTTP exceptions if the
    token is missing or invalid.
    
    Returns:
        The JWT token string.
    """
    return token


def get_authenticator(
    user_repository: UserRepository = Depends(get_user_repository),
    token_blacklist_repository: TokenBlacklistRepository = Depends(get_token_blacklist_repository),
) -> AuthenticationInterface:
    """
    Dependency for injecting an Authenticator.
    
    Args:
        user_repository: An instance of UserRepository.
        token_blacklist_repository: An instance of TokenBlacklistRepository.
        config: Authentication configuration.
        
    Returns:
        An instance of Authenticator.
    """
    # Create authenticator with dependencies
    authenticator = Authenticator(
        user_repository=user_repository,
        token_blacklist_repository=token_blacklist_repository,
        secret_key=config.get(section='Authentication', option='SECRET_KEY', fallback='default_secret_key_for_development_only'),
        algorithm=config.get(section='Authentication', option='ALGORITHM', fallback='HS256'),
        access_token_expire_minutes=config.get(section='Authentication', option='TTL', fallback='30'),
    )
    
    # Initialize the username cache
    # import asyncio
    # try:
    #     # We need to run this async function in a way that's compatible with FastAPI's dependency injection
    #     loop = asyncio.get_event_loop()
    #     if loop.is_running():
    #         # If we're in an async context already, we can create a task
    #         asyncio.create_task(authenticator.initialize_username_cache())
    #     else:
    #         # Otherwise, run the coroutine directly
    #         loop.run_until_complete(authenticator.initialize_username_cache())
    # except Exception as e:
    #     import logging
    #     logging.error(f"Failed to initialize username cache: {e}")
    
    return authenticator
