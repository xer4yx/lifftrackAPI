from fastapi import Depends
from core.usecase import UserUseCase
from core.interface import NTFInterface
from infrastructure.di import get_firebase_admin, get_firebase_rest, get_authenticator
from core.interface import AuthenticationInterface

async def get_user_service_admin(
    db: NTFInterface = Depends(get_firebase_admin),
    auth_service: AuthenticationInterface = Depends(get_authenticator)
) -> UserUseCase:
    """
    Get a UserUseCase instance for the Admin database.
    
    Args:
        db: The database interface
        auth_service: The authentication service
        
    Returns:
        UserUseCase: A UserUseCase instance
        
    Raises:
        ValueError: If the database type is invalid
    """

    try:
        user_service = UserUseCase(
            auth_service=auth_service,
            database_service=db
        )
        return user_service
    except Exception as e:
        raise e


async def get_user_service_rest(
    db: NTFInterface = Depends(get_firebase_rest),
    auth_service: AuthenticationInterface = Depends(get_authenticator)
) -> UserUseCase:
    """
    Get a UserUseCase instance for the REST database.
    
    Args:
        db: The database interface
        auth_service: The authentication service
        
    Returns:
        UserUseCase: A UserUseCase instance
        
    Raises:
        ValueError: If the database type is invalid
    """
    try:
        user_service = UserUseCase(
            auth_service=auth_service,
            database_service=db
        )
        return user_service
    except Exception as e:
        raise e
    

