from fastapi import Depends
from typing import AsyncGenerator

from infrastructure.repositories import UserRepository
from infrastructure.di.db import get_firebase_rest
from infrastructure.database import FirebaseREST

async def get_user_repository(firebase: FirebaseREST = Depends(get_firebase_rest)) -> UserRepository:
    """
    Dependency for injecting a UserRepository.
    
    Args:
        firebase: An instance of FirebaseREST.
        
    Returns:
        An instance of UserRepository.
    """
    return UserRepository(firebase=firebase) 