"""
Use Cases
- Implements application-specific business rules
- Orchestrates the flow of data between entities and external services
- Contains the specific logic of how the system should behave
- Represents the business logic independent of any framework
"""
from typing import Depends

from core.interfaces import (
    DatabaseRepository,
    TokenService,
    PasswordService,
    get_database_repository, 
    get_token_service, 
    get_password_service)

from .user_service import UserService
from .auth_service import AuthService

__all__ = [
    "UserService",
    "AuthService",
]

def get_auth_service(
    database: DatabaseRepository = Depends(get_database_repository),
    token_service: TokenService = Depends(get_token_service),
    password_service: PasswordService = Depends(get_password_service)
) -> AuthService:
    return AuthService(database, token_service, password_service) 