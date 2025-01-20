"""
Use Cases
- Implements application-specific business rules
- Orchestrates the flow of data between entities and external services
- Contains the specific logic of how the system should behave
- Represents the business logic independent of any framework
"""

from fastapi import Depends
from core.interfaces import (
    DatabaseRepository,
    TokenService,
    PasswordService,
    InputValidator,
    get_database_repository, 
    get_token_service, 
    get_password_service,
    get_input_validator
)

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

def get_user_service(
    database: DatabaseRepository = Depends(get_database_repository),
    password_service: PasswordService = Depends(get_password_service),
    input_validator: InputValidator = Depends(get_input_validator)
    
    ) -> UserService:
    return UserService(database=database, password_service=password_service, input_validator=input_validator)