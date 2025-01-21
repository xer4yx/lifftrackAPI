"""
Infrastructure Layer
- Purpose: Provide concrete implementations of external concerns and integrations
- Key Directories:
    - database
    - ml_models
    - extrernal_services
"""
from typing import Annotated
from .database import DatabaseFactory
from .auth import get_password_service, get_data_validator
from core.interfaces import DatabaseRepository
from core.services import get_user_service, UserService
from utilities.config import get_database_settings

db_settings = get_database_settings()

def get_admin_firebase_db() -> DatabaseRepository:
    admin_db = DatabaseFactory.create_repository(
        type="admin",
        credentials_path=db_settings.get("FIREBASE_CREDENTIALS_PATH"),
        auth_token=db_settings.get("FIREBASE_AUTH_TOKEN"),
        options=db_settings.get("FIREBASE_OPTIONS")
    )
    return admin_db

def get_rest_firebase_db() -> DatabaseRepository:
    rest_db = DatabaseFactory.create_repository(
        type="rest",
        dsn=db_settings.get("FIREBASE_DSN"),
        auth_token=db_settings.get("FIREBASE_AUTH_TOKEN")
    )
    return rest_db

# Import auth services after database functions are defined

user_service_rtdb: UserService = Annotated[
    get_user_service, 
    get_rest_firebase_db, 
    get_password_service, 
    get_data_validator
]

user_service_admin: UserService = Annotated[
    get_user_service, 
    get_admin_firebase_db, 
    get_password_service, 
    get_data_validator
]

__all__ = [
    "get_admin_firebase_db",
    "get_rest_firebase_db",
    "user_service_rtdb",
    "user_service_admin"
]