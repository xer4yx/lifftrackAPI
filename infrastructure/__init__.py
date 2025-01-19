"""
Infrastructure Layer
- Purpose: Provide concrete implementations of external concerns and integrations
- Key Directories:
    - database
    - ml_models
    - extrernal_services
"""

from .database import DatabaseFactory
from utilities.config import get_database_settings

db_settings = get_database_settings()

def get_admin_firebase_db() -> DatabaseFactory:
    admin_db = DatabaseFactory.create_repository(
        type="admin",
        credentials_path=db_settings.get("FIREBASE_CREDENTIALS_PATH"),
        auth_token=db_settings.get("FIREBASE_AUTH_TOKEN"),
        options=db_settings.get("FIREBASE_OPTIONS")
    )
    return admin_db

def get_rest_firebase_db() -> DatabaseFactory:
    rest_db = DatabaseFactory.create_repository(
        type="rest",
        dsn=db_settings.get("FIREBASE_DSN"),
        auth_token=db_settings.get("FIREBASE_AUTH_TOKEN")
    )
    return rest_db