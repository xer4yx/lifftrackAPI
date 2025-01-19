from typing import Optional, Dict, Any
from core.interfaces import DatabaseRepository
from .firebase import FirebaseAdminRepository, FirebaseRestRepository

class DatabaseFactory:
    @staticmethod
    def create_repository(
        type: str = "admin",
        credentials_path: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        dsn: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> DatabaseRepository:
        if type == "admin":
            return FirebaseAdminRepository(credentials_path, options)
        elif type == "rest":
            if not dsn:
                raise ValueError("DSN is required for REST repository")
            return FirebaseRestRepository(dsn, auth_token)
        else:
            raise ValueError(f"Unknown repository type: {type}")