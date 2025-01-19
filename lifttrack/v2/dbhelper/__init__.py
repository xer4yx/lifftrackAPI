from os import getenv
from dotenv import load_dotenv

from lifttrack import config
from .admin_rtdb import FirebaseDBHelper
from utilities.monitoring.factory import MonitoringFactory

from fastapi import HTTPException

__all__ = ['FirebaseDBHelper']
load_dotenv('./.env')
options = {
    'databaseURL': config.get(section='Firebase', option='RTDB_DSN'),
    'databaseAuthVariableOverride': {
        'uid': config.get(section='Firebase', option='FIREBASE_AUTH_UID')
    }
}

logger = MonitoringFactory.get_logger(__name__)

def get_db():
    """
    Dependency injection for database connection.
    Ensures a single database instance is used across requests.
    """
    try:
        credentials_path = config.get(section='Firebase', option='GOOGLE_SERVICES_JSON')
        if not credentials_path:
            raise ValueError("GOOGLE_SERVICES_JSON environment variable not set")
            
        # Initialize Firebase with credentials path
        db = FirebaseDBHelper(
            credentials_path=credentials_path,
            options=options
        )
        logger.info(f"Database connection successful: {db}")
        return db
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
