from os import getenv
from dotenv import load_dotenv

from lifttrack import config
from .admin_rtdb import FirebaseDBHelper

from fastapi import HTTPException

__all__ = ['FirebaseDBHelper']
load_dotenv('./.env')
options = {
    'databaseURL': config.get(section='Firebase', option='RTDB_DSN'),
    'databaseAuthVariableOverride': {
        'uid': config.get(section='Firebase', option='FIREBASE_AUTH_UID')
    }
}


def get_db():
    """
    Dependency injection for database connection.
    Ensures a single database instance is used across requests.
    """
    try:
        # Initialize Firebase with credentials path
        db = FirebaseDBHelper(
            credentials_path=getenv('GOOGLE_SERVICES_JSON'),
            options=options
        )
        return db
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
