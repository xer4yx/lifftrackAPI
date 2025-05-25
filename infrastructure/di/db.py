from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from lifttrack import config
from routers.manager import HTTPConnectionPool

from infrastructure.database import FirebaseREST, FirebaseAdmin
from utils import FirebaseSettings

firebase_settings = FirebaseSettings()

options = {
    'databaseURL': config.get(section='Firebase', option='FIREBASE_DEV_DB'),
    'databaseAuthVariableOverride': {
        'uid': config.get(section='Firebase', option='FIREBASE_AUTH_UID'),
        'admin': True
    }
}


async def get_firebase_rest() -> AsyncGenerator[FirebaseREST, None]:
    """
    Dependency for FirebaseREST that properly manages session lifecycle.
    
    This is an async generator that creates a database connection with its own session,
    yields it for use in the endpoint, and then properly cleans up when done.
    """
    async with HTTPConnectionPool.get_session() as session:
        firebase = FirebaseREST(
            dsn=firebase_settings.database_url,
            authentication=firebase_settings.auth_token,
            session=session
        )
        try:
            yield firebase
        finally:
            # Session is automatically closed by the context manager
            pass


async def get_firebase_admin() -> AsyncGenerator[FirebaseAdmin, None]:
    try:
        # Create a proper dictionary from our FirebaseOptions model
        firebase_options: Dict[str, Any] = {}
        
        if firebase_settings.options:
            # Convert the Pydantic model to a dictionary
            firebase_options = firebase_settings.options.model_dump()
        else:
            # Fallback to the legacy hardcoded options
            firebase_options = options
        
        # Ensure databaseURL is set
        if 'databaseURL' not in firebase_options and firebase_settings.database_url:
            firebase_options['databaseURL'] = firebase_settings.database_url
            
        firebase = FirebaseAdmin(
            credentials_path=firebase_settings.admin_sdk,
            options=firebase_options
        )
        yield firebase
    finally:
        pass
    