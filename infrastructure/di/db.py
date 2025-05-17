from typing import AsyncGenerator
from infrastructure.database import FirebaseREST, FirebaseAdmin
from lifttrack import config
from routers.manager import HTTPConnectionPool

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
            dsn=config.get(section='Firebase', option='FIREBASE_DEV_DB'),
            authentication=config.get(section='Firebase', option='RTDB_AUTH'),
            session=session
        )
        try:
            yield firebase
        finally:
            # Session is automatically closed by the context manager
            pass


async def get_firebase_admin() -> FirebaseAdmin:
    firebase = FirebaseAdmin(
        credentials_path=config.get(section='Firebase', option='ADMIN_SDK_DEV'),
        options=options
    )
    return firebase
    