import firebase_admin
from firebase_admin import credentials, db
from firebase_admin.exceptions import FirebaseError
from fastapi import status
import asyncio
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any, Dict, Optional

from core.interface import NTFInterface
from lifttrack.utils.logging_config import setup_logger

class FirebaseREST(NTFInterface):
    """
    FirebaseREST is a class that provides a REST interface to the Firebase Realtime Database.
    """
    def __init__(self, dsn: str, authentication: Optional[str] = None, session: Optional[Any] = None) -> None:
        """
        Initialize the FirebaseREST class.
        Args:
            dsn: The database URL.
            authentication: The authentication token.
            session: Session object for the database.
        """
        self.logger = setup_logger("firebase-rest", "api_di.log")
        self.dsn = dsn
        self.auth = authentication
        self.__lock = asyncio.Lock()
        self.__session = session
        
    async def create_pool(self):
        """Create the connection pool if it doesn't exist"""
        if self.__session is not None:
            self.logger.info("Instance of db pool called but was already created")
            return
            
        async with self.__lock:
            if self.__session is None:
                conn = aiohttp.TCPConnector(limit=self.__pool_size)
                self.__session = aiohttp.ClientSession(connector=conn)
                self.logger.info("Created instance of db pool")

    async def close_pool(self):
        """Close the connection pool"""
        if self.__session is not None:
            async with self.__lock:
                if self.__session is not None:
                    await self.__session.aclose()
                    self.__session = None
                    self.logger.info("Closed instance of db pool")
                    
    def __aenter__(self):
        return self.create_pool()
    
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_pool()
        
    def sessionmanager(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                async with self.__lock:
                    if self.__session is None:
                        await self.create_pool()
                    return await func(self, *args, **kwargs)
            except Exception as e:
                self.logger.exception(f"Error in sessionmanager: {e}")
        return wrapper
    
    @sessionmanager
    async def get_data(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.dsn}/{key}.json"
            if self.auth:
                url += f"?auth={self.auth}"
                
            response = await self.__session.get(url)
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                self.logger.debug(f"Data retrieved for key: {key}")
                return data if data is not None else {}
            else:
                self.logger.error(f"Failed to get data for key {key}. Reason: {response.content}")
                return {}
        except Exception as e:
            self.logger.exception(f"Error in get_data for key {key}: {e}")
            return {}
    
    @sessionmanager
    async def set_data(self, key: str, value: Any) -> None:
        try:
            url = f"{self.dsn}/{key}.json"
            if self.auth:
                url += f"?auth={self.auth}"
            
            response = await self.__session.put(url, json=value)
            if response.status_code == status.HTTP_200_OK:
                return
            else:
                raise Exception(f"Error setting data: {response.content}")
        except Exception as e:
            self.logger.exception(f"Error in set_data: {e}")
    
    @sessionmanager
    async def delete_data(self, key: str) -> None:
        url = f"{self.dsn}/{key}.json"
        if self.auth:
            url += f"?auth={self.auth}"
        
        response = await self.__session.delete(url)
        if response.status_code == status.HTTP_200_OK:
            return True
        else:
            raise Exception(f"Error deleting data: {response.content}")


class FirebaseAdmin(NTFInterface):
    _logger = setup_logger("firebase-admin", "api_di.log")
    _instance = None
    _lock = threading.Lock()
    
    def __new__(
        cls, 
        credentials_path: Optional[str] = None, 
        options: Optional[Dict[str, Any]] = None):
        if not cls._instance:
            with cls._lock:
                try:
                    credential = credentials.Certificate(credentials_path) if credentials_path else None
                    if not firebase_admin._apps:
                        if not options or 'databaseURL' not in options:
                            raise ValueError("Database URL not provided")
                        firebase_admin.initialize_app(
                            credential=credential, 
                            options=options)
                    
                    cls._instance = super().__new__(cls)
                except Exception as e:
                    cls._logger.error(f"Error initializing Firebase Admin: {e}")
                    raise e
        return cls._instance
    
    def __init__(
        self, 
        credentials_path: Optional[str] = None, 
        options: Optional[Dict[str, Any]] = None):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._executor = ThreadPoolExecutor(
                max_workers=10,
                thread_name_prefix='firebase-admin-executor')
            self._db = db.reference()
            self._initialized = True
    
    async def get_data(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            snapshot = self._db.child(key).get()
            return snapshot
        except FirebaseError as fe:
            self._logger.exception(f"Error in get_data for key {key}: {fe}")
            raise fe
        except Exception as e:
            self._logger.exception(f"Error in get_data for key {key}: {e}")
            raise
        
    async def set_data(self, key: Optional[str] = None, value: Dict[str, Any] = None) -> None:
        try:
            if not key or not value:
                raise ValueError("Key and value not provided in the parameter")
            
            self._db.child(key).set(value=value)
        except FirebaseError as fe:
            self._logger.exception(f"Error in set_data: {fe}")
            raise fe
        except Exception as e:
            self._logger.exception(f"Error in set_data: {e}")
            raise
        
    async def delete_data(self, key: str) -> None:
        try:
            self._db.child(key).delete()
        except FirebaseError as fe:
            self._logger.exception(f"Error in delete_data: {fe}")
            raise fe
        except Exception as e:
            self._logger.exception(f"Error in delete_data: {e}")
            raise
