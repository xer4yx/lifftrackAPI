import asyncio
import threading
from firebase import firebase
from functools import partial
from typing import Dict, Any, Optional, List

from core.interfaces import DatabaseRepository
from core.exceptions import QueryError
from utilities.monitoring import MonitoringFactory

logger = MonitoringFactory.get_logger("firebase-rest-repository")

class FirebaseRestRepository(DatabaseRepository):
    def __init__(self, dsn: str, auth_token: Optional[str] = None):
        self.__dsn = dsn
        self.__auth = auth_token
        self.__connections = {}
        
    def _get_connection(self):
        """Get or create a database connection for the current thread."""
        try:
            thread_id = threading.get_ident()
        
            if thread_id not in self.__connections:
                self.__connections[thread_id] = firebase.FirebaseApplication(
                    dsn=self.__dsn,
                    authentication=(lambda: None, lambda: self.__auth)[self.__auth != 'None']()
                )
                logger.debug(f"Created new connection for thread {thread_id}")
            
            return self.__connections[thread_id]
        except Exception as e:
            logger.error(f"Error in _get_connection: {e}")
            raise

    async def _run_in_executor(self, func, *args, **kwargs):
        """Run synchronous Firebase operations in a thread pool."""
        loop = asyncio.get_event_loop()
        # Use the default thread pool executor
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    async def set(self, path: str, data: Dict[str, Any], key: Optional[str] = None) -> str:
        """Set data at specified path"""
        try:
            db = self._get_connection()
            result = await self._run_in_executor(db.put, path, key, data)
            return str(result) if result else ""
        except Exception as e:
            logger.error(f"Error in set operation: {e}")
            raise

    async def push(self, path: str, data: Dict[str, Any]) -> str:
        """Push data to specified path"""
        try:
            db = self._get_connection()
            result = await self._run_in_executor(db.post, path, data)
            return str(result.get('name')) if result else ""
        except Exception as e:
            logger.error(f"Error in push operation: {e}")
            raise

    async def get(self, path: str, key: str) -> Optional[Dict[str, Any]]:
        """Get data from specified path"""
        try:
            db = self._get_connection()
            result = await self._run_in_executor(db.get, path, key)
            return result
        except Exception as e:
            logger.error(f"Error in get operation: {e}")
            raise

    async def update(self, path: str, key: str, data: Dict[str, Any]) -> bool:
        """Update data at specified path"""
        try:
            db = self._get_connection()
            
            existing_data = await self.get(path, key)
            if existing_data is None:
                raise QueryError(f"User not found: {key}")
            
            updated_data = {**existing_data}
            for field, value in data.items():
                if value is not None:
                    updated_data[field] = value
            
            result = await self._run_in_executor(db.put, path, key, updated_data)
            return bool(result is None)
        except Exception as e:
            logger.error(f"Error in update operation: {e}")
            raise

    async def delete(self, path: str, key: str) -> bool:
        """Delete data at specified path"""
        try:
            db = self._get_connection()
            result = await self._run_in_executor(db.delete, f"{path}/{key}", None)
            return bool(result is None)
        except Exception as e:
            logger.error(f"Error in delete operation: {e}")
            raise

    async def query(
        self,
        path: str,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        start_at: Optional[Any] = None,
        end_at: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Query data with filters"""
        try:
            db = self._get_connection()
            
            # For simple key lookups, don't use orderBy
            if start_at and not order_by:
                result = await self._run_in_executor(db.get, f"{path}/{start_at}", None)
                return [result] if result else []
            
            # For complex queries, use the query parameters
            params = {
                'orderBy': f'"{order_by}"' if order_by else None,  # Wrap orderBy value in quotes
                'limitToFirst': limit,
                'startAt': f'"{start_at}"' if start_at else None,  # Wrap startAt value in quotes
                'endAt': f'"{end_at}"' if end_at else None  # Wrap endAt value in quotes
            }
            # Remove None values from params
            params = {k: v for k, v in params.items() if v is not None}
            
            result = await self._run_in_executor(db.get, path, None, params)
            return result if result else []
        except Exception as e:
            logger.error(f"Error in query operation: {e}")
            raise

    def __del__(self):
        """Cleanup connections when the helper is destroyed."""
        self.__connections.clear()
        logger.info("FirebaseRestRepository connections cleaned up.")