from typing import Dict, Any, Optional, List
from firebase_admin import credentials, db
import firebase_admin
import threading
from concurrent.futures import ThreadPoolExecutor

from core.interfaces.database import DatabaseRepository
from core.exceptions import QueryError
from utilities.monitoring.factory import MonitoringFactory

logger = MonitoringFactory.get_logger("firebase-admin")

class FirebaseAdminRepository(DatabaseRepository):
    _instance = None
    _lock = threading.Lock()
    
    def __new__(
        cls, 
        credentials_path: Optional[str] = None, 
        options: Optional[Dict[str, Any]] = None
    ):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    try:
                        if credentials_path:
                            cred = credentials.Certificate(credentials_path)
                            if not firebase_admin._apps:
                                if not options or 'databaseURL' not in options:
                                    logger.error("databaseURL must be provided in options")
                                    raise ValueError("databaseURL must be provided in options")
                                firebase_admin.initialize_app(
                                    credential=cred, 
                                    options=options
                                )
                        
                        cls._instance = super().__new__(cls)
                        cls._instance._executor = ThreadPoolExecutor(
                            max_workers=10,
                            thread_name_prefix='admin-rtdb-pool'
                        )
                        cls._instance._db = db.reference()
                    except Exception as e:
                        logger.error(f"Firebase initialization error: {e}")
                        raise
        
        return cls._instance
    
    def get_reference(self, path: str) -> db.Reference:
        """
        Get a reference to a specific path in the Realtime Database.
        
        Args:
            path (str): Path to the database location
        
        Returns:
            db.Reference: Reference to the specified location
        """
        try:
            return self._db.child(path)
        except Exception as e:
            logger.error(f"Exception at {self.__class__.__name__}.{self.get_reference.__name__}: {e}")
            raise
    
    async def set(self, path: str, data: Dict[str, Any], key: Optional[str] = None) -> str:
        try:
            ref = self.get_reference(path)
            if key:
                ref.child(key).set(data)
                return key
            return ref.push(data).key
        except Exception as e:
            logger.error(f"Error setting data: {e}")
            raise QueryError(f"Failed to set data: {e}")
    
    async def push(self, path: str, data: Dict[str, Any]) -> str:
        try:
            ref = self.get_reference(path)
            return ref.push(data).key
        except Exception as e:
            logger.error(f"Error pushing data: {e}")
            raise QueryError(f"Failed to push data: {e}")
        
    async def get(self, path: str, key: str) -> Optional[Dict[str, Any]]:
        try:
            ref = self.get_reference(path)
            return ref.child(key).get()
        except Exception as e:
            logger.error(f"Error getting data: {e}")
            raise QueryError(f"Failed to get data: {e}")

    async def update(self, path: str, key: str, data: Dict[str, Any]) -> bool:
        try:
            ref = self.get_reference(path).child(key)
            ref.update(data)
            return True
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            raise QueryError(f"Failed to update data: {e}")

    async def delete(self, path: str, key: str) -> bool:
        try:
            ref = self.get_reference(path).child(key)
            ref.delete()
            return True
        except Exception as e:
            logger.error(f"Error deleting data: {e}")
            raise QueryError(f"Failed to delete data: {e}")

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
            ref = self.get_reference(path)
            return ref.order_by_child(order_by).limit_to_first(limit).start_at(start_at).end_at(end_at).get()
        except Exception as e:
            logger.error(f"Exception at {self.__class__.__name__}.{self.query.__name__}: {e}")
            return []

    def close(self):
        """
        Cleanup method to shutdown thread pool and Firebase app.
        """
        if self._executor:
            self._executor.shutdown(wait=True)
        
        # Optionally, delete the app instance
        if firebase_admin._apps:
            firebase_admin.delete_app(firebase_admin.get_app())
