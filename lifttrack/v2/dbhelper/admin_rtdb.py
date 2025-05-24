import firebase_admin
from firebase_admin import credentials, db

from typing import Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
import threading

from lifttrack.utils.logging_config import setup_logger

logger = setup_logger("lifttrack.admin_rtdb", "db.log")

class FirebaseDBHelper:
    """
    A comprehensive database helper class for Firebase Realtime Database 
    with connection pooling and advanced error handling.
    
    This class provides:
    - Singleton pattern for Firebase initialization
    - Thread-safe connection pooling
    - Comprehensive CRUD operations
    - Error handling and logging
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(
        cls,
        credentials_path: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Implement singleton pattern to ensure only one Firebase app instance.
        
        Args:
            credentials_path (Optional[str]): Path to Firebase credentials JSON file.
                If None, assumes credentials are already initialized.
        """
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    try:
                        if credentials_path:
                            cred = credentials.Certificate(credentials_path)
                            if not firebase_admin._apps:
                                if not options or 'databaseURL' not in options:
                                    raise ValueError("databaseURL must be provided in options")
                                firebase_admin.initialize_app(
                                    credential=cred, 
                                    options=options
                                )
                        
                        cls._instance = super().__new__(cls)
                        cls._instance._executor = ThreadPoolExecutor(
                            max_workers=10,
                            thread_name_prefix='rtdb-pool'
                        )
                        cls._instance._db = db.reference()
                    except Exception as e:
                        logger.error(f"Firebase initialization error: {e}")
                        raise
        
        return cls._instance
    
    def get_reference(self, path: str):
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
            logger.error(f"Error getting reference: {e}")
            raise
    
    def set_data(self, path: str, data: Dict[str, Any], key: Optional[str] = None) -> str:
        """
        Add data to a specified path.
        
        Args:
            path (str): Database path
            data (Dict): Data to be added
            key (Optional[str]): Custom key. If None, auto-generated.
        
        Returns:
            str: Key of the added data
        """
        try:
            ref = self.get_reference(path)
            if key:
                ref.child(key).set(data)
                return key
            else:
                return ref.push(data).key
        except Exception as e:
            logger.error(f"Error adding data: {e}")
            raise
        
    def push_data(self, path: str, data: Dict[str, Any], key: Optional[str] = None) -> str:
        """
        Add data to a specified path using a push.
        
        Args:
            path (str): Database path
            data (Dict): Data to be added
            key (Optional[str]): Custom key. If None, auto-generated.
        
        Returns:
            str: Key of the added data
        """
        try:
            ref = self.get_reference(path)
            if key:
                ref.child(key).push(data)
                return key
            else:
                return ref.push(data).key
        except Exception as e:
            logger.error(f"Error adding data: {e}")
            raise
    
    def get_data(self, path: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data at a specific path and key.
        
        Args:
            path (str): Database path
            key (str): Data key to retrieve
        
        Returns:
            Optional[Dict]: Data or None if not found
        """
        try:
            return self.get_reference(path).child(key).get()
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            raise
    
    def update_data(self, path: str, key: str, update_data: Dict[str, Any]) -> bool:
        """
        Update existing data.
        
        Args:
            path (str): Database path
            key (str): Key to update
            update_data (Dict): Fields to update
        """
        try:
            self.get_reference(path).child(key).update(update_data)
            return True
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return False
    
    def delete_data(self, path: str, key: str) -> bool:
        """
        Delete data at a specific path and key.
        
        Args:
            path (str): Database path
            key (str): Key to delete
        """
        try:
            self.get_reference(path).child(key).delete()
            return True
        except Exception as e:
            logger.error(f"Error deleting data: {e}")
            return False
    
    def query_data(self, path: str, 
                   order_by: Optional[str] = None,
                   limit: Optional[int] = None,
                   start_at: Optional[Any] = None,
                   end_at: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Query data with ordering and filtering.
        
        Args:
            path (str): Database path to query
            order_by (Optional[str]): Child key to order by
            limit (Optional[int]): Maximum number of results
            start_at (Optional[Any]): Value to start at
            end_at (Optional[Any]): Value to end at
        
        Returns:
            List[Dict]: List of matching data
        """
        try:
            ref = self.get_reference(path)
            
            if order_by:
                ref = ref.order_by_child(order_by)
            if start_at is not None:
                ref = ref.start_at(start_at)
            if end_at is not None:
                ref = ref.end_at(end_at)
            if limit:
                ref = ref.limit_to_first(limit)
            
            return ref.get()
        except Exception as e:
            logger.error(f"Query error: {e}")
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