import firebase_admin
from firebase_admin import credentials, firestore

from typing import Any, Dict, Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor
import threading


class FirebaseDBHelper:
    """
    A comprehensive database helper class for Firebase Firestore 
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
                    # Initialize the Firebase app if credentials are provided
                    if credentials_path:
                        cred = credentials.Certificate(credentials_path)
                        firebase_admin.initialize_app(
                            credential=cred,
                            options=options
                        )
                    
                    # Create the singleton instance
                    cls._instance = super().__new__(cls)
                    
                    # Setup thread pool for connection pooling
                    cls._instance._executor = ThreadPoolExecutor(
                        max_workers=10,  # Configurable number of worker threads
                        thread_name_prefix='firestore-pool'
                    )
                    
                    # Get Firestore client
                    cls._instance._db = firestore.client()
        
        return cls._instance
    
    def get_collection(self, collection_name: str):
        """
        Retrieve a specific Firestore collection.
        
        Args:
            collection_name (str): Name of the Firestore collection
        
        Returns:
            firestore.CollectionReference: Reference to the specified collection
        """
        return self._db.collection(collection_name)
    
    def add_document(self, collection_name: str, data: Dict[str, Any], 
                     doc_id: Optional[str] = None) -> str:
        """
        Add a document to a specified collection.
        
        Args:
            collection_name (str): Target collection name
            data (Dict): Document data to be added
            doc_id (Optional[str]): Custom document ID. If None, auto-generated.
        
        Returns:
            str: ID of the added document
        """
        try:
            collection_ref = self.get_collection(collection_name)
            
            if doc_id:
                # Add document with specific ID
                doc_ref = collection_ref.document(doc_id)
                doc_ref.set(data)
                return doc_id
            else:
                # Add document with auto-generated ID
                doc_ref = collection_ref.add(data)
                return doc_ref[1].id
        except Exception as e:
            print(f"Error adding document: {e}")
            raise
    
    def get_document(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            collection_name (str): Source collection name
            doc_id (str): Document ID to retrieve
        
        Returns:
            Optional[Dict]: Document data or None if not found
        """
        try:
            doc_ref = self.get_collection(collection_name).document(doc_id)
            doc = doc_ref.get()
            
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            print(f"Error retrieving document: {e}")
            raise
    
    def update_document(self, collection_name: str, doc_id: str, 
                        update_data: Dict[str, Any]) -> bool:
        """
        Update an existing document.
        
        Args:
            collection_name (str): Target collection name
            doc_id (str): Document ID to update
            update_data (Dict): Fields to update
        
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            doc_ref = self.get_collection(collection_name).document(doc_id)
            doc_ref.update(update_data)
            return True
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def delete_document(self, collection_name: str, doc_id: str) -> bool:
        """
        Delete a specific document.
        
        Args:
            collection_name (str): Source collection name
            doc_id (str): Document ID to delete
        
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            doc_ref = self.get_collection(collection_name).document(doc_id)
            doc_ref.delete()
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def batch_write(self, operations: List[Callable]) -> bool:
        """
        Perform batch write operations with connection pooling.
        
        Args:
            operations (List[Callable]): List of database operations to execute
        
        Returns:
            bool: True if all operations successful, False otherwise
        """
        try:
            batch = self._db.batch()
            
            # Submit all operations to thread pool
            futures = [
                self._executor.submit(op, batch) 
                for op in operations
            ]
            
            # Wait for all futures to complete
            for future in futures:
                future.result()
            
            # Commit the batch
            batch.commit()
            return True
        except Exception as e:
            print(f"Batch write error: {e}")
            return False
    
    def query_collection(self, collection_name: str, 
                         filters: Optional[List[tuple]] = None, 
                         order_by: Optional[str] = None, 
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform advanced querying on a collection.
        
        Args:
            collection_name (str): Collection to query
            filters (Optional[List[tuple]]): List of filter tuples (field, op, value)
            order_by (Optional[str]): Field to order results by
            limit (Optional[int]): Maximum number of results
        
        Returns:
            List[Dict]: List of matching documents
        """
        try:
            query = self.get_collection(collection_name)
            
            # Apply filters
            if filters:
                for field, op, value in filters:
                    query = query.where(field, op, value)
            
            # Apply ordering
            if order_by:
                query = query.order_by(order_by)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Execute query and return results
            docs = query.stream()
            return [doc.to_dict() for doc in docs]
        
        except Exception as e:
            print(f"Query error: {e}")
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