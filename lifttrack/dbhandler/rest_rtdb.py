import logging
from firebase import firebase
from lifttrack import config
from lifttrack.utils.logging_config import setup_logger
from typing import Dict, Optional
from lifttrack.models import Exercise, Progress, ExerciseData
from concurrent.futures import ThreadPoolExecutor
from functools import wraps


# Logging Configuration
logger = setup_logger("rtdbHelper", "lifttrack_db.log") 


class RTDBHelper:
    def __init__(self, dsn=None, authentication=None, max_workers=5):
        self.__dsn = config.get(section='Firebase', option='RTDB_DSN') or dsn
        self.__auth = config.get(section='Firebase', option='RTDB_AUTH') or authentication
        self.__pool = ThreadPoolExecutor(max_workers=max_workers)
        self.__connections = {}
        logger.info(f"RTDBHelper initialized with {max_workers} workers.")

    def _get_connection(self):
        """Get or create a database connection for the current thread."""
        import threading
        thread_id = threading.get_ident()
        
        if thread_id not in self.__connections:
            self.__connections[thread_id] = firebase.FirebaseApplication(
                dsn=self.__dsn,
                authentication=(lambda: None, lambda: self.__auth)[self.__auth != 'None']()
            )
            logger.debug(f"Created new connection for thread {thread_id}")
        
        return self.__connections[thread_id]

    def _with_connection(func):
        """Decorator to handle database connections."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            db = self._get_connection()
            try:
                return func(self, db, *args, **kwargs)
            except Exception as e:
                logger.exception(f"Database operation failed: {str(e)}")
                raise
        return wrapper

    @_with_connection
    def put_data(self, db, user_data: dict[str, any]):
        """
        Adds a new user to the database via HTTP PUT request.
        """
        try:
            logger.debug(f"Attempting to create user with data: {user_data}")
            future = self.__pool.submit(
                db.put,
                url='/users',
                name=user_data['username'],
                data=user_data
            )
            snapshot = future.result()
            if snapshot is None:
                logger.error(f"Failed to create user: {user_data['username']}")
                raise ValueError('User not created')
            logger.info(f"User created successfully: {user_data['username']}")
        except Exception as e:
            logger.exception(f"Exception in put_data for user {user_data['username']}: {e}")
            raise
    
    @_with_connection
    def get_data(self, db, username, data=None):
        """
        Retrieves user data from the database.

        :param username: Username of the user.
        :param data: Specific field to retrieve.
        :return: User data or specific field.
        """
        try:
            snapshot = db.get(
                url=f'/users/{username}',
                name=data
            )
            if snapshot is None:
                logger.warning(f"Data not found for user: {username}")
            else:
                logger.info(f"Data retrieved for user: {username}")
            return snapshot
        except Exception as e:
            logger.exception(f"Exception in get_data for user {username}: {e}")
            raise
    
    @_with_connection
    def get_all_data(self, db):
        """
        Retrieves all user data from the database.
        """
        return db.get('/users', None)

    @_with_connection
    def update_data(self, db, username, user_data):
        """
        Updates user data in the database.

        :param username: Username of the user.
        :param user_data: Data to update.
        :return: Snapshot of updated data.
        """
        try:
            snapshot = db.put(
                url=f'/users',
                name=username,
                data=user_data
            )
            if snapshot is None:
                logger.error(f"Failed to update user: {username}")
            else:
                logger.info(f"User updated successfully: {username}")
            return snapshot
        except Exception as e:
            logger.exception(f"Exception in update_data for user {username}: {e}")
            raise

    @_with_connection
    def delete_data(self, db, username):
        """
        Deletes a user from the database.

        :param username: Username of the user to delete.
        :return: True if deleted, else False.
        """
        try:
            deleted = db.delete(
                url=f'/users',
                name=username
            )
            if deleted is False:
                logger.error(f"Failed to delete user: {username}")
            else:
                logger.info(f"User deleted successfully: {username}")
            return deleted
        except Exception as e:
            logger.exception(f"Exception in delete_data for user {username}: {e}")
            raise

    @_with_connection
    def put_progress(self, db, username: str, exercise: Exercise):
        """
        Adds exercise progress data for a user. New data is appended under the date,
        not updated.
        
        Args:
            username: Username of the user
            exercise_name: Name of the exercise
            exercise_data: ExerciseData object containing the progress data
        """
        try:
            # Save to database
            future = self.__pool.submit(
                db.put,
                url=f'/progress',
                name=username,
                data=exercise.model_dump()
            )
            snapshot = future.result()
            
            if snapshot is None:
                logger.error(f"Failed to save progress for user: {username}")
                raise ValueError('Progress not saved')
            logger.info(f"Progress saved successfully for user: {username}")
            
        except Exception as e:
            logger.exception(f"Exception in put_progress for user {username}: {e}")
            raise

    @_with_connection
    def get_progress(self, db, username: str, exercise_name: Optional[str] = None) -> Optional[Dict]:
        """
        Retrieves progress data from the database.

        Args:
            username: Username of the user
            exercise_name: Optional specific exercise to retrieve
        Returns:
            Progress data dictionary or None
        """
        try:
            snapshot = db.get(
                url=f'/progress/{username}',
                name=None
            )
            
            if snapshot is None:
                logger.warning(f"Progress data not found for user: {username}")
                return None
            
            # Validate against Progress model
            progress_data = Progress(**snapshot)
            
            if exercise_name:
                if exercise_name in progress_data.exercise:
                    return {"username": username, "exercise": {
                        exercise_name: progress_data.exercise[exercise_name]
                    }}
                return None
            
            logger.info(f"Progress data retrieved for user: {username}")
            return progress_data.model_dump()
            
        except Exception as e:
            logger.exception(f"Exception in get_progress for user {username}: {e}")
            raise

    @_with_connection
    def update_progress(self, db, username: str, exercise_name: str, exercise_data: any):
        """
        Updates exercise progress data for a user.
        
        Args:
            username: Username of the user
            exercise_name: Name of the exercise
            exercise_data: ExerciseData object containing the updated progress data
        """
        try:
            return self.put_progress(username, exercise_name, exercise_data)
        except Exception as e:
            logger.exception(f"Exception in update_progress for user {username}: {e}")
            raise

    @_with_connection
    def delete_progress(self, db, username: str, exercise_name: Optional[str] = None):
        """
        Deletes progress data from the database.

        Args:
            username: Username of the user
            exercise_name: Optional specific exercise to delete
        Returns:
            True if deleted, else False
        """
        try:
            if exercise_name:
                # Delete specific exercise
                existing_data = self.get_progress(username)
                if existing_data and "exercise" in existing_data:
                    if exercise_name in existing_data["exercise"]:
                        del existing_data["exercise"][exercise_name]
                        snapshot = db.put(
                            url='/progress',
                            name=username,
                            data=existing_data
                        )
                        return snapshot is not None
                return False
            else:
                # Delete all progress
                deleted = db.delete(
                    url='/progress',
                    name=username
                )
                if deleted is False:
                    logger.error(f"Failed to delete progress for user: {username}")
                else:
                    logger.info(f"Progress deleted successfully for user: {username}")
                return deleted
                
        except Exception as e:
            logger.exception(f"Exception in delete_progress for user {username}: {e}")
            raise

    def __del__(self):
        """Cleanup connections when the helper is destroyed."""
        self.__pool.shutdown(wait=True)
        self.__connections.clear()
        logger.info("RTDBHelper connections cleaned up.")


rtdb = RTDBHelper()
