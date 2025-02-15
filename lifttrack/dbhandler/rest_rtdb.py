import aiohttp
import asyncio
from typing import Optional, Dict, Any
from functools import wraps
import logging
from concurrent.futures import ThreadPoolExecutor
from lifttrack import config

logger = logging.getLogger(__name__)

class RTDBHelper:
    def __init__(self, dsn=None, authentication=None, pool_size=10):
        self.__dsn = config.get(section='Firebase', option='RTDB_DSN') or dsn
        self.__auth = config.get(section='Firebase', option='RTDB_AUTH') or authentication
        self.__pool_size = pool_size
        self.__session = None
        self.__lock = asyncio.Lock()
        logger.info(f"RTDBHelper initialized with pool size {pool_size}")

    async def __aenter__(self):
        await self.create_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_pool()

    async def create_pool(self):
        """Create the connection pool if it doesn't exist"""
        if self.__session is None:
            async with self.__lock:
                if self.__session is None:
                    conn = aiohttp.TCPConnector(limit=self.__pool_size)
                    self.__session = aiohttp.ClientSession(connector=conn)
                    logger.debug("Created new connection pool")

    async def close_pool(self):
        """Close the connection pool"""
        if self.__session is not None:
            async with self.__lock:
                if self.__session is not None:
                    await self.__session.close()
                    self.__session = None
                    logger.debug("Closed connection pool")

    def _with_session(func):
        """Decorator to handle session management"""
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            await self.create_pool()
            try:
                return await func(self, self.__session, *args, **kwargs)
            except Exception as e:
                logger.exception(f"Database operation failed: {str(e)}")
                raise
        return wrapper

    @_with_session
    async def put_data(self, session: aiohttp.ClientSession, user_data: dict[str, Any]):
        """
        Adds a new user to the database via HTTP PUT request.
        """
        try:
            logger.debug(f"Attempting to create user with data: {user_data}")
            url = f"{self.__dsn}/users/{user_data['username']}.json"
            if self.__auth:
                url += f"?auth={self.__auth}"
            
            async with session.put(url, json=user_data) as response:
                snapshot = await response.json()
                if snapshot is None:
                    logger.error(f"Failed to create user: {user_data['username']}")
                    raise ValueError('User not created')
                logger.info(f"User created successfully: {user_data['username']}")
                return snapshot
        except Exception as e:
            logger.exception(f"Exception in put_data for user {user_data['username']}: {e}")
            raise

    @_with_session
    async def get_data(self, session: aiohttp.ClientSession, username: str, data: Optional[str] = None):
        """
        Retrieves user data from the database.
        """
        try:
            url = f"{self.__dsn}/users/{username}"
            if data:
                url += f"/{data}"
            url += ".json"
            if self.__auth:
                url += f"?auth={self.__auth}"

            async with session.get(url) as response:
                snapshot = await response.json()
                if snapshot is None:
                    logger.warning(f"Data not found for user: {username}")
                else:
                    logger.info(f"Data retrieved for user: {username}")
                return snapshot
        except Exception as e:
            logger.exception(f"Exception in get_data for user {username}: {e}")
            raise

    @_with_session
    async def get_all_data(self, session: aiohttp.ClientSession):
        """
        Retrieves all user data from the database.
        """
        url = f"{self.__dsn}/users.json"
        if self.__auth:
            url += f"?auth={self.__auth}"
        
        async with session.get(url) as response:
            return await response.json()

    @_with_session
    async def update_data(self, session: aiohttp.ClientSession, username: str, user_data: dict):
        """
        Updates user data in the database.
        """
        try:
            url = f"{self.__dsn}/users/{username}.json"
            if self.__auth:
                url += f"?auth={self.__auth}"
            
            async with session.put(url, json=user_data) as response:
                snapshot = await response.json()
                if snapshot is None:
                    logger.error(f"Failed to update user: {username}")
                else:
                    logger.info(f"User updated successfully: {username}")
                return snapshot
        except Exception as e:
            logger.exception(f"Exception in update_data for user {username}: {e}")
            raise

    @_with_session
    async def delete_data(self, session: aiohttp.ClientSession, username: str):
        """
        Deletes a user from the database.
        """
        try:
            url = f"{self.__dsn}/users/{username}.json"
            if self.__auth:
                url += f"?auth={self.__auth}"
            
            async with session.delete(url) as response:
                deleted = await response.json()
                if deleted is False:
                    logger.error(f"Failed to delete user: {username}")
                else:
                    logger.info(f"User deleted successfully: {username}")
                return deleted
        except Exception as e:
            logger.exception(f"Exception in delete_data for user {username}: {e}")
            raise

    @_with_session
    async def get_progress(self, session: aiohttp.ClientSession, username: str, exercise_name: Optional[str] = None) -> Optional[Dict]:
        """
        Retrieves progress data for a user.
        """
        url = f"{self.__dsn}/progress/{username}"
        if exercise_name:
            url += f"/{exercise_name}"
        url += ".json"
        if self.__auth:
            url += f"?auth={self.__auth}"
        
        async with session.get(url) as response:
            return await response.json()

    @_with_session
    async def put_progress(self, session: aiohttp.ClientSession, username: str, exercise_name: str, exercise_data: dict):
        """
        Adds exercise progress data for a user. New data is appended under the date,
        not updated.
        
        Args:
            username: Username of the user
            exercise_name: Name of the exercise
            exercise_data: ExerciseData object containing the progress data
        """
        try:
            # Get existing data for the user
            existing_data = await self.get_progress(username)
            
            # Add new data
            if not isinstance(existing_data, dict):
                existing_data = {}
            
            if exercise_name not in existing_data:
                existing_data[exercise_name] = {}
                
            # Save to database
            url = f"{self.__dsn}/progress/{username}.json"
            if self.__auth:
                url += f"?auth={self.__auth}"
            
            async with session.put(url, json=existing_data) as response:
                snapshot = await response.json()
                if snapshot is None:
                    logger.error(f"Failed to save progress for user {username}, exercise: {exercise_name}")
                    raise ValueError('Progress not saved')
                logger.info(f"Progress saved successfully for user {username}, exercise: {exercise_name}")
            
        except Exception as e:
            logger.exception(f"Exception in put_progress for user {username}, exercise {exercise_name}: {e}")
            raise

    @_with_session
    async def update_progress(self, session: aiohttp.ClientSession, username: str, exercise_name: str, exercise_data: dict):
        """
        Updates exercise progress data for a user.
        
        Args:
            username: Username of the user
            exercise_name: Name of the exercise
            exercise_data: ExerciseData object containing the updated progress data
        """
        try:
            return await self.put_progress(username, exercise_name, exercise_data)
        except Exception as e:
            logger.exception(f"Exception in update_progress for user {username}: {e}")
            raise

    @_with_session
    async def delete_progress(self, session: aiohttp.ClientSession, username: str, exercise_name: Optional[str] = None):
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
                existing_data = await self.get_progress(username)
                if existing_data and exercise_name in existing_data:
                    del existing_data[exercise_name]
                    url = f"{self.__dsn}/progress/{username}.json"
                    if self.__auth:
                        url += f"?auth={self.__auth}"
                    
                    async with session.put(url, json=existing_data) as response:
                        snapshot = await response.json()
                        return snapshot is not None
                return False
            else:
                # Delete all progress
                url = f"{self.__dsn}/progress/{username}.json"
                if self.__auth:
                    url += f"?auth={self.__auth}"
                
                async with session.delete(url) as response:
                    deleted = await response.json()
                    if deleted is False:
                        logger.error(f"Failed to delete progress for user: {username}")
                    else:
                        logger.info(f"Progress deleted successfully for user: {username}")
                    return deleted
                
        except Exception as e:
            logger.exception(f"Exception in delete_progress for user {username}: {e}")
            raise
