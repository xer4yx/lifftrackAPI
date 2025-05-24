import logging
from typing import List, Optional, Dict, Any

from core.entities import UserEntity
from infrastructure.database import FirebaseREST

class UserRepository:
    """
    Repository for user-related operations with Firebase.
    """
    
    def __init__(self, firebase: FirebaseREST):
        self.firebase = firebase
        self.collection = "users"
        self.logger = logging.getLogger("user_repository")
        
    async def get_all_users(self) -> List[UserEntity]:
        """
        Retrieve all users from the database.
        """
        try:
            result = await self.firebase.get_data(self.collection)
            if not result:
                return []
                
            users = []
            for user_id, user_data in result.items():
                user_data["id"] = user_id
                users.append(UserEntity.model_validate(user_data))
                
            return users
        except Exception as e:
            self.logger.error(f"Error retrieving all users: {e}")
            return []
    
    async def get_user_by_username(self, username: str) -> Optional[UserEntity]:
        """
        Retrieve a user by username.
        """
        try:
            # Firebase doesn't support direct queries, so we need to get all and filter
            result = await self.firebase.get_data(f"{self.collection}/{username}")
            
            if not result:
                return None
            
            return UserEntity.model_validate(result)
        except Exception as e:
            self.logger.error(f"Error retrieving user by username {username}: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserEntity]:
        """
        Retrieve a user by ID.
        """
        try:
            result = await self.firebase.get_data(f"{self.collection}/{user_id}")
            if not result:
                return None
                
            result["id"] = user_id
            return UserEntity.model_validate(result)
        except Exception as e:
            self.logger.error(f"Error retrieving user by ID {user_id}: {e}")
            return None
    
    async def update_password(self, user_id: str, hashed_password: str) -> bool:
        """
        Update a user's password.
        """
        try:
            await self.firebase.set_data(f"{self.collection}/{user_id}", {"password": hashed_password})
            return True
        except Exception as e:
            self.logger.error(f"Error updating password for user {user_id}: {e}")
            return False
    
    async def check_username_exists(self, username: str) -> bool:
        """
        Check if a username already exists.
        """
        try:
            result = await self.firebase.get_data(f"{self.collection}/{username}")
            return bool(result == None)
        except Exception as e:
            self.logger.error(f"Error checking if username {username} exists: {e}")
            return False
    
    async def create_user(self, user_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new user in the database.
        """
        try:
            result =await self.firebase.set(f"{self.collection}", user_data)
            if result is None:
                return user_data.get("username") 
            else:
                raise Exception("User already exists") # Firebase returns the new ID as "name"
        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            return None
    
    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """
        Update a user's data.
        """
        try:
            await self.firebase.set_data(f"{self.collection}/{user_id}", user_data)
            return True
        except Exception as e:
            self.logger.error(f"Error updating user {user_id}: {e}")
            return False
    
    async def delete_user(self, user_id: str) -> bool:
        """
        Delete a user from the database.
        """
        try:
            await self.firebase.delete_data(f"{self.collection}/{user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting user {user_id}: {e}")
            return False 