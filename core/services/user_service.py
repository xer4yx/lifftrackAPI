from typing import Optional, Dict, Any
from datetime import datetime, timezone

from core.entities.user import User
from core.interfaces.database import DatabaseRepository
from core.interfaces.auth import PasswordService, InputValidator
from core.exceptions.database import DatabaseError, QueryError, ValidationError
from utilities.monitoring.factory import MonitoringFactory

logger = MonitoringFactory.get_logger("user-service")

class UserService:
    def __init__(
        self, 
        database: DatabaseRepository,
        password_service: PasswordService,
        input_validator: InputValidator
    ):
        self.database = database
        self.password_service = password_service
        self.input_validator = input_validator

    async def create_user(
            self, 
            username: str, 
            email: str, 
            password: str, 
            first_name: str, 
            last_name: str, 
            phone_number: str
        ) -> str:
        """User service for creating a new user data in the database"""
        try:
            # Create initial user dict for validation
            user_dict = {
                'username': username,
                'email': email,
                'password': password,
                'first_name': first_name,
                'last_name': last_name,
                'phone_number': phone_number
            }
            
            # Add more specific validation checks
            if not password or len(password) < 8:
                raise ValidationError("Password must be at least 8 characters long")
            
            if not username or len(username) < 3:
                raise ValidationError("Username must be at least 3 characters long")
            
            if not email or '@' not in email:
                raise ValidationError("Invalid email format")
            
            # Validate user data before hashing
            validation_result = self.input_validator.validate(user_dict)
            if not validation_result:
                logger.error(f"Validation failed for user {username}")
                validation_errors = self.input_validator.get_validation_errors()  # Assuming this method exists
                raise ValidationError(f"Invalid user data: {validation_errors}")
            
            # After validation, create User entity with hashed password
            user = User(
                username=username,
                email=email,
                password=self.password_service.hash_password(password),
                first_name=first_name,
                last_name=last_name,
                phone_number=phone_number
            )
            
            user_dict = user.to_dict()
            
            snapshot = await self.database.set(path="users", data=user_dict, key=username)
            if not snapshot:
                raise DatabaseError(f"Database operation failed for user {username}")
            
            logger.info(f"User created successfully: {username}")
            return snapshot
        except ValidationError as ve:
            logger.error(f"Validation error for user {username}: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Failed to create user {username}: {str(e)}")
            raise

    async def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """User service for fetching a user data from the database"""
        try:
            logger.info(f"Fetching user: {username}")
            user = await self.database.get(path="users", key=username)
            
            if user is None:
                logger.warning(f"User not found: {username}")
                raise DatabaseError(f"Database operation {self.database.get.__name__} failed for user {username}")
                
            logger.info(f"User found: {username}")
            return user
        except Exception as e:
            logger.error(f"Failed to fetch for user {username}: {e}")

    async def update_user(self, username: str, user_data: User) -> bool:
        """User service for updating a user data in the database"""
        try:
            # First check if user exists
            existing_user = await self.database.get("users", username)
            if not existing_user:
                raise QueryError(f"User not found: {username}")
        
            if not self.input_validator.validate(user_data.to_dict()):
                raise ValidationError(f"Invalid user data: {user_data.to_dict()}")
            
            logger.info(f"Updating user: {username}")
            user_data.password = self.password_service.hash_password(user_data.password)
            success = await self.database.update(path="users", key=username, data=user_data.to_dict())
            if not success:
                raise DatabaseError(f"Database operation {self.database.update.__name__} failed for user {username}")
            
            logger.info(f"User updated successfully: {username}")
            return success
        except Exception as e:
            logger.error(f"Failed to update for user {username}: {e}")
            return False

    async def delete_user(self, user_id: str) -> bool:
        """User service for deleting a user data from the database"""
        try:
            # First check if user exists
            user = await self.database.get(path="users", key=user_id)
            if user is None:
                raise QueryError(f"User not found for deletion: {user_id}")
            
            logger.info(f"Deleting user: {user_id}")
            success = await self.database.delete(path="users", key=user_id)
            
            if not success:
                raise DatabaseError(f"Database operation {self.database.delete.__name__} failed for user {user_id}")
            
            logger.info(f"User deleted successfully: {user_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete for user {user_id}: {e}")

    async def get_user_password(self, username: str) -> str:
        """Get user's hashed password"""
        user = await self.database.get(path="users", key=username)
        if user:
            return user.get("password")
        return None

    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.password_service.verify_password(plain_password, hashed_password)

    async def hash_password(self, password: str) -> str:
        """Hash password"""
        return self.password_service.hash_password(password)
