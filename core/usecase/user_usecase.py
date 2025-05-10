from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging

from core.entities import UserEntity
from core.interface import AuthInterface, UserValidationInterface, PasswordValidationInterface, NTFInterface
from core.service import generate_user_id, validate_email, validate_phone_number, validate_password


class UserUseCase:
    """
    Use case class for handling user-related business logic.
    Implements operations like user registration, authentication, and profile management.
    """
    
    def __init__(
        self,
        auth_service: AuthInterface,
        user_validation: UserValidationInterface,
        password_validation: PasswordValidationInterface,
        database_service: NTFInterface
    ):
        self.auth_service = auth_service
        self.user_validation = user_validation
        self.password_validation = password_validation
        self.database_service = database_service
        self.logger = logging.getLogger(__name__)
    
    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        first_name: str,
        last_name: str
    ) -> Optional[UserEntity]:
        """
        Register a new user in the system.
        
        Args:
            username: User's chosen username
            email: User's email address
            password: User's password
            first_name: User's first name
            last_name: User's last name
            
        Returns:
            UserEntity if registration successful, None otherwise
        """
        try:
            # Validate input data
            if not validate_email(email):
                return None
                
            if not validate_password(password):
                return None
                
            # Check if user already exists
            if await self.user_validation.is_username_taken(username):
                return None
                
            if await self.user_validation.is_email_taken(email):
                return None
            
            # Create user entity
            hashed_password = await self.password_validation.hash_password(password)
            
            user = UserEntity(
                id=generate_user_id(),
                username=username,
                email=email,
                password=hashed_password,
                first_name=first_name,
                last_name=last_name,
                created_at=datetime.now(timezone.utc)
            )
            
            # Save user to database
            await self.database_service.set_data(user.id, user.model_dump(exclude_none=True))
            return user
        except Exception as e:
            self.logger.error(f"Error registering user: {str(e)}")
            return None
    
    async def authenticate_user(self, credentials: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Authenticate a user and generate access tokens.
        
        Args:
            credentials: Dictionary containing username_or_email and password
            
        Returns:
            Dictionary with access and refresh tokens if authentication successful, None otherwise
        """
        try:
            user = await self.auth_service.authenticate(credentials)
            if not user:
                return None
                
            # Update last login time
            user.last_login = datetime.now(timezone.utc)
            await self.database_service.set_data(user.id, user.model_dump())
            
            # Generate tokens
            return await self.auth_service.generate_tokens(user.id)
        except Exception as e:
            self.logger.error(f"Error authenticating user: {str(e)}")
            return None
    
    async def get_user_profile(self, user_id: str) -> Optional[UserEntity]:
        """
        Retrieve a user's profile information.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            UserEntity if user exists, None otherwise
        """
        try:
            return await self.database_service.get_data(user_id)
        except Exception as e:
            self.logger.error(f"Error getting user profile: {str(e)}")
            return None
    
    async def update_user_profile(
        self,
        user: UserEntity
    ) -> Optional[UserEntity]:
        """
        Update a user's profile information.
        
        Args:
            user: UserEntity containing updated profile information
            
        Returns:
            Updated UserEntity if successful, None otherwise
        """
        try:
            user = await self.database_service.get_data(user.id)
            if not user:
                return None
                
            # Update fields if provided
            if user.first_name:
                user.first_name = user.first_name
                
            if user.last_name:
                user.last_name = user.last_name
                
            if user.email and user.email != user.email:
                if not validate_email(user.email):
                    return None
                    
                if await self.user_validation.is_email_taken(user.email):
                    return None
                    
                user.email = user.email
                user.is_verified = False  # Require re-verification for new email
            
            user.updated_at = datetime.now(timezone.utc)
            
            await self.database_service.set_data(user.id, user.model_dump(exclude_none=True))
            return user
        except Exception as e:
            self.logger.error(f"Error updating user profile: {str(e)}")
            return None
    
    async def change_user_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """
        Change a user's password.
        
        Args:
            user_id: User's unique identifier
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully, False otherwise
        """
        try:
            if not validate_password(new_password):
                return False
                
            return await self.auth_service.change_password(user_id, old_password, new_password)
        except Exception as e:
            self.logger.error(f"Error changing user password: {str(e)}")
            return False
    
    async def delete_user(self, user_id: str) -> bool:
        """
        Mark a user as deleted in the system.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            True if user was deleted successfully, False otherwise
        """
        try:
            user = await self.database_service.get_data(user_id)
            if not user:
                return False
                
            user.is_deleted = True
            user.updated_at = datetime.now(timezone.utc)
            
            await self.database_service.set_data(user.id, user.model_dump(exclude_none=True))
            return True
        except Exception as e:
            self.logger.error(f"Error deleting user: {str(e)}")
            return False

