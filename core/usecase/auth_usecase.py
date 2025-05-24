from datetime import timedelta
from typing import Dict, Any, Optional, Tuple

from core.entities.user_entity import UserEntity
from core.entities.auth_entity import (
    TokenEntity, 
    TokenDataEntity, 
    CredentialsEntity,
    ValidationResultEntity
)
from core.interface import AuthenticationInterface


class AuthUseCase:
    """
    Use case for authentication-related operations.
    Implements the business logic using the authentication service.
    """
    
    def __init__(self, auth_service: AuthenticationInterface):
        self.auth_service = auth_service
        
    async def register_user(self, user_data: Dict[str, Any]) -> Tuple[bool, Optional[UserEntity], Optional[str]]:
        """
        Register a new user with validation.
        
        Args:
            user_data: User registration data
            
        Returns:
            Tuple containing (success, user_entity, error_message)
        """
        # Validate user input
        validation_result = await self.auth_service.validate_user_input(user_data)
        if not validation_result.is_valid:
            return False, None, "; ".join(validation_result.errors)
            
        # Check if username is taken
        if await self.auth_service.is_username_taken(user_data.get("username")):
            return False, None, "Username already exists"
            
        # Hash the password
        user_data["password"] = await self.auth_service.hash_password(user_data.get("password"))
        
        # Ensure new fields have default values if not provided
        if "phone_number" not in user_data:
            return False, None, "Phone number is required"
        if "profile_picture" not in user_data:
            user_data["profile_picture"] = None
            
        # This will be handled by the repository implementation
        user_entity = UserEntity(**user_data)
        return True, user_entity, None
    
    async def login(self, credentials: CredentialsEntity) -> Tuple[bool, Optional[TokenEntity], Optional[UserEntity], Optional[str]]:
        """
        Authenticate a user and generate a token.
        
        Args:
            credentials: Login credentials
            
        Returns:
            Tuple containing (success, token_entity, user_entity, error_message)
        """
        # Authenticate user
        user = await self.auth_service.authenticate(credentials)
        if not user:
            return False, None, None, "Invalid username or password"
            
        # Create token
        token = await self.auth_service.create_token(
            user, 
            timedelta(minutes=30)  # Use config value in actual implementation
        )
        
        return True, token, user, None
    
    async def validate_token(self, token: str) -> Tuple[bool, Optional[UserEntity], Optional[str]]:
        """
        Validate a token and get the associated user.
        
        Args:
            token: JWT token to validate
            
        Returns:
            Tuple containing (is_valid, user_entity, error_message)
        """
        # Verify token
        token_data = await self.auth_service.verify_token(token)
        if not token_data:
            return False, None, "Invalid or expired token"
            
        # Get user by username from token data
        # This should be implemented by repository in actual implementation
        # Here we're assuming the authentication service can handle this
        user = await self.auth_service.authenticate(
            CredentialsEntity(username=token_data.username, password="")
        )
        
        if not user:
            return False, None, "User not found"
            
        return True, user, None
    
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> Tuple[bool, Optional[str]]:
        """
        Change a user's password.
        
        Args:
            user_id: User identifier
            old_password: Current password 
            new_password: New password
            
        Returns:
            Tuple containing (success, error_message)
        """
        # Validate new password strength
        validation = await self.auth_service.validate_password_strength(new_password)
        if not validation.is_valid:
            return False, "; ".join(validation.errors)
            
        # Attempt to change password
        return await self.auth_service.change_password(user_id, old_password, new_password)
    
    async def logout(self, token: str) -> bool:
        """
        Log out a user by invalidating their token.
        
        Args:
            token: JWT token to invalidate
            
        Returns:
            True if successful, False otherwise
        """
        return await self.auth_service.logout(token)
    
    async def refresh_token(self, refresh_token: str) -> Tuple[bool, Optional[TokenEntity], Optional[str]]:
        """
        Refresh an access token.
        
        Args:
            refresh_token: The refresh token
            
        Returns:
            Tuple containing (success, new_token, error_message)
        """
        new_token = await self.auth_service.refresh_token(refresh_token)
        if not new_token:
            return False, None, "Invalid or expired refresh token"
            
        return True, new_token, None
        
    async def validate_user_input(self, user_data: Dict[str, Any], is_update: bool = False) -> ValidationResultEntity:
        """
        Validate user input for registration or profile update.
        
        Args:
            user_data: User data to validate
            is_update: Whether this is an update operation
            
        Returns:
            ValidationResultEntity with validation results
        """
        return await self.auth_service.validate_user_input(user_data, is_update) 