from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

from core.entities.user_entity import UserEntity
from core.entities.auth_entity import (
    TokenEntity,
    TokenDataEntity,
    CredentialsEntity,
    ValidationResultEntity,
)


class AuthenticationInterface(ABC):
    """
    Abstract interface for authentication services.
    Implements OWASP security best practices and JWT authentication.
    """

    @abstractmethod
    async def validate_password_strength(self, password: str) -> ValidationResultEntity:
        """
        Validate password strength against security requirements.

        Args:
            password: The password to validate

        Returns:
            ValidationResultEntity with validation results
        """
        pass

    @abstractmethod
    async def hash_password(self, password: str) -> str:
        """
        Securely hash a password using strong algorithms.

        Args:
            password: Plain text password to hash

        Returns:
            Securely hashed password
        """
        pass

    @abstractmethod
    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password to compare against

        Returns:
            True if password matches the hash, False otherwise
        """
        pass

    @abstractmethod
    async def authenticate(
        self, credentials: CredentialsEntity
    ) -> Optional[UserEntity]:
        """
        Authenticate a user with provided credentials.

        Args:
            credentials: CredentialsEntity containing authentication credentials

        Returns:
            UserEntity if authentication is successful, None otherwise
        """
        pass

    @abstractmethod
    async def create_token(
        self, user_data: UserEntity, expires_delta: Optional[timedelta] = None
    ) -> TokenEntity:
        """
        Create a JWT token for an authenticated user.

        Args:
            user_data: User information to encode in the token
            expires_delta: Optional expiration time delta

        Returns:
            TokenEntity containing the token and related information
        """
        pass

    @abstractmethod
    async def verify_token(self, token: str) -> Optional[TokenDataEntity]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token to verify

        Returns:
            TokenDataEntity with decoded token data if valid, None otherwise
        """
        pass

    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> Optional[TokenEntity]:
        """
        Refresh an expired access token using a refresh token.

        Args:
            refresh_token: The refresh token

        Returns:
            New TokenEntity if valid, None otherwise
        """
        pass

    @abstractmethod
    async def change_password(
        self, user_id: str, old_password: str, new_password: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Change a user's password with verification of the old password.

        Args:
            user_id: User identifier
            old_password: Current password for verification
            new_password: New password to set

        Returns:
            Tuple with (success_status, error_message_if_any)
        """
        pass

    @abstractmethod
    async def logout(self, token: str) -> bool:
        """
        Invalidate a user's token (add to blacklist or similar mechanism).

        Args:
            token: JWT token to invalidate

        Returns:
            True if logout was successful, False otherwise
        """
        pass

    @abstractmethod
    async def is_username_taken(self, username: str) -> bool:
        """
        Check if a username is already taken.

        Args:
            username: Username to check

        Returns:
            True if username is taken, False otherwise
        """
        pass

    @abstractmethod
    async def validate_user_input(
        self, user_data: Dict[str, Any], is_update: bool = False
    ) -> ValidationResultEntity:
        """
        Validates user input data.

        Args:
            user_data: Dictionary containing user data to validate
            is_update: Whether this is an update operation

        Returns:
            ValidationResultEntity with validation results
        """
        pass
