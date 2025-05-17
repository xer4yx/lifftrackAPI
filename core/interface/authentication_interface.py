from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class PasswordValidationInterface(ABC):
    """
    Abstract interface for password validation according to OWASP security standards.
    """
    
    @abstractmethod
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength against OWASP requirements.
        
        Args:
            password: The password to validate
            
        Returns:
            Dictionary with validation results including:
            - is_valid: Boolean indicating if password meets all requirements
            - errors: List of validation errors if any
            - score: Numeric score of password strength
        """
        pass
    
    @abstractmethod
    def hash_password(self, password: str) -> str:
        """
        Securely hash a password using strong algorithms (e.g., bcrypt, Argon2).
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Securely hashed password
        """
        pass
    
    @abstractmethod
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password to compare against
            
        Returns:
            True if password matches the hash, False otherwise
        """
        pass


class UserValidationInterface(ABC):
    """
    Abstract interface for user data validation and verification.
    """
    
    @abstractmethod
    async def validate_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate user registration/profile data.
        
        Args:
            user_data: Dictionary containing user data to validate
            
        Returns:
            Dictionary with validation results including:
            - is_valid: Boolean indicating if all data is valid
            - errors: Dictionary of field-specific validation errors
        """
        pass
    
    @abstractmethod
    async def verify_email(self, email: str, verification_token: str) -> bool:
        """
        Verify a user's email address using a verification token.
        
        Args:
            email: Email address to verify
            verification_token: Token sent to the user's email
            
        Returns:
            True if email was successfully verified, False otherwise
        """
        pass
    
    @abstractmethod
    async def is_email_taken(self, email: str) -> bool:
        """
        Check if an email address is already taken.
        
        Args:
            email: Email address to check
            
        Returns:
            True if email is taken, False otherwise
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
    async def generate_verification_token(self, user_id: str, purpose: str) -> str:
        """
        Generate a verification token for email verification or password reset.
        
        Args:
            user_id: User identifier
            purpose: Purpose of the token (e.g., 'email_verification', 'password_reset')
            
        Returns:
            Generated verification token
        """
        pass
    
    @abstractmethod
    async def check_user_exists(self, identifier: str, identifier_type: str = "email") -> bool:
        """
        Check if a user exists by a given identifier.
        
        Args:
            identifier: User identifier value (email, username, etc.)
            identifier_type: Type of identifier being checked
            
        Returns:
            True if user exists, False otherwise
        """
        pass


class AuthInterface(ABC):
    """
    Abstract interface for authentication services.
    Implements OWASP security best practices and JWT authentication.
    """
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with provided credentials.
        
        Args:
            credentials: Dictionary containing authentication credentials (username/email, password)
            
        Returns:
            User data if authentication is successful, None otherwise
        """
        pass
    
    @abstractmethod
    async def create_token(self, user_data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT token for an authenticated user.
        
        Args:
            user_data: User information to encode in the token
            expires_delta: Optional expiration time delta
            
        Returns:
            JWT token string
        """
        pass
    
    @abstractmethod
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token data if valid, None otherwise
        """
        pass
    
    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """
        Refresh an expired access token using a refresh token.
        
        Args:
            refresh_token: The refresh token
            
        Returns:
            Dictionary with new access and refresh tokens if valid, None otherwise
        """
        pass
    
    @abstractmethod
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """
        Change a user's password with verification of the old password.
        
        Args:
            user_id: User identifier
            old_password: Current password for verification
            new_password: New password to set
            
        Returns:
            True if password was changed successfully, False otherwise
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

