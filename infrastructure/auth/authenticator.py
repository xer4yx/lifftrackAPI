import re
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from lifttrack.utils.logging_config import setup_logger
from core.interface import AuthenticationInterface, TokenBlacklistRepository
from core.entities import (
    UserEntity,
    TokenEntity, 
    TokenDataEntity, 
    CredentialsEntity,
    ValidationResultEntity
)


class Authenticator(AuthenticationInterface):
    """
    Implementation of the AuthenticationService interface.
    """
    
    def __init__(
        self, 
        secret_key: str,
        algorithm: str,
        access_token_expire_minutes: int,
        user_repository,  # Will be injected, avoid circular imports
        token_blacklist_repository: TokenBlacklistRepository = None  # Optional, can be implemented later
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.user_repository = user_repository
        self.token_blacklist_repository = token_blacklist_repository
        self.password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.logger = setup_logger("infrastructure.auth", "auth.log")
        
        # Initialize username cache if needed
        self.username_cache = set()
        self.initialize_username_cache()
        
    async def initialize_username_cache(self):
        """Initialize the username cache from the user repository."""
        try:
            users = await self.user_repository.get_all_users()
            if users:
                self.username_cache = {user.username for user in users}
                self.logger.info(f"Username cache initialized with {len(self.username_cache)} entries")
        except Exception as e:
            self.logger.error(f"Failed to initialize username cache: {e}")
    
    async def validate_password_strength(self, password: str) -> ValidationResultEntity:
        """
        Validate password strength against OWASP security standards.
        """
        errors = []
        
        # Password pattern based on requirements
        password_pattern = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$])[A-Za-z\d@$]{8,12}$'
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        elif len(password) > 12:
            errors.append("Password must not exceed 12 characters")
            
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
            
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
            
        if not re.search(r'[@$]', password):
            errors.append("Password must contain at least one special character (@, $)")
            
        is_valid = len(errors) == 0
        
        return ValidationResultEntity(is_valid=is_valid, errors=errors)
    
    async def hash_password(self, password: str) -> str:
        """
        Securely hash a password using bcrypt.
        """
        return self.password_context.hash(password)
    
    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        """
        is_valid = self.password_context.verify(plain_password, hashed_password)
        self.logger.debug(f"Password verification returned {is_valid}")
        return is_valid
    
    async def authenticate(self, credentials: CredentialsEntity) -> Optional[UserEntity]:
        """
        Authenticate a user with provided credentials.
        """
        try:
            user = await self.user_repository.get_user_by_username(credentials.username)
            if not user:
                return None
                
            if not credentials.password:  # For token validation when password not needed
                return user
                
            if not await self.verify_password(credentials.password, user.password):
                return None
                
            return user
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None
    
    async def create_token(self, user_data: UserEntity, expires_delta: Optional[timedelta] = None) -> TokenEntity:
        """
        Create a JWT token for an authenticated user.
        """
        to_encode = {"sub": user_data.username}
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
            
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        self.logger.info(f"Access token created for user: {user_data.username}")
        
        return TokenEntity(
            access_token=encoded_jwt,
            token_type="bearer",
            expires_at=expire
        )
    
    async def verify_token(self, token: str) -> Optional[TokenDataEntity]:
        """
        Verify and decode a JWT token.
        """
        try:
            # Check if token is blacklisted
            if self.token_blacklist_repository and await self.token_blacklist_repository.is_blacklisted(token):
                self.logger.warning(f"Attempt to use blacklisted token")
                return None
                
            # Decode the token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username = payload.get("sub")
            exp = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)
            
            if not username:
                return None
                
            return TokenDataEntity(username=username, exp=exp)
            
        except JWTError as e:
            self.logger.error(f"JWT verification error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during token verification: {e}")
            return None
    
    async def refresh_token(self, refresh_token: str) -> Optional[TokenEntity]:
        """
        Refresh an expired access token using a refresh token.
        """
        # For now, this is a simple implementation
        # In a production system, this would use a separate refresh token mechanism
        token_data = await self.verify_token(refresh_token)
        if not token_data:
            return None
            
        user = await self.user_repository.get_user_by_username(token_data.username)
        if not user:
            return None
            
        # Create a new token with a fresh expiry time
        # Adding a small offset to ensure the token is different
        custom_expiry = timedelta(minutes=self.access_token_expire_minutes + 1)
        return await self.create_token(user, custom_expiry)
    
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> Tuple[bool, Optional[str]]:
        """
        Change a user's password with verification of the old password.
        """
        try:
            # Get user
            user = await self.user_repository.get_user_by_id(user_id)
            if not user:
                return False, "User not found"
                
            # Verify old password
            if not await self.verify_password(old_password, user.password):
                return False, "Current password is incorrect"
                
            # Validate new password
            validation = await self.validate_password_strength(new_password)
            if not validation.is_valid:
                return False, "; ".join(validation.errors)
                
            # Hash and update password
            hashed_password = await self.hash_password(new_password)
            success = await self.user_repository.update_password(user_id, hashed_password)
            
            if not success:
                return False, "Failed to update password"
                
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error changing password: {e}")
            return False, "An unexpected error occurred"
    
    async def logout(self, token: str) -> bool:
        """
        Invalidate a user's token (add to blacklist).
        """
        try:
            # If no blacklist repository is configured, simply return True
            if not self.token_blacklist_repository:
                return True
                
            # Verify token before blacklisting
            token_data = await self.verify_token(token)
            if not token_data:
                return False
                
            # Add to blacklist
            return await self.token_blacklist_repository.add_to_blacklist(token, token_data.exp)
            
        except Exception as e:
            self.logger.error(f"Error during logout: {e}")
            return False
    
    async def is_username_taken(self, username: str) -> bool:
        """
        Check if a username is already taken using cache for performance.
        """
        try:
            # First check the cache for the username
            if username in self.username_cache:
                return True
                
            # If not in cache, check directly with database for registration
            exists = await self.user_repository.check_username_exists(username)
            
            # Only if found, add to cache
            if exists:
                self.username_cache.add(username)
            
            return exists
        except Exception as e:
            self.logger.error(f"Error checking if username is taken: {e}")
            # In case of error, default to false to allow registration attempt
            # The database constraints will still prevent duplicates
            return False
    
    async def validate_user_input(self, user_data: Dict[str, Any], is_update: bool = False) -> ValidationResultEntity:
        """
        Validates user input data.
        """
        errors = []
        
        # Define patterns for validation
        password_pattern = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$])[A-Za-z\d@$]{8,12}$'
        mobileno_pattern = r'^(?:\+63\d{10}|09\d{9})$'
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        
        # Username validation (only for new users)
        if not is_update and user_data.get("username"):
            if await self.is_username_taken(user_data["username"]):
                errors.append("Username already exists")
                
        # Password validation
        if password := user_data.get("password"):
            if re.match(password_pattern, password) is None:
                errors.append("Password must be 8-12 characters long, with at least one uppercase letter, one digit, and one special character (@, $)")
                
        # Phone number validation
        if phone := user_data.get("phoneNum") or user_data.get("phone_number"):
            if re.match(mobileno_pattern, phone) is None:
                errors.append("Invalid mobile number format")
                
        # Email validation
        if email := user_data.get("email"):
            if re.match(email_pattern, email) is None:
                errors.append("Invalid email address format")
                
        return ValidationResultEntity(
            is_valid=len(errors) == 0,
            errors=errors
        ) 