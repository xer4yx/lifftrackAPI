import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from lifttrack import config
from lifttrack.dbhandler.rest_rtdb import rtdb
from lifttrack.models import User, TokenData
from lifttrack.utils.logging_config import setup_logger

from core.interfaces import TokenService, PasswordService, InputValidator, DatabaseRepository


# Configure logging for auth.py
logger = setup_logger("auth", "lifttrack_auth.log")

# Secret Configuration
SECRET_KEY = config.get(section='Authentication', option='SECRET_KEY')
ALGORITHM = config.get(section='Authentication', option='ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = int(config.get(section='Authentication', option='TTL'))

# Password Hashing
hash_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

existing_usernames = set()
try:
    users = rtdb.get_all_data()
    existing_usernames.update(username for username in users.keys())
    logger.info(f"Username cache initialized with {len(existing_usernames)} entries")
except Exception as e:
    logger.error(f"Failed to initialize username cache: {e}")
    
    
# Add these helper functions to manage the cache
def add_to_username_cache(username: str):
    """Add a username to the cache."""
    existing_usernames.add(username)


def remove_from_username_cache(username: str):
    """Remove a username from the cache."""
    existing_usernames.discard(username)  # Using discard instead of remove to avoid KeyError


def validate_input(data: Dict[str, Any]) -> bool:
    """Validates user input and returns True if valid, False otherwise."""
    password_pattern = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$])[A-Za-z\d@$]{8,12}$'
    mobileno_pattern = r'^(?:\+63\d{10}|09\d{9})$'
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    
    # Extract values from dictionary
    username = data.get('username')
    password = data.get('password')
    phone_number = data.get('phone_number')
    email = data.get('email')
    
    if not username or not password or not phone_number or not email:
        logger.error("Missing required fields in user data")
        return False
    
    # Check for existing username
    if username in existing_usernames:
        logger.info(f"Attempt to create duplicate username: {username}")
        return False

    # Validate password format
    if re.match(password_pattern, password) is None:
        logger.error(f"Invalid password format for user: {username}")
        return False

    # Validate phone number format
    if re.match(mobileno_pattern, phone_number) is None:
        logger.error(f"Invalid phone number format for user: {username}")
        return False

    # Validate email format
    if re.match(email_pattern, email) is None:
        logger.error(f"Invalid email format for user: {username}")
        return False

    logger.info(f"User data validated successfully: {username}")
    return True


def verify_password(plain_password, hashed_password):
    is_valid = hash_context.verify(plain_password, hashed_password)
    logger.debug(f"Password verification for user: {hashed_password} returned {is_valid}")
    return is_valid


def get_password_hash(password):
    hashed = hash_context.hash(password)
    logger.debug("Password hashed successfully.")
    return hashed


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"Access token created for user: {data.get('sub')}")
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_data = TokenData(username=username)
    except JWTError as e:
        logger.error(f"JWT decoding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_data = rtdb.get_data(username=token_data.username)
    if user_data is None:
        logger.warning(f"User not found for token: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Convert dictionary to User model
        user = User(
            username=user_data.get("username"),
            first_name=user_data.get("first_name"),
            last_name=user_data.get("last_name"),
            phone_number=user_data.get("phone_number"),
            email=user_data.get("email"),
            password=user_data.get("password"),
            profile_picture=user_data.get("profile_picture"),
            is_authenticated=user_data.get("is_authenticated", False),
            is_deleted=user_data.get("is_deleted", False)
        )
        logger.info(f"User retrieved successfully: {username}")
        return user
    except Exception as e:
        logger.error(f"Error converting user data to User model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing user data"
        )
        

async def check_password_update(user: User = Depends(lambda user: validate_input(user, is_update=True))):
    """
    Dependency to check if password is being updated in user data.
    Returns tuple of (user, is_password_update).
    """
    existing_user_data = rtdb.get_data(user.username)
    if not existing_user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    is_password_update = existing_user_data.get("password") != user.password
    return user, is_password_update


def verify_and_update_password(plain_password, hashed_password):
    verified, updated = hash_context.verify_and_update(secret=plain_password, hash=hashed_password)
    
    if updated is not None:
        logger.info(f"Hash scheme updated for {hash_context.default_scheme()}")
        
    return verified

class LiftTrackAuthenticator(TokenService, PasswordService, InputValidator):
    def __init__(self, database: DatabaseRepository):
        self.database = database
        self._validation_errors = []
        
    def create_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        return create_access_token(data, expires_delta)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    
    def hash_password(self, password: str) -> str:
        return get_password_hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return verify_password(plain_password, hashed_password)
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validates user input and returns True if valid, False otherwise."""
        self._validation_errors = []  # Reset errors before validation
        
        password_pattern = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$])[A-Za-z\d@$]{8,12}$'
        mobileno_pattern = r'^(?:\+63\d{10}|09\d{9})$'
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        
        # Extract values from dictionary
        username = data.get('username')
        password = data.get('password')
        phone_number = data.get('phone_number')
        email = data.get('email')
        
        if not username or not password or not phone_number or not email:
            self._validation_errors.append("Missing required fields in user data")
            return False
        
        # Check for existing username
        if username in existing_usernames:
            self._validation_errors.append(f"Username '{username}' already exists")
            return False

        # Validate password format
        if not re.match(password_pattern, password):
            self._validation_errors.append(
                "Password must be 8-12 characters long and contain at least "
                "one uppercase letter, one number, and one special character (@$)"
            )
            return False

        # Validate phone number format
        if not re.match(mobileno_pattern, phone_number):
            self._validation_errors.append(
                "Phone number must be in format: +63XXXXXXXXXX or 09XXXXXXXXX"
            )
            return False

        # Validate email format
        if not re.match(email_pattern, email):
            self._validation_errors.append("Invalid email format")
            return False

        logger.info(f"User data validated successfully: {username}")
        return True
    
    def get_validation_errors(self) -> list:
        """Returns the list of validation errors from the last validation attempt."""
        return self._validation_errors

