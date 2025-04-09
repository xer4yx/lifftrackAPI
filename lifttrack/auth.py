import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from lifttrack import config
from lifttrack.dbhandler.rest_rtdb import RTDBHelper
from lifttrack.models import User, TokenData
from lifttrack.utils.logging_config import setup_logger
from routers.manager import HTTPConnectionPool


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

# Initialize username cache
existing_usernames = set()

async def initialize_username_cache():
    """Initialize the username cache asynchronously."""
    try:
        async with RTDBHelper() as rtdb:
            users = await rtdb.get_all_data()
            if users:
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


def validate_input(user: User, is_update: bool = False):
    """Validates user input and returns the validated user object."""
    password_pattern = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$])[A-Za-z\d@$]{8,12}$'
    mobileno_pattern = r'^(?:\+63\d{10}|09\d{9})$'
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    
    if not is_update:
        if user.username in existing_usernames:
            logger.info(f"Attempt to create duplicate username: {user.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists."
            )

    if re.match(password_pattern, user.password) is None:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Password must be 8-12 characters long, with at least one uppercase letter, one digit, and one special character."
        )

    if re.match(mobileno_pattern, user.phoneNum) is None:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Invalid mobile number."
        )

    if re.match(email_pattern, user.email) is None:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Invalid email address."
        )

    logger.info(f"{user.username} updated profile information.")
    return user  # Return the validated user object


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
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"Access token created for user: {data.get('sub')}")
    return encoded_jwt


def validate_token(token: str, username: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") != username:
            return False
        return True
    except JWTError as e:
        logger.error(f"JWT decoding error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error validating token: {str(e)}")
        return False


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
    async with HTTPConnectionPool.get_session() as session:
        async with RTDBHelper(session) as rtdb:
            user_data = await rtdb.get_data(username=token_data.username)
            if user_data is None:
                logger.warning(f"User not found for token: {username}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            try:
                # Convert dictionary to User model
                user = User.model_validate(user_data)
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
    async with HTTPConnectionPool.get_session() as session:
        async with RTDBHelper(session) as rtdb:
            existing_user_data = await rtdb.get_data(user.username)
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