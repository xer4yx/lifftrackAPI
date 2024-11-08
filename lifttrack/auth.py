import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from lifttrack import config
from lifttrack.dbhandler.rtdbHelper import rtdb
from lifttrack.models import User, TokenData


# Logging Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('lifttrack_auth.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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


def validate_input(user: User):
    """Validates user input and returns the validated user object."""
    password_pattern = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$])[A-Za-z\d@$]{8,12}$'
    mobileno_pattern = r'^(?:\+63\d{10}|09\d{9})$'
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'

    if re.match(password_pattern, user.password) is None:
        logger.warning(f"Password validation failed for user: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Password must be 8-12 characters long, with at least one uppercase letter, one digit, and one special character."
        )

    if re.match(mobileno_pattern, user.phoneNum) is None:
        logger.warning(f"Mobile number validation failed for user: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Invalid mobile number."
        )

    if re.match(email_pattern, user.email) is None:
        logger.warning(f"Email validation failed for user: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Invalid email address."
        )
    
    # Check if user already exists
    existing_user = rtdb.get_data(user.username)
    if existing_user:
        logger.info(f"Attempt to create duplicate username: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists."
        )

    logger.info(f"User input validated successfully for user: {user.username}")
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


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Token payload missing 'sub'")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(username=username)
    except JWTError as e:
        logger.error(f"JWT decoding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = rtdb.get_data(username=token_data.username)
    if user is None:
        logger.warning(f"User not found for token: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info(f"User retrieved successfully: {username}")
    return user
