from lifttrack.dbhandler import rtdb
from lifttrack import datetime, timedelta, Optional, config
from lifttrack.models import TokenData

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = config.get(section='Authentication', option='SECRET_KEY')
ALGORITHM = config.get(section='Authentication', option='ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = int(config.get(section='Authentication', option='TTL'))

hash_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return hash_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return hash_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = rtdb.get_data(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user
