from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends
from jose import JWTError, jwt

from core.interfaces.auth import TokenService
from core.exceptions.auth import TokenError
from utilities.monitoring.factory import MonitoringFactory
from utilities.config import get_security_settings
from utilities.validators import SecurityConfig

logger = MonitoringFactory.get_logger("jwt-auth")

class JWTTokenService(TokenService):
    def __init__(self, security_settings: SecurityConfig = Depends(get_security_settings)):
        self.secret_key = security_settings.JWT_SECRET_KEY
        self.algorithm = security_settings.JWT_ALGORITHM
        self.access_token_expire_minutes = security_settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES

    async def create_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        try:
            token = await self._generate_token(data, expires_delta)
            logger.info(f"Token created for user: {data.get('sub')}")
            return token
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise

    async def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.error(f"Token verification error: {e}")
            raise TokenError(f"Invalid token: {e}") 