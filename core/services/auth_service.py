from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from fastapi import Depends

from core.interfaces.auth import TokenService, PasswordService
from core.interfaces.database import DatabaseRepository
from core.exceptions.auth import AuthenticationError
from utilities.monitoring.factory import MonitoringFactory

logger = MonitoringFactory.get_logger("auth-service")

class AuthService:
    def __init__(
        self,
        database: DatabaseRepository,
        token_service: TokenService,
        password_service: PasswordService
    ):
        self.database = database
        self.token_service = token_service
        self.password_service = password_service
        
    async def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Dict[str, Any]:
        try:
            # Get user from database
            user = await self.database.get(
                path="users",
                key=username
            )
            
            if not user:
                logger.warning(f"User not found: {username}")
                raise AuthenticationError("Invalid credentials")
                
            user = user[0]  # Get first user from query results
            
            # Verify password
            if not await self.password_service.verify_password(
                password,
                user["password_hash"]
            ):
                logger.warning(f"Invalid password for user: {username}")
                raise AuthenticationError("Invalid credentials")
            
            # Generate token
            token_data = {
                "sub": user["id"],
                "username": username,
                "roles": user.get("roles", [])
            }
            
            expires_delta = timedelta(minutes=30)  # Token expires in 30 minutes
            access_token = await self.token_service.create_token(
                token_data,
                expires_delta
            )
            
            # Update last login
            await self.database.update(
                path="users",
                key=user["id"],
                data={"last_login": datetime.now(timezone.utc).isoformat()}
            )
            
            return {
                "access_token": access_token,
                "expires_at": datetime.utcnow() + expires_delta,
                "user_id": user["id"]
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise

    async def logout_user(self, user_id: str) -> None:
        try:
            # Update user's last activity
            await self.database.update(
                path="users",
                key=user_id,
                data={"last_logout": datetime.now(timezone.utc).isoformat()}
            )
            
            # Note: For JWT, we don't actually invalidate the token
            # In a production system, you might want to add the token
            # to a blacklist or use Redis to track invalidated tokens
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
            raise 
