from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from core.interfaces.auth import TokenService, PasswordService
from infrastructure.auth.jwt_service import JWTTokenService
from infrastructure.auth.password_service import BcryptPasswordService
from utilities.monitoring.logger import MonitoringService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_token_service() -> TokenService:
    return JWTTokenService()

def get_password_service() -> PasswordService:
    return BcryptPasswordService()

def get_monitoring_service() -> MonitoringService:
    return MonitoringService("lifttrack")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    token_service: TokenService = Depends(get_token_service)
):
    payload = await token_service.verify_token(token)
    return payload 