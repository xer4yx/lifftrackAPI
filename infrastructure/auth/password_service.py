from passlib.context import CryptContext
from core.interfaces.auth import PasswordService
from utilities.monitoring.factory import MonitoringFactory

logger = MonitoringFactory.get_logger("password-service")

class BcryptPasswordService(PasswordService):
    def __init__(self):
        self.context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    async def hash_password(self, password: str) -> str:
        return self.context.hash(password)

    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.context.verify(plain_password, hashed_password) 