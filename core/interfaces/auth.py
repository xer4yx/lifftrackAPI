from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import timedelta

class TokenService(ABC):
    @abstractmethod
    async def create_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create an access token"""
        pass

    @abstractmethod
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a token"""
        pass

class PasswordService(ABC):
    @abstractmethod
    async def hash_password(self, password: str) -> str:
        """Hash a password"""
        pass

    @abstractmethod
    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        pass 
    
class InputValidator(ABC):
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate input data"""
        pass
