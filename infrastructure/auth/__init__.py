from .jwt_service import JWTTokenService
from .password_service import BcryptPasswordService
from .input_validator import DataValidator

__all__ = [
    "JWTTokenService",
    "BcryptPasswordService",
    "DataValidator"
]

def get_token_service() -> JWTTokenService:
    return JWTTokenService()

def get_password_service() -> BcryptPasswordService:
    return BcryptPasswordService()

def get_data_validator() -> DataValidator:
    return DataValidator()