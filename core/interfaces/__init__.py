"""
Interfaces
- Defines abstract interfaces and contracts
- Creates clear boundaries between different components
- Enables dependency inversion and easier testing
"""

from .auth import TokenService, PasswordService, InputValidator
from .database import DatabaseRepository
from .monitoring import MetricsExporter
from .exercise import ExerciseRepository
from .vision import ModelInference
from .repositories import FrameRepository

__all__ = [
    'TokenService',
    'PasswordService',
    'InputValidator',
    'DatabaseRepository',
    'MetricsExporter',
    'ModelInference',
    'FrameRepository',
    'ExerciseRepository'
]

def get_database_repository() -> DatabaseRepository:
    return DatabaseRepository()

def get_token_service() -> TokenService:
    return TokenService()

def get_password_service() -> PasswordService:
    return PasswordService()

def get_input_validator() -> InputValidator:
    return InputValidator()