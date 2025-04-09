from pydantic import BaseModel

from .auth_schema import LoginForm, Token, TokenData
from .exercise_schema import *
from .progress_schema import Progress
from .user_schema import User, UserUpdate
from .stats_schema import Stats


class AppInfo(BaseModel):
    app_name: str = "LiftTrack"
    version: str = "1.0.0"
    description: str = "An app to track your lifts and provide feedback on your form."
    
__all__ = [
    "AppInfo",
    "ExerciseData",
    "Features",
    "Frame",
    "LoginForm",
    "Object",
    "Token",
    "TokenData",
    "User",
    "UserUpdate",
    "Progress",
    "Stats"
]
