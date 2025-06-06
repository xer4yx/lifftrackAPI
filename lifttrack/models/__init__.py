from pydantic import BaseModel

from .auth_schema import LoginForm, Token, TokenData, TokenBlacklist
from .exercise_schema import *
from .progress_schema import Progress
from .user_schema import User, UserUpdate, UserResponse
from .stats_schema import Stats


class AppInfo(BaseModel):
    app_name: str = "LiftTrack"
    version: str = "1.0.0"
    description: str = "An app to track your lifts and provide feedback on your form."


class AppUpdate(BaseModel):
    current_version: str
    latest_version: str
    update_available: bool
    update_message: str
    download_url: str
    login_message: str

__all__ = [
    "AppInfo",
    "ExerciseData",
    "Features",
    "Frame",
    "LoginForm",
    "Object",
    "Token",
    "TokenData",
    "TokenBlacklist",
    "User",
    "UserUpdate",
    "UserResponse",
    "Progress",
    "Stats"
]
