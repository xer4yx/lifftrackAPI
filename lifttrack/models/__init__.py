import re
from datetime import datetime
from typing import Optional, Union, Dict, Any

from pydantic import BaseModel, validator


class User(BaseModel):
    id: str = datetime.strftime(datetime.now(), '%Y%H%d%m')
    fname: str
    lname: str
    username: str
    phoneNum: str
    email: str
    password: str
    pfp: Optional[str] = None
    isAuthenticated: bool = False
    isDeleted: bool = False


class LoginForm(BaseModel):
    username: str
    password: str

    @validator('username')
    def validate_username(cls, v):
        if not v:
            raise ValueError("Username cannot be empty.")
        return v

    @validator('password')
    def validate_password_not_empty(cls, v):
        if not v:
            raise ValueError("Password cannot be empty.")
        return v


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class AppInfo(BaseModel):
    app_name: str = "LiftTrack"
    version: str = "1.0.0"
    description: str = "An app to track your lifts and provide feedback on your form."


class Frame(BaseModel):
    user: str
    original_frame: Union[int, int]
    image: bytes


class Features(BaseModel):
    objects: str
    joint_angles: Dict[str, Any]
    movement_pattern: str
    speeds: dict
    body_alignment: Any
    stability: float


class ExerciseData(BaseModel):
    date: str = datetime.now().isoformat()
    suggestion: str
    features: Features
    frame: str


class Progress(BaseModel):
    username: str
    exercise: Dict[str, Dict[str, ExerciseData]]  # exercise name -> date -> exercise data