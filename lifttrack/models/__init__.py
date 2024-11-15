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


class Exercise(BaseModel):
    """
    Model for exercise data where the exercise name is the direct parent
    of the exercise data
    """
    rdl: Optional[ExerciseData] = None
    shoulder_press: Optional[ExerciseData] = None
    bench_press: Optional[ExerciseData] = None
    deadlift: Optional[ExerciseData] = None
    # Add other exercises as needed

    def set_exercise_data(self, exercise_name: str, data: ExerciseData):
        """Helper method to set exercise data for a specific exercise"""
        setattr(self, exercise_name.lower(), data)


class Progress(BaseModel):
    username: str
    exercise: Exercise
    