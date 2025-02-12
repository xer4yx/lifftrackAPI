import re
from datetime import datetime
from typing import Optional, Union, Dict, Any

from pydantic import BaseModel, Field, model_validator


class User(BaseModel):
    id: Optional[str] = datetime.strftime(datetime.now(), '%Y%H%d%m')
    fname: str
    lname: str
    username: str
    phoneNum: str
    email: str
    password: str
    pfp: Optional[str] = None
    isAuthenticated: Optional[bool] = False
    isDeleted: Optional[bool] = False


class LoginForm(BaseModel):
    username: str
    password: str


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


class Object(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    type: str = Field(
        default="barbell", 
        alias="class", 
        validation_alias="class", 
        populate_by_name=True)
    classs_id: Optional[int] = Field(default=0, alias="class_id")

    class Config:
        populate_by_name = True


class Features(BaseModel):
    # objects: Optional[Dict[str, Object]] = Field(default_factory=dict)  # Make objects optional
    objects: Optional[Object]
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
    """
    Model for storing user progress data
    """
    data: Dict[str, Dict[str, Dict[str, ExerciseData]]] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def transform_data(cls, values):
        # If the input is already in the correct format, return as is
        if "data" in values:
            return values
            
        # Otherwise, wrap the input data in the expected structure
        return {"data": values}

    class Config:
        extra = "allow"  # Allow extra fields in the data 

__all__ = [
    "User",
    "LoginForm",
    "Token",
    "TokenData",
    "AppInfo",
    "Frame",
    "Object",
    "Features",
    "ExerciseData",
    "Exercise",
    "Progress"
]
