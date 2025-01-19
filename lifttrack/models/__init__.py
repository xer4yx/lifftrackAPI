from datetime import datetime
from typing import Optional, Union, Dict, Any

from pydantic import BaseModel, ConfigDict, Field
from core.entities.user import User as UserEntity

class User(BaseModel):
    """
    User base model for the application.
    """
    model_config = ConfigDict(
        extra="allow"
    )
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    profile_picture: Optional[str] = None
    is_authenticated: Optional[bool] = False
    is_deleted: Optional[bool] = False

    def to_entity(self) -> UserEntity:
        """Convert Pydantic User model to User entity"""
        entity_data = {}
        for field, value in self.model_dump(exclude_unset=True).items():
            if value is not None:
                entity_data[field] = value
        return UserEntity(**entity_data)

    @classmethod
    def from_entity(cls, entity: UserEntity) -> 'User':
        """Create Pydantic User model from User entity"""
        return cls(
            username=entity.username,
            email=entity.email,
            password=entity.password,
            first_name=entity.first_name,
            last_name=entity.last_name,
            phone_number=entity.phone_number,
            profile_picture=entity.profile_picture,
            is_authenticated=entity.is_authenticated,
            is_deleted=entity.is_deleted
        )


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
    type: str = Field(alias="class")
    classs_id: Optional[int] = None


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
    username: str
    exercise: Exercise
    