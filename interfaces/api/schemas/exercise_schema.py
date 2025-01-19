from pydantic import BaseModel, Field
from typing import Optional, Dict, Union
from enum import auto, IntFlag
from datetime import datetime

class ExerciseType(IntFlag):
    shoulder_press = auto()
    bench_press = auto()
    deadlift = auto()
    rdl = auto()


class Object(BaseModel):
    x: Union[int, float]
    y: Union[int, float]
    width: Union[int, float]
    height: Union[int, float]
    confidence: Union[int, float]
    class_name: str = Field(alias="class")
    classs_id: Optional[int] = None


class Features(BaseModel):
    objects: Optional[Object]
    joint_angles: Dict[str, Union[int, float]]
    movement_pattern: str
    speeds: Dict[str, Union[int, float]]
    body_alignment: Dict[str, Union[int, float]]
    stability: float


class ExerciseData(BaseModel):
    seconds: str = datetime.now().strftime("%M:%S")
    suggestion: str
    features: Features


class ExerciseDate(BaseModel):
    date: str = datetime.now().strftime("%m-%d-%Y")
    
    def set_exercise_data(self, data: ExerciseData):
        setattr(self, self.date, data)


class Exercise(BaseModel):
    exercise_type: ExerciseType
    
    def set_exercise_date(self, exercise_type: ExerciseType, date: ExerciseDate):
        setattr(self, exercise_type.name, date)
