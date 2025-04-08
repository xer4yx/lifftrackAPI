from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


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
    frame_id: Optional[str]


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