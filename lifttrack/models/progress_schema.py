from typing import Dict
from pydantic import BaseModel, Field, model_validator
from .exercise_schema import ExerciseData


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
    