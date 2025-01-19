from typing import List, Optional
from core.entities import (
    Exercise, 
    ExerciseFeature, 
    ExerciseType, 
    JointAngles, 
    BodyAlignment, 
    Object
)

from core.interfaces import ExerciseRepository, DatabaseRepository

class ExerciseService:
    def __init__(self, database: DatabaseRepository, exercise_repository: ExerciseRepository):
        self.database = database
        self.exercise_repository = exercise_repository
        
    def save_data_features(self, data: object) -> None:
        """Save exercise data to repository"""
        self.exercise_repository.save(data)
        
    def load_data_features(self, data_type: type) -> Optional[object]:
        """Load exercise data from repository"""
        return self.exercise_repository.load(data_type)
        
    def load_exercise_features(self) -> Optional[Exercise]:
        """Load all exercise features and create Exercise object"""
        exercise_feature = self.load_data_features(ExerciseFeature)
        if not exercise_feature:
            return None
            
        return Exercise(
            features=exercise_feature
        )
        
    def set_exercise(self, username: str, exercise_data: ExerciseFeature, suggestions):
        exercise = Exercise(
            username=username,
            exercise_type=None,
            suggestion=suggestions,
            features=exercise_data
        )