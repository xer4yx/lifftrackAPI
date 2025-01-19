"""
Entities
- Represents the fundamental business objects
- Contains pure domain models that define the core data structures
- Independent of any external frameworks or implementation details
"""

from .base import EntityBase
from .user import User
from .exercise import Exercise, ExerciseType, ExerciseFeature, Object, BodyAlignment, JointAngles
from .monitoring import Metric
# from .analysis import FormAnalysis, ProgressTracking, FormFeedbackType

__all__ = [
    'EntityBase',
    'User',
    'Exercise',
    'ExerciseType',
    'ExerciseFeature',
    'Object',
    'BodyAlignment',
    'JointAngles',
    'Metric',
    # 'FormAnalysis',
    # 'ProgressTracking',
    # 'FormFeedbackType'
]
