from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum
from datetime import datetime, timezone
from .base import EntityBase, EntityDefaultBase
from .exercise import ExerciseType

class FormFeedbackType(Enum):
    POSTURE = "posture"
    RANGE_OF_MOTION = "range_of_motion"
    STABILITY = "stability"
    TEMPO = "tempo"
    ALIGNMENT = "alignment"
    
@dataclass
class FormAnalysisBase(EntityBase):
    """Base class for all form analysis entities"""
    exercise_set_id: str
    user_id: str
    keypoints: Dict[str, List[float]]
    joint_angles: Dict[str, float]
    movement_patterns: Dict[str, Any]
    stability_score: float
    form_score: float
    feedback: List[Dict[str, Any]]

@dataclass
class FormAnalysisDefaultBase(EntityDefaultBase):
    """Default values for form analysis entities"""
    pass

@dataclass
class FormAnalysis(FormAnalysisDefaultBase, FormAnalysisBase):
    """Entity representing exercise form analysis"""
    def add_feedback(
        self,
        feedback_type: FormFeedbackType,
        message: str,
        severity: float
    ) -> None:
        """Add form feedback"""
        self.feedback.append({
            "type": feedback_type.value,
            "message": message,
            "severity": severity
        })
        self.update_timestamp()
        
@dataclass
class ProgressTrackingBase(EntityBase):
    """Base class for all progress tracking entities"""
    user_id: str
    exercise_type: ExerciseType
    sessions: List[str]  # List of WorkoutSession IDs
    form_improvement: float
    weight_progression: List[Dict[str, Any]]
    personal_records: Dict[str, Any]
    
@dataclass
class ProgressTrackingDefaultBase(EntityDefaultBase):
    """Default values for progress tracking entities"""
    pass

@dataclass
class ProgressTracking(ProgressTrackingDefaultBase, ProgressTrackingBase):
    """Entity for tracking user's exercise progress"""
    def update_progress(
        self,
        session_id: str,
        form_score: float,
        weight: float
    ) -> None:
        """Update progress with new session data"""
        self.sessions.append(session_id)
        self.weight_progression.append({
            "date": datetime.now(timezone.utc),
            "weight": weight
        })
        # Update personal records if applicable
        if weight > self.personal_records.get("max_weight", 0):
            self.personal_records["max_weight"] = weight
            self.personal_records["date_achieved"] = datetime.now(timezone.utc)
        self.update_timestamp() 