from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional
from enum import Enum

class ExerciseType(str, Enum):
    BENCHPRESS = "benchpress"
    DEADLIFT = "deadlift"
    SHOULDERPRESS = "shoulderpress"
    RDL = "rdl"

class Keypoint(BaseModel):
    x: int
    y: int
    confidence: float

class DetectedObject(BaseModel):
    class_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int

class JointAngles(BaseModel):
    left_shoulder_elbow_wrist: Optional[float] = None
    right_shoulder_elbow_wrist: Optional[float] = None
    left_hip_knee_ankle: Optional[float] = None
    right_hip_knee_ankle: Optional[float] = None
    # Add other relevant angles

class MovementFeatures(BaseModel):
    joint_angles: JointAngles
    body_alignment: Dict[str, float]
    movement_patterns: Optional[Dict[str, float]] = None
    speeds: Optional[Dict[str, float]] = None
    stability: Optional[float] = None

class FrameAnalysis(BaseModel):
    keypoints: Dict[str, Keypoint]
    detected_objects: List[DetectedObject]
    features: MovementFeatures
    exercise_type: Optional[ExerciseType] = None
    exercise_confidence: Optional[float] = None
