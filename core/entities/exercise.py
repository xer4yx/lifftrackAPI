from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from enum import Enum

class ExerciseType(Enum):
    BENCHPRESS = "benchpress"
    DEADLIFT = "deadlift"
    SHOULDERPRESS = "shoulderpress"
    RDL = "rdl"
    
@dataclass
class BodyAlignment:
    """Body alignment entity"""
    vertical_alignment: int | float
    horizontal_alignment: int | float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert body alignment entity to a python dictionary"""
        return asdict(self)
    
@dataclass
class JointAngles:
    """Joint angles entity"""
    left_sew: int | float
    left_hka: int | float
    left_shk: int | float
    right_sew: int | float
    right_hka: int | float
    right_shk: int | float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert joint angles entity to a python dictionary"""
        return asdict(self)
    
@dataclass
class Object:
    """Object entity"""
    object_type: str
    object_height: int | float
    object_width: int | float
    object_x_coordinate: int | float
    object_y_coordinate: int | float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object entity to a python dictionary"""
        return asdict(self)
    
@dataclass
class ExerciseFeature:
    """Exercise feature entity"""
    body_alignment: Optional[BodyAlignment]
    joint_angles: Optional[JointAngles]
    movement_pattern: Optional[List[str]]
    objects: Optional[Object]
    movement_speed: Optional[int | float]
    stability: Optional[int | float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exercise feature entity to a python dictionary"""
        return asdict(self)

@dataclass
class Exercise:
    """Exercise entity"""
    username: Optional[str]
    exercise_type: Optional[ExerciseType]
    date_performed: Optional[str] = field(default_factory=datetime.now(timezone.utc).strftime("%d-%m-%Y"))
    time_frame: Optional[str] = field(default_factory=datetime.now(timezone.utc).strftime("%H:%M:%S"))
    suggestion: Optional[List[str]] = None
    features: Optional[ExerciseFeature] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exercise entity to a python dictionary"""
        return asdict(self)
    

