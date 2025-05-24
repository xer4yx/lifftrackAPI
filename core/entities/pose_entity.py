from pydantic import BaseModel, Field, AliasChoices
from typing import Dict, Tuple, List, Any, Optional


class Keypoint(BaseModel):
    x: float = Field(..., serialization_alias="x")
    y: float = Field(..., serialization_alias="y")
    confidence: float = Field(..., serialization_alias="confidence")
    
    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.confidence)


class Object(BaseModel):
    x: float = Field(..., serialization_alias="x")
    y: float = Field(..., serialization_alias="y")
    width: float = Field(..., serialization_alias="width")
    height: float = Field(..., serialization_alias="height")
    confidence: float = Field(..., serialization_alias="confidence")
    type: str = Field(
        default=..., 
        alias="class",
        serialization_alias="class",
        validation_alias=AliasChoices("class", "type"), 
        populate_by_name=True)
    classs_id: Optional[int] = Field(default=0, alias="class_id")


class KeypointCollection(BaseModel):
    keypoints: Dict[str, Keypoint] = Field(default_factory=dict)
    
    def get(self, keypoint_name: str) -> Optional[Keypoint]:
        return self.keypoints.get(keypoint_name)


class JointAngle(BaseModel):
    name: str
    value: float


class BodyAlignment(BaseModel):
    vertical_alignment: float = Field(..., alias="0", 
                                      serialization_alias="vertical_alignment", 
                                      validation_alias=AliasChoices("0", "vertical_alignment"))
    lateral_alignment: float = Field(..., alias="1", 
                                     serialization_alias="lateral_alignment", 
                                     validation_alias=AliasChoices("1", "lateral_alignment"))


class PoseFeatures(BaseModel):
    keypoints: KeypointCollection = Field(..., serialization_alias="keypoints")
    joint_angles: Dict[str, float] = Field(default_factory=dict, serialization_alias="joint_angles")
    movement_patterns: Dict[str, float] = Field(default_factory=dict, serialization_alias="movement_patterns")
    movement_pattern: str = Field(default="", serialization_alias="movement_pattern")
    body_alignment: Optional[BodyAlignment] = Field(default=None, 
                                                    serialization_alias="body_alignment")
    stability: float = Field(default=0.0, serialization_alias="stability")
    speeds: Dict[str, float] = Field(default_factory=dict, serialization_alias="speeds")
    form_issues: Dict[str, bool] = Field(default_factory=dict, serialization_alias="form_issues")
    objects: Dict[str, Any] = Field(default_factory=dict, serialization_alias="objects")
    
    def model_dump(self, **kwargs):
        """Override model_dump to customize keypoints serialization."""
        data = super().model_dump(**kwargs)
        # Replace the nested keypoints structure with direct keypoints
        if 'keypoints' in data and isinstance(data['keypoints'], dict) and 'keypoints' in data['keypoints']:
            data['keypoints'] = data['keypoints']['keypoints']
        return data


class FormAnalysis(BaseModel):
    accuracy: float = Field(..., serialization_alias="accuracy")
    suggestions: List[str] = Field(..., serialization_alias="suggestions")


class ExerciseData(BaseModel):
    """
    Model for storing exercise data with associated features and form suggestions.
    """
    frame: str = Field(..., serialization_alias="frame")
    suggestion: str = Field(..., serialization_alias="suggestion")
    features: PoseFeatures = Field(..., serialization_alias="features")
    frame_id: Optional[str] = Field(default=None, serialization_alias="frame_id")
