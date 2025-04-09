from typing import Optional
from pydantic import BaseModel, Field


class Stats(BaseModel):
    body_alignment_score: Optional[float] = Field(default=0.0, title="Body Alignment Score")
    joint_angle_consistency: Optional[float] = Field(default=0.0, title="Joint Angle Consistency")
    load_control: Optional[float] = Field(default=0.0, title="Load Control")
    speed_control: Optional[float] = Field(default=0.0, title="Speed Control")
    overall_stability: Optional[float] = Field(default=0.0, title="Overall Stability")
    