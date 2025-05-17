from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserEntity(BaseModel):
    """
    User entity model representing a user in the system.
    Uses Pydantic BaseModel for validation and serialization.
    """
    id: Optional[str] = Field(default=None, description="Unique identifier for the user")
    username: str = Field(..., min_length=3, max_length=50, description="User's username")
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="Securely hashed password")
    first_name: str = Field(..., description="User's first name")
    last_name: str = Field(..., description="User's last name")
    is_deleted: bool = Field(default=False, description="Whether the user account is deleted")
    is_verified: bool = Field(default=False, description="Whether the user's email is verified")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the user was created")
    updated_at: Optional[datetime] = Field(default=None, description="When the user was last updated")
    last_login: Optional[datetime] = Field(default=None, description="When the user last logged in")

    class Config:
        """Pydantic model configuration"""
        validate_assignment = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
