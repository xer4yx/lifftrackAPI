from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, AliasChoices


class UserEntity(BaseModel):
    """
    User entity model representing a user in the system.
    Uses Pydantic BaseModel for validation and serialization.
    """

    id: Optional[str] = Field(
        default=None, description="Unique identifier for the user"
    )
    username: str = Field(
        ..., min_length=3, max_length=50, description="User's username"
    )
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="Securely hashed password")
    first_name: str = Field(
        ...,
        alias="fname",
        validation_alias=AliasChoices("fname", "first_name"),
        serialization_alias="fname",
        description="User's first name",
    )
    last_name: str = Field(
        ...,
        alias="lname",
        validation_alias=AliasChoices("lname", "last_name"),
        serialization_alias="lname",
        description="User's last name",
    )
    phone_number: Optional[str] = Field(
        ...,
        alias="phoneNum",
        validation_alias=AliasChoices("phoneNum", "phone_number"),
        serialization_alias="phoneNum",
        description="User's phone number",
    )
    profile_picture: Optional[str] = Field(
        default=None,
        alias="pfp",
        validation_alias=AliasChoices("pfp", "profile_picture"),
        serialization_alias="pfp",
        description="URL to user's profile picture",
    )
    is_deleted: bool = Field(
        default=False,
        alias="isDeleted",
        validation_alias=AliasChoices("isDeleted", "is_deleted"),
        serialization_alias="isDeleted",
        description="Whether the user account is deleted",
    )
    is_authenticated: bool = Field(
        default=False,
        alias="isAuthenticated",
        validation_alias=AliasChoices("isAuthenticated", "is_verified"),
        serialization_alias="isAuthenticated",
        description="Whether the user's email is verified",
    )
    created_at: datetime = Field(
        default=datetime.now(tz=timezone.utc), description="When the user was created"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="When the user was last updated"
    )
    last_login: Optional[datetime] = Field(
        default=None, description="When the user last logged in"
    )

    class Config:
        """Pydantic model configuration"""

        validate_assignment = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}
