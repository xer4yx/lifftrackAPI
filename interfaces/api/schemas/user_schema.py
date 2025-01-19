from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, EmailStr

from core.entities import User as UserEntity

class UserSchema(BaseModel):
    """
    User base schema for the application.
    """
    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True
    )
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[EmailStr] = None
    profile_picture: Optional[str] = None
    
    def to_entity(self) -> UserEntity:
        """Convert Pydantic User model to User entity"""
        entity_data = self.model_dump(
            exclude_unset=True,
            exclude_none=True
        )
        return UserEntity(**entity_data)

class UserCreateSchema(UserSchema):
    """
    User create schema for the application.
    Requires all necessary fields for user creation.
    """
    username: str
    email: EmailStr
    password: str = Field(min_length=8, max_length=12)
    first_name: str
    last_name: str
    phone_number: str

class UserUpdateSchema(UserSchema):
    """
    User update schema for the application.
    All fields are optional for updates.
    """
    password: Optional[str] = Field(None, min_length=8, max_length=12)

class UserResponseSchema(UserSchema):
    """
    User response schema for the application.
    """
    pass
