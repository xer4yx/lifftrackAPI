from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, AliasChoices


class User(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Optional[str] = Field(default=datetime.strftime(datetime.now(), '%Y%H%d%m'), title="User ID")
    fname: str = Field(default=..., alias="first_name", serialization_alias="fname", validation_alias=AliasChoices('fname', 'first_name'), title="First Name")
    lname: str = Field(default=..., alias="last_name", serialization_alias="lname", validation_alias=AliasChoices('lname', 'last_name'), title="Last Name")
    username: str = Field(default=..., title="Username")
    phoneNum: str = Field(default=..., alias="phone_number", serialization_alias="phoneNum", validation_alias=AliasChoices('phoneNum', 'phone_number'), title="Phone Number")
    email: str = Field(default=..., title="Email")
    password: str = Field(default=..., title="Password")
    pfp: Optional[str] = Field(default=None, alias="profile_picture", serialization_alias="pfp", validation_alias=AliasChoices('pfp', 'profile_picture'), title="Profile Picture")
    isAuthenticated: Optional[bool] = Field(default=False, alias="is_authenticated", serialization_alias="isAuthenticated", validation_alias=AliasChoices('isAuthenticated', 'is_authenticated'), title="Authenticated")
    isDeleted: Optional[bool] = Field(default=False, alias="is_deleted", serialization_alias="isDeleted", validation_alias=AliasChoices('isDeleted', 'is_deleted'), title="Deleted")
    
    
class UserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fname: Optional[str] = Field(default=None, title="First Name")
    lname: Optional[str] = Field(default=None, title="Last Name")
    phoneNum: Optional[str] = Field(default=None, title="Phone Number")
    email: Optional[str] = Field(default=None, title="Email")
    password: Optional[str] = Field(default=None, title="Password")
    pfp: Optional[str] = Field(default=None, title="Profile Picture")


class UserCreateResponse(BaseModel):
    """Response model for user creation without password field"""
    model_config = ConfigDict(extra="forbid")
    id: Optional[str] = Field(title="User ID")
    fname: str = Field(alias="first_name", serialization_alias="fname", validation_alias=AliasChoices('fname', 'first_name'), title="First Name")
    lname: str = Field(alias="last_name", serialization_alias="lname", validation_alias=AliasChoices('lname', 'last_name'), title="Last Name")
    username: str = Field(title="Username")
    phoneNum: str = Field(alias="phone_number", serialization_alias="phoneNum", validation_alias=AliasChoices('phoneNum', 'phone_number'), title="Phone Number")
    email: str = Field(title="Email")
    pfp: Optional[str] = Field(default=None, alias="profile_picture", serialization_alias="pfp", validation_alias=AliasChoices('pfp', 'profile_picture'), title="Profile Picture")
    isAuthenticated: Optional[bool] = Field(default=False, alias="is_authenticated", serialization_alias="isAuthenticated", validation_alias=AliasChoices('isAuthenticated', 'is_authenticated'), title="Authenticated")
    isDeleted: Optional[bool] = Field(default=False, alias="is_deleted", serialization_alias="isDeleted", validation_alias=AliasChoices('isDeleted', 'is_deleted'), title="Deleted")


class UserResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Optional[str] = Field(title="User ID")
    fname: str = Field(alias="first_name", serialization_alias="fname", validation_alias=AliasChoices('fname', 'first_name'), title="First Name")
    lname: str = Field(alias="last_name", serialization_alias="lname", validation_alias=AliasChoices('lname', 'last_name'), title="Last Name")
    username: str = Field(title="Username")
    phoneNum: str = Field(alias="phone_number", serialization_alias="phoneNum", validation_alias=AliasChoices('phoneNum', 'phone_number'), title="Phone Number")
    email: str = Field(title="Email")
    pfp: Optional[str] = Field(default=None, alias="profile_picture", serialization_alias="pfp", validation_alias=AliasChoices('pfp', 'profile_picture'), title="Profile Picture")
    isAuthenticated: Optional[bool] = Field(default=False, alias="is_authenticated", serialization_alias="isAuthenticated", validation_alias=AliasChoices('isAuthenticated', 'is_authenticated'), title="Authenticated")
    isDeleted: Optional[bool] = Field(default=False, alias="is_deleted", serialization_alias="isDeleted", validation_alias=AliasChoices('isDeleted', 'is_deleted'), title="Deleted")
    createdAt: Optional[datetime] = Field(default=None, alias="created_at", validation_alias=AliasChoices('createdAt', 'created_at'), title="Date the user was created")
    updatedAt: Optional[datetime] = Field(default=None, alias="updated_at", validation_alias=AliasChoices('updatedAt', 'updated_at'), title="Date the user was last updated")
    lastLogin: Optional[datetime] = Field(default=None, alias="last_login", validation_alias=AliasChoices('lastLogin', 'last_login'), title="Date the user was last logged in")

