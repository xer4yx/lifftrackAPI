from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class User(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Optional[str] = Field(default=datetime.strftime(datetime.now(), '%Y%H%d%m'), title="User ID")
    fname: str = Field(default=..., title="First Name")
    lname: str = Field(default=..., title="Last Name")
    username: str = Field(default=..., title="Username")
    phoneNum: str = Field(default=..., title="Phone Number")
    email: str = Field(default=..., title="Email")
    password: str = Field(default=..., title="Password")
    pfp: Optional[str] = Field(default=None, title="Profile Picture")
    isAuthenticated: Optional[bool] = Field(default=False, title="Authenticated")
    isDeleted: Optional[bool] = Field(default=False, title="Deleted")
    
    
class UserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fname: Optional[str] = Field(default=None, title="First Name")
    lname: Optional[str] = Field(default=None, title="Last Name")
    phoneNum: Optional[str] = Field(default=None, title="Phone Number")
    email: Optional[str] = Field(default=None, title="Email")
    password: Optional[str] = Field(default=None, title="Password")
    pfp: Optional[str] = Field(default=None, title="Profile Picture")
    