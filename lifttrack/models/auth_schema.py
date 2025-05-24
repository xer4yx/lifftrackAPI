from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class LoginForm(BaseModel):
    username: str = Field(default=..., title="Username")
    password: str = Field(default=..., title="Password")
    
    
class Token(BaseModel):
    access_token: str = Field(default=..., title="Access Token")
    token_type: str = Field(default=..., title="Token Type")


class TokenData(BaseModel):
    username: Optional[str] = Field(default=None, title="Username")


class TokenBlacklist(BaseModel):
    token: str = Field(default=..., title="Blacklisted Token")
    expiry: datetime = Field(default=..., title="Token Expiry Time")
    blacklisted_on: datetime = Field(default=datetime.now(tz=timezone.utc), title="Blacklisted Timestamp")