from typing import Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class TokenEntity(BaseModel):
    """
    Token entity model representing an authentication token.
    """
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_at: datetime = Field(..., description="Token expiration time")


class TokenDataEntity(BaseModel):
    """
    Token data entity representing the payload of a JWT token.
    """
    username: str = Field(..., description="Username associated with the token")
    exp: datetime = Field(..., description="Token expiration timestamp")
    

class TokenBlacklistEntity(BaseModel):
    """
    Token blacklist entity representing invalidated tokens.
    """
    token: str = Field(..., description="The blacklisted token")
    expiry: datetime = Field(..., description="When the token expires")
    blacklisted_on: datetime = Field(default=datetime.now(timezone.utc), description="When the token was blacklisted")


class CredentialsEntity(BaseModel):
    """
    Credentials entity for authentication requests.
    """
    username: str = Field(..., description="Username for authentication")
    password: str = Field(..., description="Password for authentication")


class ValidationResultEntity(BaseModel):
    """
    Validation result entity for input validation operations.
    """
    is_valid: bool = Field(..., description="Whether the validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors if any") 