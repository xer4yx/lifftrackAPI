from enum import Enum
from typing import Optional
from pydantic import BaseModel

class AuthScheme(str, Enum):
    BEARER = "Bearer"
    API_KEY = "ApiKey"

class SecurityHeaders(BaseModel):
    authorization: str
    x_api_key: Optional[str] = None

class PaginationHeaders(BaseModel):
    x_total_count: int
    x_page: int
    x_per_page: int
    x_total_pages: int