from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List


class CorsSettings(BaseSettings):
    allowed_origins: List[str] = Field(default=["*"], description="List of allowed origins")
    allowed_methods: List[str] = Field(default=["*"], description="List of allowed methods")
    allowed_headers: List[str] = Field(default=["*"], description="List of allowed headers")
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    expose_headers: List[str] = Field(default=["*"], description="List of exposed headers")

    class Config:
        env_prefix = "CORS_"
        case_sensitive = False
        validate_assignment = True
        extra = "ignore"
        env_file = ".env"  # Specify the .env file to load
        env_file_encoding = "utf-8"
