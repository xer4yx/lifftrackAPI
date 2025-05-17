from pydantic import Field
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Configuration settings model using Pydantic for validation.
    Loads configuration from environment variables and .env file.
    """
    
    # Basic settings
    name: str = Field(default="MyApp", description="Name of the application")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    version: str = Field(default="1.0.0", description="Application version")
    
    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    
    # Logging settings
    file_log_level: str = Field(default="INFO", description="Logging level")
    screen_log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "APP_"  # Environment variables prefix
        case_sensitive = False
        validate_assignment = True
        extra = "ignore"  # Ignore extra attributes
        env_file = ".env"  # Specify the .env file to load
        env_file_encoding = "utf-8"
