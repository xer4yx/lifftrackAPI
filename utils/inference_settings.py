from pydantic import Field
from pydantic_settings import BaseSettings


class InferenceSdkSettings(BaseSettings):
    """Configuration settings for Roboflow Inference SDK."""
    api_url: str = Field(..., description="Roboflow API URL")
    api_key: str = Field(..., description="Roboflow API key")
    project_id: str = Field(..., description="Roboflow project ID")
    model_version: int = Field(..., description="Roboflow model version")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "ROBOFLOW_"  # Environment variables prefix
        case_sensitive = False
        validate_assignment = True
        extra = "ignore"  # Ignore extra attributes
        env_file = ".env"  # Specify the .env file to load
        env_file_encoding = "utf-8"

