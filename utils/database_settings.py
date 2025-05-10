from pydantic import Field
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings


class NoSQLSettings(BaseSettings):
    """Configuration settings for NoSQL databases (MongoDB, Firebase RTDB).
    Loads configuration from environment variables and .env file.
    """
    
    # MongoDB settings
    mongodb_uri: Optional[str] = Field(default=None, description="MongoDB connection URI")
    mongodb_db_name: Optional[str] = Field(default=None, description="MongoDB database name")
    mongodb_min_pool_size: Optional[int] = Field(default=None, description="MongoDB minimum connection pool size")
    mongodb_max_pool_size: Optional[int] = Field(default=None, description="MongoDB maximum connection pool size")
    mongodb_timeout_ms: Optional[int] = Field(default=None, description="MongoDB connection timeout in milliseconds")
    
    # Firebase RTDB settings
    firebase_api_key: Optional[str] = Field(default=None, description="Firebase API key")
    firebase_auth_domain: Optional[str] = Field(default=None, description="Firebase auth domain")
    firebase_database_url: Optional[str] = Field(default=None, description="Firebase Realtime Database URL")
    firebase_project_id: Optional[str] = Field(default=None, description="Firebase project ID")
    firebase_storage_bucket: Optional[str] = Field(default=None, description="Firebase storage bucket")
    firebase_messaging_sender_id: Optional[str] = Field(default=None, description="Firebase messaging sender ID")
    firebase_app_id: Optional[str] = Field(default=None, description="Firebase application ID")
    firebase_measurement_id: Optional[str] = Field(default=None, description="Firebase measurement ID")
    
    # Common NoSQL settings
    nosql_timeout: int = Field(default=5, description="General timeout for NoSQL operations in seconds")
    nosql_retry_attempts: int = Field(default=3, description="Number of retry attempts for failed operations")
    nosql_cache_enabled: bool = Field(default=True, description="Enable caching for NoSQL queries")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "NOSQL_"  # Environment variables prefix
        case_sensitive = False
        validate_assignment = True
        extra = "ignore"  # Ignore extra attributes
        env_file = ".env"  # Specify the .env file to load
        env_file_encoding = "utf-8"

