from pydantic import Field, BaseModel
import json
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class NoSQLSettings(BaseSettings):
    """Configuration settings for NoSQL databases (MongoDB, Firebase RTDB)"""

    nosql_timeout: int = Field(
        default=5, description="General timeout for NoSQL operations in seconds"
    )
    nosql_retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed operations"
    )
    nosql_cache_enabled: bool = Field(
        default=True, description="Enable caching for NoSQL queries"
    )

    class Config:
        """Pydantic configuration."""

        env_prefix = "NOSQL_"  # Environment variables prefix
        case_sensitive = False
        validate_assignment = True
        extra = "ignore"  # Ignore extra attributes
        env_file = ".env"  # Specify the .env file to load
        env_file_encoding = "utf-8"


class FirebaseDatabaseAuthVariableOverride(BaseModel):
    """Configuration settings for Firebase RTDB options."""

    uid: Optional[str] = Field(default=None, description="Firebase auth UID")
    admin: Optional[bool] = Field(default=False, description="Firebase admin")


class FirebaseOptions(BaseModel):
    """Configuration settings for Firebase RTDB options."""

    databaseURL: str = Field(..., description="Firebase Realtime Database URL")
    databaseAuthVariableOverride: FirebaseDatabaseAuthVariableOverride = Field(
        ..., description="Firebase auth variable override"
    )


class FirebaseSettings(BaseSettings):
    """Configuration settings for Firebase RTDB."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="_",
        env_prefix="FIREBASE_",
        case_sensitive=False,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )
    database_url: Optional[str] = Field(
        default=None, description="Firebase Realtime Database URL"
    )
    options: Optional[FirebaseOptions] = Field(
        default=None, description="Firebase options"
    )
    admin_sdk: Optional[str] = Field(
        default=None, description="Firebase admin SDK path"
    )
    auth_uid: Optional[str] = Field(default=None, description="Firebase auth UID")
    auth_token: Optional[str] = Field(default=None, description="Firebase auth token")


class MongoDbSettings(BaseSettings):
    """Configuration settings for MongoDB."""

    uri: Optional[str] = Field(default=None, description="MongoDB connection URI")
    db_name: Optional[str] = Field(default=None, description="MongoDB database name")
    min_pool_size: Optional[int] = Field(
        default=None, description="MongoDB minimum connection pool size"
    )
    max_pool_size: Optional[int] = Field(
        default=None, description="MongoDB maximum connection pool size"
    )
    timeout_ms: Optional[int] = Field(
        default=None, description="MongoDB connection timeout in milliseconds"
    )

    class Config:
        """Pydantic configuration."""

        env_prefix = "MONGO_"  # Environment variables prefix
        case_sensitive = False
        validate_assignment = True
        extra = "ignore"  # Ignore extra attributes
        env_file = ".env"  # Specify the .env file to load
        env_file_encoding = "utf-8"
