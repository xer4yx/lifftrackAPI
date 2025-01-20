from typing import Any, Dict, List, Optional
import json
from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseSettings):
    """Database configuration with defaults"""
    model_config = ConfigDict(
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_prefix="",
        env_nested_delimiter="__"
    )
    DB_TYPE: str = Field(
        default="admin",
        description="Database type (admin or rest)"
    )
    FIREBASE_DB_URL: Optional[str] = Field(
        default=None,
        description="Firebase Database URL"
    )
    FIREBASE_OPTIONS: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Firebase Database Options"
    )
    FIREBASE_CREDENTIALS_PATH: Optional[str] = Field(
        default=None,
        description="Path to Firebase credentials file"
    )
    FIREBASE_AUTH_UID: Optional[str] = Field(
        default=None,
        description="Firebase Authentication UID"
    )
    FIREBASE_DSN: Optional[str] = Field(
        default=None,
        description="Firebase Data Source Name"
    )
    FIREBASE_AUTH_TOKEN: Optional[str] = Field(
        default=None,
        description="Firebase Authentication Token"
    )
    
    @field_validator('DB_TYPE')
    def validate_db_type(cls, v):
        if v not in ['admin', 'rest']:
            raise ValueError('DB_TYPE must be either "admin" or "rest"')
        return v

class SecurityConfig(BaseSettings):
    """Security configuration with defaults"""
    model_config = ConfigDict(
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_prefix="",
        env_nested_delimiter="__"
    )
    JWT_SECRET_KEY: str = Field(
        default="secret-key-default",
        description="Secret key for JWT encoding/decoding"
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="Algorithm used for JWT"
    )
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="JWT token expiration time in minutes"
    )
    ENCRYPTION_KEY: Optional[str] = Field(
        default=None,
        description="Key used for encryption"
    )
    PASSWORD_SALT: Optional[str] = Field(
        default=None,
        description="Salt used for password hashing"
    )
    
    @field_validator('JWT_ACCESS_TOKEN_EXPIRE_MINUTES')
    def validate_token_expire(cls, v):
        if v < 1:
            raise ValueError('Token expiration must be at least 1 minute')
        return v

class MonitoringConfig(BaseSettings):
    """Monitoring configuration with defaults"""
    model_config = ConfigDict(
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_prefix="",
        env_nested_delimiter="__"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    LOG_DIR: str = Field(
        default="logs",
        description="Directory for log files"
    )
    METRICS_DIR: str = Field(
        default="metrics",
        description="Directory for metrics files"
    )
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    METRICS_INTERVAL: int = Field(
        default=60,
        description="Metrics collection interval in seconds"
    )

class VisionConfig(BaseSettings):
    """Vision processing configuration with defaults"""
    model_config = ConfigDict(
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_prefix="",
        env_nested_delimiter="__"
    )
    MODEL_URL: str = Field(
        default="https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-lightning/4",
        description="URL to vision model"
    )
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.5,
        description="Minimum confidence threshold for detections"
    )
    MODEL_PATH: Optional[str] = Field(
        default=None,
        description="Path to vision model"
    )
    MAX_FRAME_SIZE: int = Field(
        default=1024,
        description="Maximum frame size for processing"
    )
    ENABLE_GPU: bool = Field(
        default=False,
        description="Enable GPU acceleration"
    )
    ROBOFLOW_API_URL: str = Field(
        default="http://localhost:9001",
        description="Roboflow API URL"
    )
    ROBOFLOW_API_KEY: Optional[str] = Field(
        default=None,
        description="Roboflow API key"
    )
    ROBOFLOW_PROJECT_ID: Optional[str] = Field(
        default=None,
        description="Roboflow project ID"
    )
    ROBOFLOW_MODEL_VER: Optional[str] = Field(
        default=None,
        description="Roboflow model version"
    )
    
    @field_validator('CONFIDENCE_THRESHOLD')
    def validate_confidence_threshold(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Confidence threshold must be between 0 and 1')
        return v

class AppConfig(BaseSettings):
    """Application configuration with defaults and environment variable support"""
    # Basic Settings
    model_config = ConfigDict(
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_nested_delimiter="__"
    )
    APP_NAME: str = Field(
        default="lifttrack",
        description="Application name"
    )
    ENV: str = Field(
        default="development",
        description="Environment (development, testing, production)"
    )
    DEBUG: bool = Field(
        default=False,
        description="Debug mode"
    )
    API_PREFIX: str = Field(
        default="/api",
        description="API prefix"
    )
    
    # Component Configurations
    DATABASE: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration"
    )
    SECURITY: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration"
    )
    MONITORING: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring configuration"
    )
    VISION: VisionConfig = Field(
        default_factory=VisionConfig,
        description="Vision configuration"
    )
    
    # Server Settings
    HOST: str = Field(
        default="127.0.0.1",
        description="Server host"
    )
    PORT: int = Field(
        default=8000,
        description="Server port"
    )
    RELOAD: bool = Field(
        default=False,
        description="Enable or disable hot reload"
    )
    RELOAD_EXCLUDES: List[str] = Field(
        default=None,
        description="Excluded files and/or directories from reload."
    )
    WORKERS: int = Field(
        default=1,
        description="Number of worker processes"
    )
    
    # CORS Settings
    ALLOWED_ORIGINS: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    ALLOWED_METHODS: list[str] = Field(
        default=["*"],
        description="Allowed CORS methods"
    )
    ALLOWED_HEADERS: list[str] = Field(
        default=["*"],
        description="Allowed CORS headers"
    )
    
    @field_validator('ENV')
    def validate_env(cls, v):
        if v not in ['development', 'testing', 'production']:
            raise ValueError('ENV must be one of: development, testing, production')
        return v
