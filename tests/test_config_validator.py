import pytest
from utilities.validators.config_validator import (
    AppConfig,
    DatabaseConfig,
    SecurityConfig,
    MonitoringConfig,
    VisionConfig
)
import os
from pathlib import Path

@pytest.fixture(autouse=True)
def setup_env():
    """Setup the environment for tests"""
    # Get the path to your .env file
    env_path = Path(__file__).parent.parent / ".env"
    os.environ["ENV_FILE"] = str(env_path)
    yield
    # Clean up after tests
    if "ENV_FILE" in os.environ:
        del os.environ["ENV_FILE"]

def test_app_config_defaults():
    """Test default values of AppConfig"""
    config = AppConfig()
    assert config.APP_NAME == "lifttrack-local-test"
    assert config.ENV == "testing"
    assert config.DEBUG == False
    assert config.API_PREFIX == "/api"
    assert config.HOST == "127.0.0.1"
    assert config.PORT == 8080

def test_app_config_env_validation():
    """Test ENV validator"""
    config = AppConfig(ENV="development")
    assert config.ENV == "development"
    
    with pytest.raises(ValueError, match="ENV must be one of: development, testing, production"):
        AppConfig(ENV="invalid")

def test_database_config():
    """Test DatabaseConfig validation and defaults"""
    config = DatabaseConfig()
    assert config.DB_TYPE == "admin"
    
    # Test valid DB_TYPE values
    config = DatabaseConfig(DB_TYPE="rest")
    assert config.DB_TYPE == "rest"
    
    # Test invalid DB_TYPE
    with pytest.raises(ValueError, match='DB_TYPE must be either "admin" or "rest"'):
        DatabaseConfig(DB_TYPE="invalid")

def test_security_config():
    """Test SecurityConfig validation and defaults"""
    config = AppConfig()
    assert config.SECURITY.JWT_ALGORITHM == "HS256"
    assert config.SECURITY.JWT_ACCESS_TOKEN_EXPIRE_MINUTES == 10
    
    # Test token expiration validation
    with pytest.raises(ValueError, match="Token expiration must be at least 1 minute"):
        SecurityConfig(JWT_ACCESS_TOKEN_EXPIRE_MINUTES=0)

def test_monitoring_config():
    """Test MonitoringConfig defaults"""
    config = AppConfig()
    assert config.MONITORING.LOG_LEVEL == "WARNING"
    assert config.MONITORING.LOG_DIR == "logs"
    assert config.MONITORING.ENABLE_METRICS is True
    assert config.MONITORING.METRICS_INTERVAL == 80

def test_vision_config():
    """Test VisionConfig defaults"""
    config = AppConfig()
    assert config.VISION.MODEL_PATH == "Z:\Python Projects\lifttrackAPI\model\lifttrack_cnn_bypass.keras"
    assert config.VISION.CONFIDENCE_THRESHOLD == 0.1
    assert config.VISION.MAX_FRAME_SIZE == 1024
    assert config.VISION.ENABLE_GPU is False

def test_custom_values():
    """Test setting custom values"""
    config = AppConfig(
        APP_NAME="custom_app",
        ENV="production",
        DATABASE={"DB_TYPE": "rest"},
        SECURITY={"JWT_ACCESS_TOKEN_EXPIRE_MINUTES": 60},
        VISION={"ENABLE_GPU": True}
    )
    
    assert config.APP_NAME == "custom_app"
    assert config.ENV == "production"
    assert config.DATABASE.DB_TYPE == "rest"
    assert config.SECURITY.JWT_ACCESS_TOKEN_EXPIRE_MINUTES == 60
    assert config.VISION.ENABLE_GPU is True

def test_nested_config_validation():
    """Test validation in nested configs"""
    with pytest.raises(ValueError):
        AppConfig(DATABASE={"DB_TYPE": "invalid"})
    
    with pytest.raises(ValueError):
        AppConfig(SECURITY={"JWT_ACCESS_TOKEN_EXPIRE_MINUTES": 0})