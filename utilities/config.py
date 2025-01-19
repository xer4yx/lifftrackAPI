import os
import dotenv
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

from .validators.config_validator import AppConfig
from .monitoring.factory import MonitoringFactory

logger = MonitoringFactory.get_logger("config")

@lru_cache()
def get_config() -> AppConfig:
    """Get application configuration with environment variable overrides"""
    try:
        # Load configuration
        config = AppConfig()
        
        # Log configuration source
        env_file = Path(".env")
        if dotenv.find_dotenv(filename=env_file) != "":
            dotenv.load_dotenv(dotenv_path=env_file)
            logger.info("Configuration loaded from .env file")
        else:
            logger.info("Configuration loaded from environment variables")
            
        # Log important settings
        logger.info(f"Environment: {config.ENV}")
        logger.info(f"Debug mode: {config.DEBUG}")
        logger.info(f"Database type: {config.DATABASE.DB_TYPE}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.warning("Using default configuration")
        return AppConfig()

def get_database_settings() -> Dict[str, Any]:
    """Get database-specific settings"""
    config = get_config()
    return config.DATABASE.model_dump()

def get_security_settings() -> Dict[str, Any]:
    """Get security-specific settings"""
    config = get_config()
    return config.SECURITY.model_dump()

def get_monitoring_settings() -> Dict[str, Any]:
    """Get monitoring-specific settings"""
    config = get_config()
    return config.MONITORING.model_dump()

def get_vision_settings() -> Dict[str, Any]:
    """Get vision-specific settings"""
    config = get_config()
    return config.VISION.model_dump()