from typing import Optional, Dict, Any
import logging
import sys
from .handlers import CustomRotatingFileHandler, CustomTimedRotatingFileHandler
from .formatters import JSONFormatter

def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    rotation_type: str = "size",
    **kwargs: Any
) -> logging.Logger:
    """Set up a logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    json_formatter = JSONFormatter()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    if rotation_type == "time":
        file_handler = CustomTimedRotatingFileHandler(
            log_file,
            when=kwargs.get('when', 'midnight'),
            interval=kwargs.get('interval', 1),
            backup_count=kwargs.get('backup_count', 30)
        )
    else:
        file_handler = CustomRotatingFileHandler(
            log_file,
            max_bytes=kwargs.get('max_bytes', 10 * 1024 * 1024),
            backup_count=kwargs.get('backup_count', 5)
        )
    
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger
