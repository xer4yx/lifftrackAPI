import os
import logging

def setup_logger(logger_name: str, log_file: str) -> logging.Logger:
    """
    Set up a logger with dynamic log directory creation.
    
    Args:
        logger_name: Name of the logger
        log_file: Name of the log file
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    handler = logging.FileHandler(os.path.join(logs_dir, log_file))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler if it doesn't exist
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger 