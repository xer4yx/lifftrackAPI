import os
import logging
import psutil
import time
import threading


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


def log_cpu_and_mem_usage(logger=None, interval=5):
    """
    Logs system resource usage at specified intervals.
    
    Args:
        logger (logging.Logger, optional): Custom logger instance. If None, uses default resource_logger
        interval (int): Time in seconds between logging resource usage
    """
    if logger is None:
        logger = setup_logger("cpu-mem-log", "server_resource.log")
        
    while True:
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        logger.info(f"CPU Usage: {cpu_usage}% | Memory Usage: {memory_info.percent}%")
        time.sleep(interval)


def start_resource_monitoring(logger=None, target=None, interval=30):
    """
    Starts resource monitoring in a background thread.
    
    Args:
        logger (logging.Logger, optional): Custom logger instance to use
        interval (int): Time in seconds between logging resource usage
    
    Returns:
        threading.Thread: The monitoring thread instance
    """
    resource_thread = threading.Thread(
        target=target, 
        args=(logger, interval), 
        daemon=True
    )
    
    resource_thread.start()
    return resource_thread 


def log_network_io(logger, endpoint: str, response_status: int):
    """
    Logs network I/O counters when an endpoint is called.
    
    Args:
        logger: Logger instance
        endpoint: The endpoint that was called
        response_status: HTTP response status code
    """
    try:
        net_io = psutil.net_io_counters()
        logger.info(
            f"Network I/O - Endpoint: {endpoint} | Status: {response_status} | "
            f"Bytes Sent: {net_io.bytes_sent} | "
            f"Bytes Recv: {net_io.bytes_recv} | "
            f"Packets Sent: {net_io.packets_sent} | "
            f"Packets Recv: {net_io.packets_recv} | "
            f"Err In: {net_io.errin} | "
            f"Err Out: {net_io.errout} | "
            f"Drop In: {net_io.dropin} | "
            f"Drop Out: {net_io.dropout}"
        )
    except Exception as e:
        logger.error(f"Error logging network I/O: {str(e)}")