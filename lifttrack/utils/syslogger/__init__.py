import psutil
import threading
import time
from fastapi import Request
from lifttrack.utils.logging_config import setup_logger

def log_network_io(logger, endpoint: str, response_status: int):
    """
    Logs network I/O counters when an endpoint is called.
    """
    try:
        net_io = psutil.net_io_counters()
        logger.info(
            f"Network I/O - Endpoint: {endpoint} | Status: {response_status} | "
            f"Bytes Sent: {net_io.bytes_sent} | Bytes Recv: {net_io.bytes_recv}"
        )
    except Exception as e:
        logger.error(f"Error logging network I/O: {str(e)}")

def log_cpu_and_mem_usage(logger=None, interval=5):
    """Logs system resource usage at specified intervals."""
    if logger is None:
        logger = setup_logger("resources", "lifttrack_resources.log")
        
    while True:
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        logger.info(f"CPU Usage: {cpu_usage}% | Memory Usage: {memory_info.percent}%")
        time.sleep(interval)

def start_resource_monitoring(logger=None, target=None, interval=30):
    """Starts resource monitoring in a background thread."""
    resource_thread = threading.Thread(
        target=target, 
        args=(logger, interval), 
        daemon=True
    )
    resource_thread.start()
    return resource_thread