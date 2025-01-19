import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
from typing import Optional

class CustomRotatingFileHandler(RotatingFileHandler):
    def __init__(
        self,
        filename: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: Optional[str] = None
    ):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        super().__init__(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding
        )

class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(
        self,
        filename: str,
        when: str = 'midnight',
        interval: int = 1,
        backup_count: int = 30,
        encoding: Optional[str] = None
    ):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        super().__init__(
            filename,
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding=encoding
        ) 