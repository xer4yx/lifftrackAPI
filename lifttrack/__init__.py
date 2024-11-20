from lifttrack.utils.warning_suppressor import suppress_warnings

# Suppress TensorFlow warnings
suppress_warnings()

import os
import warnings
import threading
import asyncio
from datetime import datetime, timedelta
import logging
import configparser
from pydantic import BaseModel
from typing import Optional, Union
import cv2
from cv2 import Mat
import base64
from lifttrack.utils.logging_config import setup_logger

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini')

config = configparser.ConfigParser()
config.read(config_path)

network_logger = setup_logger("network", "network.log")
