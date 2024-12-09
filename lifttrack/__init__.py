from lifttrack.utils.warning_suppressor import suppress_warnings
from lifttrack.utils.logging_config import setup_logger

# Suppress TensorFlow warnings
suppress_warnings()

import os
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional, Union
import warnings
from dotenv import load_dotenv

import threading
import asyncio
import logging
import configparser

import cv2
from cv2 import Mat
import base64

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini')

config = configparser.ConfigParser()
config.read(config_path)

load_dotenv('./.env')

network_logger = setup_logger("network", "network.log")
