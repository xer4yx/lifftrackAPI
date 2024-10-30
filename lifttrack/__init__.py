import os
import threading
import asyncio
from datetime import datetime, timedelta
import logging
import configparser
from pydantic import BaseModel
from typing import Optional, Union
import cv2
import ast

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini')

config = configparser.ConfigParser()
config.read(config_path)
