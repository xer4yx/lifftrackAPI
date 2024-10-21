import threading
import asyncio
from datetime import datetime, timedelta
import configparser
from pydantic import BaseModel
from typing import Optional, Union

import cv2

config = configparser.ConfigParser()
config.read('../config.ini')
