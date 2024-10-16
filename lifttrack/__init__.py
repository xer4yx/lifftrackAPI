from datetime import datetime, timedelta
import configparser
from pydantic import BaseModel
from typing import Optional, Union

config = configparser.ConfigParser()
config.read('../config.ini')
