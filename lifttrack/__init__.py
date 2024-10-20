from datetime import datetime, timedelta
import configparser
from pydantic import BaseModel
from typing import Optional, Union
from firebase import firebase

config = configparser.ConfigParser()
config.read('../config.ini')


class RTDBHelper:
    def __init__(self):
        self.__config = configparser.ConfigParser()

        if not self.__config.read('config.ini'):
            raise FileNotFoundError('Config file not found')

        if not self.__config.has_section('Firebase'):
            raise KeyError('Config file is missing Firebase section')

        self.__dsn = self.__config.get(section='Firebase', option='dsn')
        self.__auth = self.__config.get(section='Firebase', option='authentication')
        self.__db = firebase.FirebaseApplication(
            dsn=self.__dsn,
            authentication=(lambda: None, lambda: self.__auth)[self.__auth != 'None']()
        )


rtdb = RTDBHelper()
