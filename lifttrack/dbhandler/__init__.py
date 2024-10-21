from firebase import firebase

from lifttrack import config


class RTDBHelper:
    def __init__(self, dsn=None, authentication=None):
        self.__dsn = config.get(section='Firebase', option='dsn') or dsn
        self.__auth = config.get(section='Firebase', option='authentication') or authentication
        self.__db = firebase.FirebaseApplication(
            dsn=self.__dsn,
            authentication=(lambda: None, lambda: self.__auth)[self.__auth != 'None']()
        )


rtdb = RTDBHelper()
