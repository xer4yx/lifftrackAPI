import configparser
from firebase import firebase


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

    def put_data(self, user_data: dict[str, any]):
        """
        Adds a new user to the database via HTTP PUT request.
        """
        snapshot = self.__db.put(
            url='/users',
            name=user_data['username'],
            data=user_data,
        )

        if snapshot is None:
            raise ValueError('User not created')

    def get_data(self, username, data=None):
        """

        :param username:
        :param data:
        :return:
        """
        snapshot = self.__db.get(
            url=f'/users/{username}',
            name=data)

        if snapshot is None:
            raise ValueError('User doesn\'t exist!')

        return snapshot

    def update_data(self, username, user_data):
        """

        :param username:
        :param user_data:
        :param column:
        :return:
        """
        return self.__db.put(
            url=f'/users',
            name=username,
            data=user_data)

    def delete_data(self, username):
        """

        :param username:
        :return:
        """
        return self.__db.delete(
            url=f'/users',
            name=username)
