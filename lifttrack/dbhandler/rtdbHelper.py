import logging
from firebase import firebase
from lifttrack import config
from lifttrack.utils.logging_config import setup_logger


# Logging Configuration
logger = setup_logger("rtdbHelper", "lifttrack_db.log")


class RTDBHelper:
    def __init__(self, dsn=None, authentication=None):
        self.__dsn = config.get(section='Firebase', option='dsn') or dsn
        self.__auth = config.get(section='Firebase', option='authentication') or authentication
        self.__db = firebase.FirebaseApplication(
            dsn=self.__dsn,
            authentication=(lambda: None, lambda: self.__auth)[self.__auth != 'None']()
        )
        logger.info("RTDBHelper initialized.")

    def put_data(self, user_data: dict[str, any]):
        """
        Adds a new user to the database via HTTP PUT request.
        """
        try:
            snapshot = self.__db.put(
                url='/users',
                name=user_data['username'],
                data=user_data,
            )
            if snapshot is None:
                logger.error(f"Failed to create user: {user_data['username']}")
                raise ValueError('User not created')
            logger.info(f"User created successfully: {user_data['username']}")
        except Exception as e:
            logger.exception(f"Exception in put_data for user {user_data['username']}: {e}")
            raise

    def get_data(self, username, data=None):
        """
        Retrieves user data from the database.

        :param username: Username of the user.
        :param data: Specific field to retrieve.
        :return: User data or specific field.
        """
        try:
            snapshot = self.__db.get(
                url=f'/users/{username}',
                name=data
            )
            if snapshot is None:
                logger.warning(f"Data not found for user: {username}")
            else:
                logger.info(f"Data retrieved for user: {username}")
            return snapshot
        except Exception as e:
            logger.exception(f"Exception in get_data for user {username}: {e}")
            raise

    def update_data(self, username, user_data):
        """
        Updates user data in the database.

        :param username: Username of the user.
        :param user_data: Data to update.
        :return: Snapshot of updated data.
        """
        try:
            snapshot = self.__db.put(
                url=f'/users',
                name=username,
                data=user_data
            )
            if snapshot is None:
                logger.error(f"Failed to update user: {username}")
            else:
                logger.info(f"User updated successfully: {username}")
            return snapshot
        except Exception as e:
            logger.exception(f"Exception in update_data for user {username}: {e}")
            raise

    def delete_data(self, username):
        """
        Deletes a user from the database.

        :param username: Username of the user to delete.
        :return: True if deleted, else False.
        """
        try:
            deleted = self.__db.delete(
                url=f'/users',
                name=username
            )
            if deleted is False:
                logger.error(f"Failed to delete user: {username}")
            else:
                logger.info(f"User deleted successfully: {username}")
            return deleted
        except Exception as e:
            logger.exception(f"Exception in delete_data for user {username}: {e}")
            raise


rtdb = RTDBHelper()
