from lifttrack.dbhandler import rtdb


def put_data(user_data: dict[str, any]):
    """
    Adds a new user to the database via HTTP PUT request.
    """
    snapshot = rtdb.put(
        url='/users',
        name=user_data['username'],
        data=user_data,
    )

    if snapshot is None:
        raise ValueError('User not created')


def get_data(username, data=None):
    """

    :param username:
    :param data:
    :return:
    """
    snapshot = rtdb.get(
        url=f'/users/{username}',
        name=data)

    if snapshot is None:
        raise ValueError('User doesn\'t exist!')

    return snapshot


def update_data(username, user_data):
    """

    :param username:
    :param user_data:
    :param column:
    :return:
    """
    return rtdb.put(
        url=f'/users',
        name=username,
        data=user_data)


def delete_data(username):
    """

    :param username:
    :return:
    """
    return rtdb.delete(
        url=f'/users',
        name=username)
