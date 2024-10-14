import time
import requests
domain = 'http://localhost:8000'


def test_create_user():
    global domain
    time.sleep(3)
    print('Creating user...')
    payload_sample = {
        "fname": "John",
        "lname": "Doe",
        "username": "johndoe",
        "phoneNum": "1234567890",
        "email": "johndoe@mail.com",
        "password": "thisishashed"
    }

    x = requests.put(f'{domain}/user/create', json=payload_sample)
    assert x.status_code == 200


def test_get_user():
    global domain
    time.sleep(3)
    print('Getting user...')
    key = 'johndoe'
    x = requests.get(f'{domain}/user/{key}')
    assert x.status_code == 200
    print(x.json())


def test_update_user():
    global domain
    time.sleep(3)
    print('Updating user...')
    key = 'johndoe'
    payload_sample = {
        "fname": "John",
        "lname": "Doe",
        "username": "johndoe",
        "phoneNum": "1234567890",
        "email": "johndoe@mail.com",
        "password": "nothashed"
    }

    x = requests.put(f'{domain}/user/{key}', json=payload_sample)
    assert x.status_code == 200


def test_delete_user():
    global domain
    time.sleep(3)
    key = "johndoe"
    print('Deleting user...')
    x = requests.delete(f'{domain}/user/{key}')
    assert x.status_code == 200


if __name__ == '__main__':
    print('Waiting for server to start...')
    test_create_user()
    test_get_user()
    test_update_user()
    test_delete_user()
