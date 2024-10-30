from lifttrack.rtdbHelper import RTDBHelper


class TestRTDBHelper:
    rtdb = RTDBHelper()

    def test_create_user(self):
        payload = {
            "id": "12",
            "name": "John Doe",
            "username": "johndoe",
            "email": "johndoe@mail.com",
            "password": "password"
        }

        self.rtdb.put_data(payload)

    def test_delete_user(self):
        self.rtdb.delete_data("johndoe")


if __name__ == '__main__':
    TestRTDBHelper().test_create_user()
