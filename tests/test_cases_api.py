import unittest
from unittest import TestCase
from fastapi.testclient import TestClient

from main import app
from lifttrack import os, logging

logger = logging.getLogger("test_cases_api")
handler = logging.FileHandler(
    filename=os.path.join(os.path.dirname(__file__), "..", "logs/test_cases.log"),
    mode="a"
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

client = TestClient(app)


class TestHttpPutMethod(TestCase):
    def setUp(self):
        logger.info(f"Setting up {self.__class__.__name__}")

        # Initial payload for user creation
        self.payload = {
            "fname": "Test",
            "lname": "User",
            "username": "testuser",
            "phoneNum": "1234567890",
            "email": "test@example.com",
            "password": "testpassword",
        }

        # Updated data for user
        self.updated_data = {
            "id": "123",
            "fname": "Updated",
            "lname": "User",
            "username": "testuser",
            "phoneNum": "9876543210",
            "email": "updated@example.com",
            "password": "newpassword",
            "pfp": "new_profile.jpg",
            "isAuthenticated": True,
            "isDeleted": False
        }

    def test_create_user(self):
        logger.info(f"Started testing {self.test_create_user.__name__}")
        response = client.put("/user/create", json=self.payload)

        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Content: {response.content}")

        self.assertNotEqual(response.status_code, 400)
        self.assertNotEqual(response.status_code, 406)

        self.assertEqual(response.status_code, 200)
        logger.info(f"{self.test_create_user.__name__} testing completed.")

    def test_update_user_data(self):
        logger.info(f"Started testing {self.test_update_user_data.__name__}")
        response = client.put("/user/testuser", json=self.updated_data)

        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Content: {response.content}")

        # Check if the response is not 404
        self.assertNotEqual(response.status_code, 404)
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the updated data
        self.assertNotEqual(response, None)
        self.assertEqual(response.json().get("content").get("fname"), 'Updated')
        self.assertEqual(response.json().get("content").get("isAuthenticated"), True)

        logger.info(f"{self.test_update_user_data.__name__} testing completed.")

    def test_change_password(self):
        logger.info(f"Started testing {self.test_change_password.__name__}")
        response = client.put("/user/testuser/change-pass", json=self.updated_data)

        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Content: {response.content}")

        # Check if the response is not 404
        self.assertNotEqual(response.status_code, 404)
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the updated data
        self.assertNotEqual(response, None)
        self.assertEqual(response.json().get("msg"), "Password changed successfully.")

        logger.info(f"{self.test_change_password.__name__} testing completed.")


class TestHttpPostMethod(TestCase):
    def setUp(self):
        logger.info(f"Setting up {self.__class__.__name__}")

        # Initial payload for user creation
        self.payload = {
            "username": "testuser",
            "password": "testpassword"
        }

    def test_login_user(self):
        logger.info(f"Started testing {self.test_login_user.__name__}")
        response = client.post("/login", json=self.payload)

        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Content: {response.content}")

        self.assertNotEqual(response.status_code, 401)
        self.assertNotEqual(response.status_code, 404)
        self.assertEqual(response.status_code, 200)

        self.assertNotEqual(response.json(), None)
        self.assertEqual(response.json().get("success"), True)

        logger.info(f"{self.test_login_user.__name__} testing completed.")

    def test_login_for_access_token(self):
        logger.info(f"Started testing {self.test_login_for_access_token.__name__}")
        response = client.post("/token", json=self.payload)

        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Content: {response.content}")

        self.assertNotEqual(response.status_code, 401)
        self.assertNotEqual(response.status_code, 404)
        self.assertEqual(response.status_code, 200)

        self.assertNotEqual(response.json(), None)
        self.assertEqual(response.json().get("access_token"), None)
        self.assertEqual(response.json().get("access_token"), str)

        logger.info(f"{self.test_login_for_access_token.__name__} testing completed.")


class TestHttpGetMethod(TestCase):
    def setUp(self):
        logger.info(f"Setting up {self.__class__.__name__}")

    def test_read_root(self):
        logger.info(f"Started testing {self.test_read_root.__name__}")
        response = client.get("/")

        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Content: {response.content}")

        self.assertNotEqual(response.status_code, 400)
        self.assertNotEqual(response.status_code, 404)
        self.assertEqual(response.status_code, 200)

        self.assertNotEqual(response.json(), None)
        self.assertEqual(response.json().get("msg"), "Welcome to LiftTrack!")

        logger.info(f"{self.test_read_root.__name__} testing completed.")

    def test_get_app_info(self):
        logger.info(f"Started testing {self.test_get_app_info.__name__}")
        response = client.get("/app-info")

        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Content: {response.content}")

        self.assertNotEqual(response.status_code, 400)
        self.assertNotEqual(response.status_code, 404)
        self.assertEqual(response.status_code, 200)

        self.assertNotEqual(response.json(), None)

        logger.info(f"{self.test_get_app_info.__name__} testing completed.")
    
    def test_get_user_data(self):
        logger.info(f"Started testing {self.test_get_user_data.__name__}")
        response = client.get("/user/testuser")

        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Content: {response.content}")

        # Check if the response is not 404
        self.assertNotEqual(response.status_code, 404)

        # Check if the response contains the user data
        self.assertNotEqual(response, None)
        self.assertEqual(response.json().get("username"), "testuser")
        self.assertEqual(response.json().get("isAuthenticated"), False)

        logger.info(f"{self.test_get_user_data.__name__} testing completed.")

    def test_read_users_me(self):
        logger.info(f"Started testing {self.test_read_users_me.__name__}")
        response = client.get("/users/me/")

        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Content: {response.content}")

        # Check if the response is not 404
        self.assertNotEqual(response.status_code, 401)
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the user data
        self.assertNotEqual(response, None)
        self.assertEqual(response.json().get("username"), "testuser")
        self.assertEqual(response.json().get("isAuthenticated"), False)

        logger.info(f"{self.test_read_users_me.__name__} testing completed.")


class TestHttpDeleteMethod(TestCase):
    def setUp(self):
        logger.info(f"Setting up {self.__class__.__name__}")

    def test_vdelete_user(self):
        logger.info(f"Started testing {self.test_vdelete_user.__name__}")
        response = client.delete("/user/testuser")

        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Content: {response.content}")

        # Check if the response is not 404
        self.assertNotEqual(response.status_code, 404)
        self.assertEqual(response.status_code, 200)

        self.assertNotEqual(response, None)

        logger.info(f"{self.test_vdelete_user.__name__} testing completed.")


if __name__ == '__main__':
    unittest.main()
