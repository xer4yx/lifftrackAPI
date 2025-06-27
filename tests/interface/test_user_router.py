import pytest
from fastapi import status, FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from interface.routers.user_router import user_router
from core.entities.user_entity import UserEntity
from lifttrack.models.user_schema import User, UserUpdate, UserResponse
from core.usecase import UserUseCase
from interface.di import get_user_service_rest, get_current_user


# Setup test app with the router
@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(user_router)
    return app


# Setup test client
@pytest.fixture
def client(app):
    return TestClient(app)


# Mock UserUseCase for dependency injection
@pytest.fixture
def mock_user_service():
    mock_service = AsyncMock(spec=UserUseCase)
    return mock_service


# Mock the current authenticated user for dependency injection
@pytest.fixture
def mock_current_user():
    return UserEntity(
        id="test_user_id",
        username="testuser",
        email="test@example.com",
        password="hashed_password",
        first_name="Test",
        last_name="User",
        phone_number="1234567890",
        profile_picture=None,
        is_deleted=False,
        is_authenticated=True,
        created_at=datetime.now(timezone.utc),
        updated_at=None,
        last_login=None,
    )


# Apply dependency overrides for all tests
@pytest.fixture(autouse=True)
def override_dependencies(app, mock_user_service, mock_current_user):
    def get_user_service_override():
        return mock_user_service

    def get_current_user_override():
        return mock_current_user

    # Set dependency overrides on the FastAPI app, not on the router
    app.dependency_overrides[get_user_service_rest] = get_user_service_override
    app.dependency_overrides[get_current_user] = get_current_user_override

    yield

    # Clear overrides after tests
    app.dependency_overrides = {}


# Tests for register_user endpoint
@pytest.mark.asyncio
async def test_register_user_success(client: TestClient, mock_user_service: AsyncMock):
    # Arrange
    user_data = {
        "username": "newuser",
        "email": "new@example.com",
        "password": "Pa$sword123",
        "fname": "New",
        "lname": "User",
        "phoneNum": "09123456789",
        "pfp": None,
    }

    created_user = UserEntity(
        id="new_user_id",
        username="newuser",
        email="new@example.com",
        password="hashed_password",  # The hashed version in the database
        first_name="New",
        last_name="User",
        phone_number="09123456789",
        profile_picture=None,
        is_deleted=False,
        is_authenticated=False,
        created_at=datetime.now(timezone.utc),
        updated_at=None,
        last_login=None,
    )

    # Setup mock return value
    mock_user_service.register_user.return_value = (True, created_user, None)

    # Act
    response = client.post("/v3/user", json=user_data)

    # Assert
    assert response.status_code == status.HTTP_201_CREATED
    assert "password" not in response.json()
    assert response.json()["username"] == "newuser"
    assert response.json()["email"] == "new@example.com"

    # Verify mock was called with correct args
    mock_user_service.register_user.assert_called_once_with(
        username="newuser",
        email="new@example.com",
        password="Pa$sword123",
        first_name="New",
        last_name="User",
        phone_number="09123456789",
        profile_picture=None,
    )


@pytest.mark.asyncio
async def test_register_user_failure(client, mock_user_service):
    # Arrange
    user_data = {
        "username": "existinguser",
        "email": "existing@example.com",
        "password": "Pa$sword123",
        "fname": "Existing",
        "lname": "User",
        "phoneNum": "09123456789",
        "pfp": None,
    }

    # Setup mock to return failure
    mock_user_service.register_user.return_value = (
        False,
        None,
        "Username already exists",
    )

    # Act
    response = client.post("/v3/user", json=user_data)

    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Username already exists"


# Tests for get_profile endpoint
@pytest.mark.asyncio
async def test_get_profile_success(client, mock_current_user):
    # Act
    with patch(
        "interface.routers.user_router.get_current_user", return_value=mock_current_user
    ):
        response = client.get("/v3/user/profile")

    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["username"] == "testuser"
    assert response.json()["email"] == "test@example.com"
    assert "password" not in response.json()


# Tests for get_user endpoint
@pytest.mark.asyncio
async def test_get_user_success(client, mock_user_service, mock_current_user):
    # Arrange
    user_id = "test_user_id"  # Same as mock_current_user.id

    # Setup mock return value
    mock_user_service.get_user_profile.return_value = mock_current_user

    # Act
    with patch(
        "interface.routers.user_router.get_user_service_rest",
        return_value=mock_user_service,
    ):
        with patch(
            "interface.routers.user_router.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.get(f"/v3/user/{user_id}")

    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["username"] == "testuser"
    assert "password" not in response.json()

    # Verify mock was called correctly with username, not user_id
    # The router passes current_user.username to the service
    mock_user_service.get_user_profile.assert_called_once_with("testuser")


@pytest.mark.asyncio
async def test_get_user_forbidden(client, mock_current_user):
    # Arrange
    different_user_id = "different_user_id"  # Different from mock_current_user.id

    # Act
    with patch(
        "interface.routers.user_router.get_current_user", return_value=mock_current_user
    ):
        response = client.get(f"/v3/user/{different_user_id}")

    # Assert
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Not authorized" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_user_not_found(client, mock_user_service, mock_current_user):
    # Arrange
    user_id = "test_user_id"  # Same as mock_current_user.id

    # Setup mock to return None (user not found)
    mock_user_service.get_user_profile.return_value = None

    # Act
    with patch(
        "interface.routers.user_router.get_user_service_rest",
        return_value=mock_user_service,
    ):
        with patch(
            "interface.routers.user_router.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.get(f"/v3/user/{user_id}")

    # Assert
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "User not found" in response.json()["detail"]


# Tests for update_user endpoint
@pytest.mark.asyncio
async def test_update_user_success(client, mock_user_service, mock_current_user):
    # Arrange
    user_id = "test_user_id"  # Same as mock_current_user.id
    update_data = {
        "fname": "Updated",
        "lname": "Name",
        "email": "updated@example.com",
        "phoneNum": "9876543210",
        "pfp": "http://example.com/new_pic.jpg",
    }

    updated_user = UserEntity(
        id=user_id,
        username="testuser",
        email="updated@example.com",
        password="hashed_password",
        first_name="Updated",
        last_name="Name",
        phone_number="9876543210",
        profile_picture="http://example.com/new_pic.jpg",
        is_deleted=False,
        is_authenticated=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        last_login=None,
    )

    # Setup mock return value
    mock_user_service.update_user_profile.return_value = (True, updated_user, None)

    # Act
    with patch(
        "interface.routers.user_router.get_user_service_rest",
        return_value=mock_user_service,
    ):
        with patch(
            "interface.routers.user_router.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.put(f"/v3/user/{user_id}", json=update_data)

    # Assert
    assert response.status_code == status.HTTP_200_OK

    # Check response format based on UserCreateResponse schema
    response_data = response.json()
    assert response_data["email"] == "updated@example.com"
    assert response_data["fname"] == "Updated"  # Using original field names in response
    assert response_data["lname"] == "Name"
    assert response_data["phoneNum"] == "9876543210"
    assert response_data["pfp"] == "http://example.com/new_pic.jpg"
    assert "password" not in response_data

    # Verify correct mapping of fields in update call
    mock_user_service.update_user_profile.assert_called_once()
    call_args = mock_user_service.update_user_profile.call_args[1]
    assert call_args["user_id"] == "testuser"  # Username is used as user_id
    assert call_args["update_data"]["first_name"] == "Updated"
    assert call_args["update_data"]["last_name"] == "Name"
    assert call_args["update_data"]["email"] == "updated@example.com"


@pytest.mark.asyncio
async def test_update_user_forbidden(client, mock_current_user):
    # Arrange
    different_user_id = "different_user_id"  # Different from mock_current_user.id
    update_data = {"fname": "New"}

    # Act
    with patch(
        "interface.routers.user_router.get_current_user", return_value=mock_current_user
    ):
        response = client.put(f"/v3/user/{different_user_id}", json=update_data)

    # Assert
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Not authorized" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_user_password_rejected(
    client, mock_user_service, mock_current_user
):
    # Arrange
    user_id = "test_user_id"  # Same as mock_current_user.id
    update_data = {"password": "new_password"}

    # Act
    with patch(
        "interface.routers.user_router.get_user_service_rest",
        return_value=mock_user_service,
    ):
        with patch(
            "interface.routers.user_router.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.put(f"/v3/user/{user_id}", json=update_data)

    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Cannot update password" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_user_failure(client, mock_user_service, mock_current_user):
    # Arrange
    user_id = "test_user_id"  # Same as mock_current_user.id
    update_data = {"email": "invalid_email"}

    # Setup mock to return failure
    mock_user_service.update_user_profile.return_value = (
        False,
        None,
        "Invalid email format",
    )

    # Act
    with patch(
        "interface.routers.user_router.get_user_service_rest",
        return_value=mock_user_service,
    ):
        with patch(
            "interface.routers.user_router.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.put(f"/v3/user/{user_id}", json=update_data)

    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Invalid email format"


# Tests for delete_user endpoint
@pytest.mark.asyncio
async def test_delete_user_success(client, mock_user_service, mock_current_user):
    # Arrange
    user_id = "test_user_id"  # Same as mock_current_user.id

    # Setup mock return value
    mock_user_service.delete_user.return_value = (True, None)

    # Act
    with patch(
        "interface.routers.user_router.get_user_service_rest",
        return_value=mock_user_service,
    ):
        with patch(
            "interface.routers.user_router.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.delete(f"/v3/user/{user_id}")

    # Assert
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert response.content == b""  # Empty response body

    # Verify mock was called correctly with username, not ID
    # The router passes current_user.username to the service
    mock_user_service.delete_user.assert_called_once_with("testuser")


@pytest.mark.asyncio
async def test_delete_user_forbidden(client, mock_current_user):
    # Arrange
    different_user_id = "different_user_id"  # Different from mock_current_user.id

    # Act
    with patch(
        "interface.routers.user_router.get_current_user", return_value=mock_current_user
    ):
        response = client.delete(f"/v3/user/{different_user_id}")

    # Assert
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Not authorized" in response.json()["detail"]


@pytest.mark.asyncio
async def test_delete_user_failure(client, mock_user_service, mock_current_user):
    # Arrange
    user_id = "test_user_id"  # Same as mock_current_user.id

    # Setup mock to return failure
    mock_user_service.delete_user.return_value = (
        False,
        "Failed to delete user from database",
    )

    # Act
    with patch(
        "interface.routers.user_router.get_user_service_rest",
        return_value=mock_user_service,
    ):
        with patch(
            "interface.routers.user_router.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.delete(f"/v3/user/{user_id}")

    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Failed to delete user" in response.json()["detail"]


# Tests for change_password endpoint
@pytest.mark.asyncio
async def test_change_password_success(client, mock_user_service, mock_current_user):
    # Arrange
    user_id = "test_user_id"  # Same as mock_current_user.id

    # Setup mock return value
    mock_user_service.change_user_password.return_value = (True, None)

    # Act
    with patch(
        "interface.routers.user_router.get_user_service_rest",
        return_value=mock_user_service,
    ):
        with patch(
            "interface.routers.user_router.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                f"/v3/user/{user_id}/change-password",
                params={
                    "old_password": "current_password",
                    "new_password": "new_password",
                },
            )

    # Assert
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert response.content == b""  # Empty response body

    # Verify mock was called correctly using username, not ID
    # The router passes current_user.username to the service
    mock_user_service.change_user_password.assert_called_once_with(
        user_id="testuser",  # Username is used as user_id
        old_password="current_password",
        new_password="new_password",
    )


@pytest.mark.asyncio
async def test_change_password_forbidden(client, mock_current_user):
    # Arrange
    different_user_id = "different_user_id"  # Different from mock_current_user.id

    # Act
    with patch(
        "interface.routers.user_router.get_current_user", return_value=mock_current_user
    ):
        response = client.post(
            f"/v3/user/{different_user_id}/change-password",
            params={"old_password": "current_password", "new_password": "new_password"},
        )

    # Assert
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Not authorized" in response.json()["detail"]


@pytest.mark.asyncio
async def test_change_password_incorrect_old_password(
    client, mock_user_service, mock_current_user
):
    # Arrange
    user_id = "test_user_id"  # Same as mock_current_user.id

    # Setup mock to return failure
    mock_user_service.change_user_password.return_value = (
        False,
        "Current password is incorrect",
    )

    # Act
    with patch(
        "interface.routers.user_router.get_user_service_rest",
        return_value=mock_user_service,
    ):
        with patch(
            "interface.routers.user_router.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                f"/v3/user/{user_id}/change-password",
                params={
                    "old_password": "wrong_password",
                    "new_password": "new_password",
                },
            )

    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Current password is incorrect" in response.json()["detail"]
