import pytest
from fastapi import status, FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

from interface.routers.auth_router import auth_router
from core.entities.auth_entity import TokenEntity, CredentialsEntity
from core.entities.user_entity import UserEntity
from core.usecase.auth_usecase import AuthUseCase
from interface.di import get_auth_service, get_current_user
from infrastructure.auth.exceptions import (
    InvalidCredentialsError,
    ValidationError,
    InvalidPasswordError
)


# Setup test app with the router
@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(auth_router)
    return app


# Setup test client
@pytest.fixture
def client(app):
    return TestClient(app)


# Mock AuthUseCase for dependency injection
@pytest.fixture
def mock_auth_service():
    mock_service = AsyncMock(spec=AuthUseCase)
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
        last_login=None
    )


# Apply dependency overrides for all tests
@pytest.fixture(autouse=True)
def override_dependencies(app, mock_auth_service, mock_current_user):
    def get_auth_service_override():
        return mock_auth_service

    def get_current_user_override():
        return mock_current_user

    # Set dependency overrides on the FastAPI app
    app.dependency_overrides[get_auth_service] = get_auth_service_override
    app.dependency_overrides[get_current_user] = get_current_user_override
    
    yield
    
    # Clear overrides after tests
    app.dependency_overrides = {}


# Tests for login endpoint (/v2/auth/token)
@pytest.mark.asyncio
async def test_login_success(client, mock_auth_service):
    # Arrange
    form_data = {
        "username": "testuser",
        "password": "Password123!"
    }
    
    # Create token to return
    expiry_time = datetime.now(timezone.utc) + timedelta(minutes=30)
    token = TokenEntity(
        access_token="valid_token_string",
        token_type="bearer",
        expires_at=expiry_time
    )
    
    user = UserEntity(
        id="test_user_id",
        username="testuser",
        email="test@example.com",
        password="hashed_password",
        first_name="Test",
        last_name="User",
        phone_number="1234567890"
    )
    
    # Setup mock return value
    mock_auth_service.login.return_value = (True, token, user, None)
    
    # Act
    response = client.post("/v2/auth/token", data=form_data)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["access_token"] == "valid_token_string"
    assert response.json()["token_type"] == "bearer"
    
    # Verify the correct credentials were passed to login
    expected_credentials = CredentialsEntity(
        username="testuser",
        password="Password123!"
    )
    
    called_credentials = mock_auth_service.login.call_args[0][0]
    assert called_credentials.username == expected_credentials.username
    assert called_credentials.password == expected_credentials.password


@pytest.mark.asyncio
async def test_login_failure_invalid_credentials(client, mock_auth_service):
    # Arrange
    form_data = {
        "username": "testuser",
        "password": "wrong_password"
    }
    
    # Setup mock to return failure
    error_message = "Invalid username or password"
    mock_auth_service.login.return_value = (False, None, None, error_message)
    
    # Act
    response = client.post("/v2/auth/token", data=form_data)
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == error_message


@pytest.mark.asyncio
async def test_login_with_special_chars(client, mock_auth_service):
    # Arrange - Test with special characters in credentials
    form_data = {
        "username": "test@user+123",
        "password": "P@$$w0rd!*&"
    }
    
    # Create token to return
    expiry_time = datetime.now(timezone.utc) + timedelta(minutes=30)
    token = TokenEntity(
        access_token="valid_token_string",
        token_type="bearer",
        expires_at=expiry_time
    )
    
    user = UserEntity(
        id="test_user_id",
        username="test@user+123",
        email="test@example.com",
        password="hashed_password",
        first_name="Test",
        last_name="User",
        phone_number="1234567890"
    )
    
    # Setup mock return value
    mock_auth_service.login.return_value = (True, token, user, None)
    
    # Act
    response = client.post("/v2/auth/token", data=form_data)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    
    # Verify the special characters were handled correctly
    called_credentials = mock_auth_service.login.call_args[0][0]
    assert called_credentials.username == "test@user+123"
    assert called_credentials.password == "P@$$w0rd!*&"


# Tests for logout endpoint (/v2/auth/logout)
@pytest.mark.asyncio
async def test_logout_success(client, mock_auth_service, mock_current_user):
    # Arrange
    mock_auth_service.logout.return_value = True
    
    # Act
    response = client.post("/v2/auth/logout")
    
    # Assert
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert response.content == b''
    
    # Verify the token from get_current_user was used
    mock_auth_service.logout.assert_called_once()
    # We don't know exactly what token was passed, but we can verify it was called


@pytest.mark.asyncio
async def test_logout_failure(client, mock_auth_service):
    # Arrange
    mock_auth_service.logout.return_value = False
    
    # Act
    response = client.post("/v2/auth/logout")
    
    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# Tests for refresh token endpoint (/v2/auth/refresh)
@pytest.mark.asyncio
async def test_refresh_token_success(client, mock_auth_service):
    # Arrange
    refresh_token = "valid_refresh_token"
    
    # Create new token to return
    expiry_time = datetime.now(timezone.utc) + timedelta(minutes=30)
    token = TokenEntity(
        access_token="new_access_token",
        token_type="bearer",
        expires_at=expiry_time
    )
    
    # Setup mock return value
    mock_auth_service.refresh_token.return_value = (True, token, None)
    
    # Act - Use query parameter for refresh_token
    response = client.post(f"/v2/auth/refresh?refresh_token={refresh_token}")
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["access_token"] == "new_access_token"
    assert response.json()["token_type"] == "bearer"
    
    # Verify the refresh token was passed correctly
    mock_auth_service.refresh_token.assert_called_once_with(refresh_token)


@pytest.mark.asyncio
async def test_refresh_token_invalid(client, mock_auth_service):
    # Arrange
    refresh_token = "invalid_refresh_token"
    
    # Setup mock to return failure
    error_message = "Invalid or expired refresh token"
    mock_auth_service.refresh_token.return_value = (False, None, error_message)
    
    # Act - Use query parameter for refresh_token
    response = client.post(f"/v2/auth/refresh?refresh_token={refresh_token}")
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == error_message


@pytest.mark.asyncio
async def test_refresh_token_expired(client, mock_auth_service):
    # Arrange
    refresh_token = "expired_refresh_token"
    
    # Setup mock to return failure with specific expired message
    error_message = "Refresh token has expired"
    mock_auth_service.refresh_token.return_value = (False, None, error_message)
    
    # Act - Use query parameter for refresh_token
    response = client.post(f"/v2/auth/refresh?refresh_token={refresh_token}")
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == error_message


# Tests for change password endpoint (/v2/auth/change-password)
@pytest.mark.asyncio
async def test_change_password_success(client, mock_auth_service, mock_current_user):
    # Arrange
    old_password = "OldPassword123!"
    new_password = "NewPassword456!"
    
    # Setup mock return value
    mock_auth_service.change_password.return_value = (True, None)
    
    # Act - Use query parameters for old_password and new_password
    response = client.post(f"/v2/auth/change-password?old_password={old_password}&new_password={new_password}")
    
    # Assert
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert response.content == b''
    
    # Verify change_password was called with correct parameters
    mock_auth_service.change_password.assert_called_once_with(
        mock_current_user.id,
        old_password,
        new_password
    )


@pytest.mark.asyncio
async def test_change_password_incorrect_old_password(client, mock_auth_service):
    # Arrange
    old_password = "WrongOldPassword"
    new_password = "NewPassword456!"
    
    # Setup mock to return failure
    error_message = "Current password is incorrect"
    mock_auth_service.change_password.return_value = (False, error_message)
    
    # Act - Use query parameters for old_password and new_password
    response = client.post(f"/v2/auth/change-password?old_password={old_password}&new_password={new_password}")
    
    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json()["detail"] == error_message


@pytest.mark.asyncio
async def test_change_password_weak_new_password(client, mock_auth_service):
    # Arrange
    old_password = "OldPassword123!"
    new_password = "weak"
    
    # Setup mock to return failure with password requirements error
    error_message = "Password does not meet security requirements: minimum 8 characters, must include uppercase, lowercase, number and special character"
    mock_auth_service.change_password.return_value = (False, error_message)
    
    # Act - Use query parameters for old_password and new_password
    response = client.post(f"/v2/auth/change-password?old_password={old_password}&new_password={new_password}")
    
    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json()["detail"] == error_message


@pytest.mark.asyncio
async def test_change_password_same_as_old(client, mock_auth_service):
    # Arrange - Edge case: new password same as old
    old_password = "Password123!"
    new_password = "Password123!"
    
    # Setup mock to return failure
    error_message = "New password cannot be the same as the current password"
    mock_auth_service.change_password.return_value = (False, error_message)
    
    # Act - Use query parameters for old_password and new_password
    response = client.post(f"/v2/auth/change-password?old_password={old_password}&new_password={new_password}")
    
    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json()["detail"] == error_message 