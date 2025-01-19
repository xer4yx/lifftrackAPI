import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from core.services.user_service import UserService
from core.entities.user import User
from core.exceptions.database import DatabaseError, QueryError, ValidationError
from lifttrack.dbhandler.rest_rtdb import RTDBHelper
from lifttrack.auth import LiftTrackAuthenticator

@pytest.fixture
def mock_database():
    db = MagicMock(spec=RTDBHelper)
    db.set = AsyncMock()
    db.get = AsyncMock()
    db.update = AsyncMock()
    db.delete = AsyncMock()
    db.query = AsyncMock()
    return db

@pytest.fixture
def mock_auth_service():
    auth = MagicMock(spec=LiftTrackAuthenticator)
    auth.hash_password = MagicMock(return_value="hashed_password")
    auth.verify_password = MagicMock(return_value=True)
    auth.validate = MagicMock(return_value=True)
    return auth

@pytest.fixture
def user_service(mock_database, mock_auth_service):
    return UserService(
        database=mock_database,
        password_service=mock_auth_service,
        input_validator=mock_auth_service
    )

@pytest.fixture
def valid_user_data():
    return {
        "username": "johndoe",
        "email": "john@example.com",
        "password": "Pas$w0rd",
        "first_name": "John",
        "last_name": "Doe",
        "phone_number": "09123456789",
        "is_deleted": False
    }

@pytest.mark.asyncio
async def test_create_user_success(user_service, valid_user_data, mock_database):
    # Setup
    mock_database.set.return_value = "user_id"
    
    # Execute
    result = await user_service.create_user(
        username=valid_user_data["username"],
        email=valid_user_data["email"],
        password=valid_user_data["password"],
        first_name=valid_user_data["first_name"],
        last_name=valid_user_data["last_name"],
        phone_number=valid_user_data["phone_number"]
    )
    
    # Assert
    assert result == "user_id"
    mock_database.set.assert_called_once()

@pytest.mark.asyncio
async def test_create_user_validation_error(user_service, valid_user_data, mock_database, mock_auth_service):
    # Setup
    mock_auth_service.validate.return_value = False
    
    # Execute & Assert
    with pytest.raises(ValidationError):
        await user_service.create_user(
            username=valid_user_data["username"],
            email=valid_user_data["email"],
            password=valid_user_data["password"],
            first_name=valid_user_data["first_name"],
            last_name=valid_user_data["last_name"],
            phone_number=valid_user_data["phone_number"]
        )
    
    mock_database.set.assert_not_called()

@pytest.mark.asyncio
async def test_get_user_success(user_service, valid_user_data, mock_database):
    # Setup
    mock_database.get.return_value = valid_user_data
    
    # Execute
    result = await user_service.get_user(valid_user_data["username"])
    
    # Assert
    assert result == valid_user_data
    mock_database.get.assert_called_once_with(path="users", key=valid_user_data["username"])

@pytest.mark.asyncio
async def test_get_user_not_found(user_service, mock_database):
    # Setup
    mock_database.get.return_value = None
    
    # Execute
    result = await user_service.get_user("nonexistent")
    
    # Assert
    assert result is None
    mock_database.get.assert_called_once_with(path="users", key="nonexistent")

@pytest.mark.asyncio
async def test_update_user_success(user_service, valid_user_data, mock_database):
    # Setup
    mock_database.update.return_value = True
    user = User(**valid_user_data)
    
    # Execute
    result = await user_service.update_user(valid_user_data["username"], user)
    
    # Assert
    assert result is True
    mock_database.update.assert_called_once()

@pytest.mark.asyncio
async def test_delete_user_success(user_service, valid_user_data, mock_database):
    # Setup
    mock_database.query.return_value = [valid_user_data]
    mock_database.delete.return_value = True
    
    # Execute
    result = await user_service.delete_user(valid_user_data["username"])
    
    # Assert
    assert result is True
    mock_database.delete.assert_called_once_with(path="users", key=valid_user_data["username"])

@pytest.mark.asyncio
async def test_verify_password(user_service, mock_auth_service):
    # Setup
    plain_password = "password123"
    hashed_password = "hashed_password"
    
    # Execute
    result = await user_service.verify_password(plain_password, hashed_password)
    
    # Assert
    assert result is True
    mock_auth_service.verify_password.assert_called_once_with(plain_password, hashed_password)

@pytest.mark.asyncio
async def test_hash_password(user_service, mock_auth_service):
    # Setup
    password = "password123"
    
    # Execute
    result = await user_service.hash_password(password)
    
    # Assert
    assert result == "hashed_password"
    mock_auth_service.hash_password.assert_called_once_with(password) 