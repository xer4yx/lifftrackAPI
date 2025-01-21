import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from infrastructure import user_service_rtdb
from routers.UsersRouter import router

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.fixture
def mock_user_service():
    service = MagicMock()
    service.create_user = AsyncMock()
    service.get_user = AsyncMock()
    service.update_user = AsyncMock()
    service.delete_user = AsyncMock()
    service.verify_password = AsyncMock()
    service.get_user_password = AsyncMock()
    service.hash_password = AsyncMock()
    service.input_validator = MagicMock()
    service.input_validator.validate = MagicMock(return_value=True)
    return service

@pytest.fixture(autouse=True)
def patch_dependencies(mock_user_service):
    """Override the FastAPI dependency"""
    app.dependency_overrides[user_service_rtdb] = lambda: mock_user_service
    yield mock_user_service

@pytest.fixture
def valid_user_data():
    """Create valid user data that passes all validation"""
    return {
        "username": "johndoe",
        "email": "john@example.com",
        "password": "Pas$w0rd",
        "first_name": "John",
        "last_name": "Doe",
        "phone_number": "09123456789"
    }

@pytest.mark.asyncio
async def test_create_user_success(mock_user_service, valid_user_data):
    """Test successful user creation"""
    # Setup
    mock_user_service.create_user.return_value = True
    
    # Execute
    response = client.put("/user/create", json=valid_user_data)
    
    # Assert
    assert response.status_code == 201
    assert response.json() == {"msg": "User created"}
    mock_user_service.create_user.assert_called_once_with(
        username=valid_user_data["username"],
        email=valid_user_data["email"],
        password=valid_user_data["password"],
        first_name=valid_user_data["first_name"],
        last_name=valid_user_data["last_name"],
        phone_number=valid_user_data["phone_number"]
    )

@pytest.mark.asyncio
async def test_create_user_failure(mock_user_service, valid_user_data):
    """Test failed user creation"""
    # Setup
    mock_user_service.create_user.return_value = False
    
    # Execute
    response = client.put("/user/create", json=valid_user_data)
    
    # Assert
    assert response.status_code == 400
    assert response.json() == {"msg": "User creation failed."}
    mock_user_service.create_user.assert_called_once_with(
        username=valid_user_data["username"],
        email=valid_user_data["email"],
        password=valid_user_data["password"],
        first_name=valid_user_data["first_name"],
        last_name=valid_user_data["last_name"],
        phone_number=valid_user_data["phone_number"]
    )

@pytest.mark.asyncio
async def test_get_user_success(mock_user_service):
    """Test successful user retrieval"""
    # Setup
    expected_data = {
        "username": "johndoe",
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe"
    }
    mock_user_service.get_user.return_value = expected_data
    
    # Execute
    response = client.get(f"/user/{expected_data['username']}")
    
    # Assert
    assert response.status_code == 200
    assert response.json() == expected_data
    mock_user_service.get_user.assert_called_once_with(expected_data["username"])

@pytest.mark.asyncio
async def test_get_user_not_found(mock_user_service):
    """Test user retrieval when user doesn't exist"""
    # Setup
    mock_user_service.get_user.return_value = None
    
    # Execute
    response = client.get("/user/nonexistent")
    
    # Assert
    assert response.status_code == 404
    assert response.json() == {"msg": "User not found."}
    mock_user_service.get_user.assert_called_once_with("nonexistent")

@pytest.mark.asyncio
async def test_update_user_success(mock_user_service, valid_user_data):
    """Test successful user update"""
    # Setup
    mock_user_service.update_user.return_value = True
    
    # Execute
    response = client.put("/user/johndoe", json=valid_user_data)
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {"msg": "User updated successfully"}
    mock_user_service.update_user.assert_called_once_with(
        username="johndoe",
        user_data=mock_user_service.update_user.call_args[1]['user_data']
    )

@pytest.mark.asyncio
async def test_update_user_failure(mock_user_service, valid_user_data):
    """Test failed user update"""
    # Setup
    mock_user_service.update_user.return_value = False
    
    # Execute
    response = client.put("/user/johndoe", json=valid_user_data)
    
    # Assert
    assert response.status_code == 400
    assert response.json() == {"msg": "User update failed"}
    mock_user_service.update_user.assert_called_once_with(
        username="johndoe",
        user_data=mock_user_service.update_user.call_args[1]['user_data']
    )
    

@pytest.mark.asyncio
async def test_delete_user_success(mock_user_service, valid_user_data):
    """Test successful user deletion"""
    # Setup
    mock_user_service.delete_user.return_value = True
    
    # Execute
    response = client.delete(f"/user/{valid_user_data['username']}")
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {"msg": "User deleted."}
    mock_user_service.delete_user.assert_called_once_with("johndoe")

@pytest.mark.asyncio
async def test_delete_user_failure(mock_user_service, valid_user_data):
    """Test failed user deletion"""
    # Setup
    mock_user_service.delete_user.return_value = False
    
    # Execute
    response = client.delete(f"/user/{valid_user_data['username']}")
    
    # Assert
    assert response.status_code == 400
    assert response.json() == {"msg": "User deletion failed."}
    mock_user_service.delete_user.assert_called_once_with("johndoe")

@pytest.mark.asyncio
async def test_change_password_success(mock_user_service, valid_user_data):
    """Test successful password change"""
    # Setup
    mock_user_service.get_user_password.return_value = "hashed_password"
    mock_user_service.verify_password.return_value = True
    mock_user_service.hash_password.return_value = "new_hashed_password"
    mock_user_service.update_user.return_value = True
    
    password_data = valid_user_data.copy()
    password_data["current_password"] = valid_user_data["password"]
    password_data["password"] = "NewPas$w0rd"
    
    # Execute
    response = client.put("/user/johndoe/change-pass", json=password_data)
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {"msg": "Password changed."}
    
    mock_user_service.update_user.assert_called_once_with(
        "johndoe",
        mock_user_service.update_user.call_args[0][1]
    )

@pytest.mark.asyncio
async def test_change_password_incorrect_password(mock_user_service, valid_user_data):
    """Test password change with incorrect current password"""
    # Setup
    mock_user_service.get_user_password.return_value = "hashed_password"
    mock_user_service.verify_password.return_value = False
    
    mock_updated_data = valid_user_data.copy()
    mock_updated_data["new_password"] = "NewPas$w0rd"
    
    # Execute
    response = client.put(f"/user/{valid_user_data['username']}/change-pass", json=mock_updated_data)
    
    # Assert
    assert response.status_code == 400
    assert response.json() == {"msg": "Incorrect password."}
    mock_user_service.update_user.assert_not_called()
    
def teardown():
    """Restore the original dependency"""
    app.dependency_overrides.clear()
