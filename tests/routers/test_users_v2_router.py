import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.v2.UsersRouter import router
from infrastructure import user_service_admin

@pytest.fixture
def mock_user_service():
    """Mock UserService with common methods"""
    service = MagicMock()
    service.create_user = AsyncMock()
    service.get_user = AsyncMock()
    service.update_user = AsyncMock()
    service.delete_user = AsyncMock()
    service.input_validator = MagicMock()
    service.input_validator.validate = MagicMock(return_value=True)
    return service

@pytest.fixture
def app(mock_user_service):
    """Create test FastAPI app with router and dependencies"""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_service_admin] = lambda: mock_user_service
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)

def test_create_user_success(client, mock_user_service):
    """Test successful user creation"""
    # Setup
    mock_user_service.create_user.return_value = True
    
    # Execute
    response = client.post(
        "/v2/users",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "Password123!",
            "first_name": "Test",
            "last_name": "User",
            "phone_number": "09123456789"
        }
    )

    # Assert
    assert response.status_code == 201
    assert response.json() == {"msg": "User created successfully"}
    mock_user_service.create_user.assert_called_once()

def test_create_user_failure(client, mock_user_service):
    """Test failed user creation"""
    # Setup
    mock_user_service.create_user.return_value = False
    
    # Execute
    response = client.post(
        "/v2/users",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "Password123!",
            "first_name": "Test",
            "last_name": "User",
            "phone_number": "1234567890"
        }
    )
    
    # Assert
    assert response.status_code == 400
    assert response.json() == {"msg": "User creation failed."}

def test_get_user_success(client, mock_user_service):
    """Test successful user retrieval"""
    # Setup
    mock_user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User"
    }
    mock_user_service.get_user.return_value = mock_user_data

    # Execute
    response = client.get("/v2/users/testuser")
    
    # Assert
    assert response.status_code == 200
    assert response.json() == mock_user_data
    mock_user_service.get_user.assert_called_once_with("testuser")

def test_get_user_not_found(client, mock_user_service):
    """Test user retrieval when user doesn't exist"""
    # Setup
    mock_user_service.get_user.return_value = None
    
    # Execute
    response = client.get("/v2/users/testuser")
    
    # Assert
    assert response.status_code == 404
    assert response.json() == {"msg": "User not found"}

def test_update_user_success(client, mock_user_service):
    """Test successful user update"""
    # Setup
    mock_user_service.get_user.return_value = {"username": "testuser"}
    mock_user_service.update_user.return_value = True
    
    # Execute
    response = client.put(
        "/v2/users/testuser",
        json={"first_name": "Updated", "last_name": "User"}
    )
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {"msg": "User updated successfully"}

def test_delete_user_success(client, mock_user_service):
    """Test successful user deletion"""
    # Setup
    mock_user_service.get_user.return_value = {"username": "testuser"}
    mock_user_service.delete_user.return_value = True
    
    # Execute
    response = client.delete("/v2/users/testuser")
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {"msg": "User deleted successfully"}

def test_delete_user_not_found(client, mock_user_service):
    """Test user deletion when user doesn't exist"""
    # Setup
    mock_user_service.get_user.return_value = None
    
    # Execute
    response = client.delete("/v2/users/testuser")
    
    # Assert
    assert response.status_code == 404
    assert response.json() == {"msg": "User not found"}