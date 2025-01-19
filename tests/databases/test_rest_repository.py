import pytest
from unittest.mock import Mock, patch, call
from firebase import firebase
import asyncio
import concurrent.futures
from functools import partial

from infrastructure.database.firebase.rest_repository import FirebaseRestRepository
from core.exceptions import QueryError

@pytest.fixture
def mock_firebase_app():
    mock_app = Mock()
    # Setup default return values for firebase methods
    mock_app.put.return_value = "test-key"
    mock_app.post.return_value = {"name": "test-key"}
    mock_app.get.return_value = {"test": "data"}
    mock_app.delete.return_value = None
    return mock_app

@pytest.fixture
def mock_firebase(mock_firebase_app):
    with patch('firebase.firebase.FirebaseApplication', return_value=mock_firebase_app) as mock:
        yield mock

@pytest.fixture
def mock_future():
    # Create a new Future object for each test
    return concurrent.futures.Future()

@pytest.fixture
def firebase_repo(mock_firebase):
    # Return the repository instance directly, not as an async generator
    return FirebaseRestRepository(
        dsn="https://test-db.firebaseio.com",
        auth_token="test-token"
    )

@pytest.mark.asyncio
async def test_set_with_key(firebase_repo, mock_firebase_app):
    with patch('asyncio.get_event_loop') as mock_loop:
        future = asyncio.Future()
        
        # Mock the run_in_executor to actually execute the function
        async def mock_run_in_executor(executor, func, *args, **kwargs):
            result = func(*args, **kwargs)
            future.set_result(result)
            return future.result()
            
        mock_loop.return_value.run_in_executor.side_effect = mock_run_in_executor
        
        data = {"name": "test"}
        result = await firebase_repo.set("users", data, "user1")
        
        mock_firebase_app.put.assert_called_once_with("users", "user1", data)
        assert result == "test-key"

@pytest.mark.asyncio
async def test_push(firebase_repo, mock_firebase_app):
    with patch('asyncio.get_event_loop') as mock_loop:
        future = asyncio.Future()
        
        # Mock the run_in_executor to actually execute the function and return the result
        async def mock_run_in_executor(executor, func, *args, **kwargs):
            result = func(*args, **kwargs)
            future.set_result(result)
            return result  # Return the result directly, not the future
            
        mock_loop.return_value.run_in_executor.side_effect = mock_run_in_executor
        
        data = {"name": "test"}
        result = await firebase_repo.push("users", data)
        
        mock_firebase_app.post.assert_called_once_with("users", data)
        assert result == "test-key"

@pytest.mark.asyncio
async def test_get(firebase_repo, mock_firebase_app):
    result = await firebase_repo.get("users", "user1")
    
    mock_firebase_app.get.assert_called_once_with("users", "user1")
    assert result == {"test": "data"}

@pytest.mark.asyncio
async def test_update(firebase_repo, mock_firebase_app):
    # Setup mock for get and put
    existing_data = {"name": "old", "age": 25}
    update_data = {"name": "new"}
    expected_data = {"name": "new", "age": 25}
    
    mock_firebase_app.get.return_value = existing_data
    mock_firebase_app.put.return_value = None
    
    result = await firebase_repo.update("users", "user1", update_data)
    
    mock_firebase_app.get.assert_called_once_with("users", "user1")
    mock_firebase_app.put.assert_called_once_with("users", "user1", expected_data)
    assert result is True

@pytest.mark.asyncio
async def test_update_nonexistent_user(firebase_repo, mock_firebase_app):
    mock_firebase_app.get.return_value = None
    
    with pytest.raises(QueryError) as exc_info:
        await firebase_repo.update("users", "nonexistent", {"name": "test"})
    
    assert "User not found: nonexistent" in str(exc_info.value)

@pytest.mark.asyncio
async def test_delete(firebase_repo, mock_firebase_app):
    result = await firebase_repo.delete("users", "user1")
    
    mock_firebase_app.delete.assert_called_once_with("users/user1", None)
    assert result is True

@pytest.mark.asyncio
async def test_query_with_key_lookup(firebase_repo, mock_firebase_app):
    mock_firebase_app.get.return_value = {"test": "data"}
    
    result = await firebase_repo.query(
        path="users",
        start_at="user1"
    )
    
    mock_firebase_app.get.assert_called_once_with("users/user1", None)
    assert result == [{"test": "data"}]

@pytest.mark.asyncio
async def test_query_with_filters(firebase_repo, mock_firebase_app):
    mock_firebase_app.get.return_value = [{"test": "data"}]
    
    result = await firebase_repo.query(
        path="users",
        order_by="age",
        limit=10,
        start_at=20,
        end_at=30
    )
    
    mock_firebase_app.get.assert_called_once_with(
        "users",
        None,
        {
            'orderBy': '"age"',
            'limitToFirst': 10,
            'startAt': '"20"',
            'endAt': '"30"'
        }
    )
    assert result == [{"test": "data"}]

@pytest.mark.asyncio
async def test_error_handling(firebase_repo, mock_firebase_app):
    with patch('asyncio.get_event_loop') as mock_loop:
        future = asyncio.Future()
        future.set_exception(Exception("Test error"))
        mock_loop.return_value.run_in_executor.return_value = future
        
        with pytest.raises(Exception) as exc_info:
            await firebase_repo.get("users", "user1")
        
        assert "Test error" in str(exc_info.value)

def test_connection_per_thread(mock_firebase):
    repo = FirebaseRestRepository(
        dsn="https://test-db.firebaseio.com",
        auth_token="test-token"
    )
    
    # Get connection from two different threads
    conn1 = repo._get_connection()
    conn2 = repo._get_connection()
    
    # In the same thread, should return the same connection
    assert conn1 is conn2
    
    # Verify Firebase was initialized with correct parameters
    mock_firebase.assert_called_once_with(
        dsn="https://test-db.firebaseio.com",
        authentication="test-token"
    )

def test_connection_with_no_auth(mock_firebase):
    repo = FirebaseRestRepository(
        dsn="https://test-db.firebaseio.com",
        auth_token=None
    )
    
    repo._get_connection()
    
    # Verify Firebase was initialized with no authentication
    mock_firebase.assert_called_once_with(
        dsn="https://test-db.firebaseio.com",
        authentication=None
    )

def test_cleanup(firebase_repo):
    """Test that resources are properly cleaned up when object is destroyed"""
    connections_before = len(firebase_repo._FirebaseRestRepository__connections)
    
    # Manually call __del__ to simulate object destruction
    firebase_repo.__del__()
    
    # Verify that connections were cleared
    assert len(firebase_repo._FirebaseRestRepository__connections) == 0