import pytest
from unittest.mock import Mock, call, patch
from infrastructure.database.firebase.admin_repository import FirebaseAdminRepository
from core.exceptions import QueryError

@pytest.fixture
def mock_db_reference():
    mock_ref = Mock()
    mock_ref.get.return_value = {"test": "data"}
    mock_ref.push.return_value = Mock(key="test-key")
    mock_ref.set.return_value = None
    mock_ref.update.return_value = None
    mock_ref.delete.return_value = None
    mock_ref.child.return_value = mock_ref
    mock_ref.order_by_child.return_value = mock_ref
    mock_ref.limit_to_first.return_value = mock_ref
    mock_ref.start_at.return_value = mock_ref
    mock_ref.end_at.return_value = mock_ref
    return mock_ref

@pytest.fixture
def firebase_repo(mock_db_reference):
    with patch('firebase_admin.credentials.Certificate'), \
         patch('firebase_admin.initialize_app'), \
         patch('firebase_admin.db.reference', return_value=mock_db_reference):
        
        FirebaseAdminRepository._instance = None
        
        repo = FirebaseAdminRepository(
            credentials_path="fake/path",
            options={"databaseURL": "https://fake-db.firebaseio.com"}
        )
        yield repo
        
        repo.close()

@pytest.mark.asyncio
async def test_set_with_key(firebase_repo, mock_db_reference):
    result = await firebase_repo.set("test/path", {"data": "value"}, "test-key")
    
    mock_db_reference.child.assert_has_calls([
        call("test/path"),
        call("test-key")
    ])
    mock_db_reference.child().set.assert_called_with({"data": "value"})
    assert result == "test-key"

@pytest.mark.asyncio
async def test_set_without_key(firebase_repo, mock_db_reference):
    result = await firebase_repo.set("test/path", {"data": "value"})
    
    mock_db_reference.child.assert_called_with("test/path")
    mock_db_reference.child().push.assert_called_with({"data": "value"})
    assert result == "test-key"

@pytest.mark.asyncio
async def test_push(firebase_repo, mock_db_reference):
    result = await firebase_repo.push("test/path", {"data": "value"})
    
    mock_db_reference.child.assert_called_with("test/path")
    mock_db_reference.child().push.assert_called_with({"data": "value"})
    assert result == "test-key"

@pytest.mark.asyncio
async def test_get(firebase_repo, mock_db_reference):
    result = await firebase_repo.get("test/path", "test-key")
    
    mock_db_reference.child.assert_has_calls([
        call("test/path"),
        call("test-key")
    ])
    mock_db_reference.child().child().get.assert_called_once()
    assert result == {"test": "data"}

@pytest.mark.asyncio
async def test_update(firebase_repo, mock_db_reference):
    result = await firebase_repo.update("test/path", "test-key", {"data": "new-value"})
    
    mock_db_reference.child.assert_has_calls([
        call("test/path"),
        call("test-key")
    ])
    mock_db_reference.child().child().update.assert_called_with({"data": "new-value"})
    assert result is True

@pytest.mark.asyncio
async def test_delete(firebase_repo, mock_db_reference):
    result = await firebase_repo.delete("test/path", "test-key")
    
    mock_db_reference.child.assert_has_calls([
        call("test/path"),
        call("test-key")
    ])
    mock_db_reference.child().child().delete.assert_called_once()
    assert result is True

@pytest.mark.asyncio
async def test_query(firebase_repo, mock_db_reference):
    result = await firebase_repo.query(
        path="test/path",
        order_by="timestamp",
        limit=10,
        start_at=0,
        end_at=100
    )
    
    mock_db_reference.child.assert_called_with("test/path")
    mock_db_reference.order_by_child.assert_called_with("timestamp")
    mock_db_reference.limit_to_first.assert_called_with(10)
    mock_db_reference.start_at.assert_called_with(0)
    mock_db_reference.end_at.assert_called_with(100)
    assert result == {"test": "data"}

def test_singleton_instance():
    with patch('firebase_admin.credentials.Certificate') as mock_cert, \
         patch('firebase_admin.initialize_app') as mock_init, \
         patch('firebase_admin.db.reference'):
        
        FirebaseAdminRepository._instance = None
        
        repo1 = FirebaseAdminRepository(
            credentials_path="fake/path",
            options={"databaseURL": "https://fake-db.firebaseio.com"}
        )
        
        repo2 = FirebaseAdminRepository(
            credentials_path="fake/path",
            options={"databaseURL": "https://fake-db.firebaseio.com"}
        )
        
        assert repo1 is repo2
        
        repo1.close()

@pytest.mark.asyncio
async def test_error_handling(firebase_repo, mock_db_reference):
    mock_db_reference.child().get.side_effect = Exception("Test error")
    
    with pytest.raises(QueryError) as exc_info:
        await firebase_repo.get("test/path", "test-key")
    
    assert "Failed to get data: Test error" in str(exc_info.value)