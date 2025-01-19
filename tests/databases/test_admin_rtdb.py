import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import firebase_admin
from firebase_admin import credentials, db
from lifttrack.v2.dbhelper.admin_rtdb import FirebaseDBHelper

@pytest.fixture
def mock_credentials():
    """Mock credentials dictionary"""
    return {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "test-key-id",
        "private_key": "test-private-key",
        "client_email": "test@test.com",
        "client_id": "test-client-id",
        "auth_uri": "https://test.com/auth",
        "token_uri": "https://test.com/token",
        "auth_provider_x509_cert_url": "https://test.com/cert",
        "client_x509_cert_url": "https://test.com/x509"
    }

@pytest.fixture
def mock_firebase_admin():
    with patch('firebase_admin.credentials') as mock_cred, \
         patch('firebase_admin.initialize_app') as mock_init, \
         patch('firebase_admin.db') as mock_db, \
         patch('firebase_admin._apps', new={'[DEFAULT]': MagicMock()}):
        
        final_mock = MagicMock()
        final_mock.get.return_value = {"test": "data"}
        final_mock.set = MagicMock()
        final_mock.update = MagicMock()
        final_mock.delete = MagicMock()
        final_mock.push.return_value.key = "test_key"
        
        intermediate_mock = MagicMock()
        intermediate_mock.child.return_value = final_mock
        intermediate_mock.get.return_value = {"test": "data"}
        intermediate_mock.push.return_value.key = "test_key"
        intermediate_mock.order_by_child.return_value = intermediate_mock
        intermediate_mock.start_at.return_value = intermediate_mock
        intermediate_mock.end_at.return_value = intermediate_mock
        intermediate_mock.limit_to_first.return_value = intermediate_mock
        
        root_mock = MagicMock()
        root_mock.child.return_value = intermediate_mock
        root_mock.get.return_value = {"test": "data"}
        root_mock.push.return_value.key = "test_key"
        root_mock.order_by_child.return_value = root_mock
        root_mock.start_at.return_value = root_mock
        root_mock.end_at.return_value = root_mock
        root_mock.limit_to_first.return_value = root_mock
        
        mock_db.reference.return_value = root_mock
        
        yield {
            'credentials': mock_cred,
            'init_app': mock_init,
            'db': mock_db,
            'ref': root_mock,
            'intermediate': intermediate_mock,
            'final': final_mock
        }

@pytest.fixture
def firebase_db(mock_firebase_admin, mock_credentials):
    """Create a FirebaseDBHelper instance with mocked dependencies"""
    options = {
        'databaseURL': 'https://test-db.firebaseio.com'
    }
    
    with patch('firebase_admin.credentials.Certificate', return_value=mock_credentials), \
         patch('firebase_admin.db.reference', return_value=mock_firebase_admin['ref']):
        instance = FirebaseDBHelper(
            credentials_path=mock_credentials,
            options=options
        )
        instance._db = mock_firebase_admin['ref']
        return instance

def test_get_reference(firebase_db, mock_firebase_admin):
    ref = firebase_db.get_reference("test/path")
    
    mock_firebase_admin['ref'].child.assert_called_once_with("test/path")
    assert ref == mock_firebase_admin['intermediate']

def test_set_data_with_key(firebase_db, mock_firebase_admin):
    test_data = {"name": "test"}
    test_key = "test_key"
    
    result = firebase_db.set_data("test/path", test_data, test_key)
    
    mock_firebase_admin['ref'].child.assert_called_with("test/path")
    mock_firebase_admin['intermediate'].child.assert_called_with(test_key)
    mock_firebase_admin['final'].set.assert_called_with(test_data)
    assert result == test_key

def test_get_data(firebase_db, mock_firebase_admin):
    result = firebase_db.get_data("test/path", "test_key")
    
    mock_firebase_admin['ref'].child.assert_called_with("test/path")
    mock_firebase_admin['intermediate'].child.assert_called_with("test_key")
    mock_firebase_admin['final'].get.assert_called_once()
    assert result == {"test": "data"}

def test_update_data(firebase_db, mock_firebase_admin):
    update_data = {"name": "updated"}
    
    result = firebase_db.update_data("test/path", "test_key", update_data)
    
    mock_firebase_admin['ref'].child.assert_called_with("test/path")
    mock_firebase_admin['intermediate'].child.assert_called_with("test_key")
    mock_firebase_admin['final'].update.assert_called_with(update_data)
    assert result is True

def test_delete_data(firebase_db, mock_firebase_admin):
    result = firebase_db.delete_data("test/path", "test_key")
    
    mock_firebase_admin['ref'].child.assert_called_with("test/path")
    mock_firebase_admin['intermediate'].child.assert_called_with("test_key")
    mock_firebase_admin['final'].delete.assert_called_once()
    assert result is True

def test_query_data(firebase_db, mock_firebase_admin):
    mock_firebase_admin['ref'].get.return_value = [{"test": "data"}]
    
    result = firebase_db.query_data(
        path="test/path",
        order_by="name",
        limit=10,
        start_at="A",
        end_at="Z"
    )
    
    mock_firebase_admin['ref'].child.assert_called_with("test/path")
    mock_firebase_admin['intermediate'].order_by_child.assert_called_with("name")
    mock_firebase_admin['intermediate'].start_at.assert_called_with("A")
    mock_firebase_admin['intermediate'].end_at.assert_called_with("Z")
    mock_firebase_admin['intermediate'].limit_to_first.assert_called_with(10)
    assert result == {"test": "data"}

def test_singleton_pattern(mock_firebase_admin, mock_credentials):
    options = {'databaseURL': 'https://test-db.firebaseio.com'}
    
    with patch('firebase_admin.credentials.Certificate', return_value=mock_credentials):
        db1 = FirebaseDBHelper(
            credentials_path=mock_credentials,
            options=options
        )
        db2 = FirebaseDBHelper(
            credentials_path=mock_credentials,
            options=options
        )
    
    assert db1 is db2

def test_close(firebase_db):
    mock_app = MagicMock(spec=firebase_admin.App)  # Create mock with App spec
    
    with patch('firebase_admin.get_app', return_value=mock_app), \
         patch('firebase_admin.delete_app') as mock_delete_app:
        firebase_db.close()
    
        assert firebase_db._executor._shutdown
        mock_delete_app.assert_called_once_with(mock_app)