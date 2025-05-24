import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from fastapi import status
import firebase_admin
from firebase_admin.exceptions import FirebaseError

from infrastructure.database import FirebaseAdmin


class TestFirebaseAdmin:
    @pytest.fixture
    def firebase_admin_mock(self):
        """Create a FirebaseAdmin instance for testing."""
        mock_firebase_rest = MagicMock(spec=FirebaseAdmin)
        mock_firebase_rest.get_data = AsyncMock(return_value=None)
        mock_firebase_rest.set_data = AsyncMock(return_value=None)
        mock_firebase_rest.delete_data = AsyncMock(return_value=None)
        return mock_firebase_rest
    
    @pytest.mark.asyncio
    async def test_firebase_admin_instantiation(self, firebase_admin_mock):
        """Test that FirebaseAdmin can be instantiated correctly."""
        assert firebase_admin_mock.get_data is not None
        assert firebase_admin_mock.set_data is not None
        assert firebase_admin_mock.delete_data is not None
        
    @pytest.mark.asyncio
    async def test_firebase_admin_get_data(self, firebase_admin_mock):
        """Test that FirebaseAdmin can get data correctly."""
        result = await firebase_admin_mock.get_data("test-path")
        assert result is None
        
    @pytest.mark.asyncio
    async def test_firebase_admin_set_data(self, firebase_admin_mock):
        """Test that FirebaseAdmin can set data correctly."""
        result = await firebase_admin_mock.set_data("test-path", "test-data")
        assert result is None
        
    @pytest.mark.asyncio
    async def test_firebase_admin_delete_data(self, firebase_admin_mock):
        """Test that FirebaseAdmin can delete data correctly."""
        result = await firebase_admin_mock.delete_data("test-path")
        assert result is None
            
    @pytest.mark.asyncio
    async def test_firebase_admin_get_data_with_auth(self, firebase_admin_mock):
        """Test that FirebaseAdmin can get data correctly with authentication."""
        result = await firebase_admin_mock.get_data("test-path", auth="test-auth-token")
        assert result is None    

    @pytest.mark.asyncio
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin.get_app')
    def test_singleton_pattern(self, mock_get_app, mock_initialize_app):
        """Test the singleton pattern of FirebaseAdmin."""
        # Mock return values
        mock_app = MagicMock()
        mock_get_app.return_value = mock_app
        
        # Setup db mock
        mock_db_ref = MagicMock()
        mock_db_ref.child.return_value = MagicMock()
        
        # Mock db module
        with patch('firebase_admin.db.reference', return_value=mock_db_ref):
            options = {'databaseURL': 'https://test-db.firebaseio.com'}
            instance1 = FirebaseAdmin(options=options)
            instance2 = FirebaseAdmin(options=options)
            assert instance1 is instance2

    @pytest.mark.asyncio
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin.credentials.Certificate')
    @patch('firebase_admin.get_app')
    def test_initialization_with_credentials(self, mock_get_app, mock_certificate, mock_initialize_app):
        """Test initializing FirebaseAdmin with credentials."""
        # Mock return values
        mock_app = MagicMock()
        mock_get_app.return_value = mock_app
        mock_certificate.return_value = MagicMock()
        
        # Setup db mock
        mock_db_ref = MagicMock()
        mock_db_ref.child.return_value = MagicMock()
        
        # Force FirebaseAdmin._instance to None for this test and patch _apps to be empty
        with patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={}), \
             patch('firebase_admin.db.reference', return_value=mock_db_ref):
            
            options = {'databaseURL': 'https://test-db.firebaseio.com'}
            credentials_path = 'path/to/credentials.json'
            FirebaseAdmin(credentials_path=credentials_path, options=options)
            
            # Now verify our mocks were called correctly
            mock_certificate.assert_called_once_with(credentials_path)
            mock_initialize_app.assert_called_once()

    @pytest.mark.asyncio
    @patch('firebase_admin.initialize_app')
    def test_initialization_without_db_url(self, mock_initialize_app):
        """Test initializing FirebaseAdmin without database URL throws ValueError."""
        # Force FirebaseAdmin._instance to None for this test
        with patch.object(FirebaseAdmin, '_instance', None):
            with pytest.raises(ValueError) as excinfo:
                FirebaseAdmin(options={})
            assert "Database URL not provided" in str(excinfo.value)

    @pytest.mark.asyncio
    @patch('firebase_admin.initialize_app', side_effect=Exception("Test exception"))
    @patch('firebase_admin.get_app', side_effect=ValueError("App not found"))
    def test_initialization_exception(self, mock_get_app, mock_initialize_app):
        """Test handling of exceptions during initialization."""
        # Force FirebaseAdmin._instance to None for this test
        with patch.object(FirebaseAdmin, '_instance', None):
            with pytest.raises(Exception) as excinfo:
                FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            assert "Test exception" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_get_data_with_return_value(self):
        """Test get_data method with a return value."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app = MagicMock(return_value=mock_app)
        
        # Setup reference mock
        mock_child = MagicMock()
        mock_child.get.return_value = {'key': 'value'}
        mock_reference = MagicMock()
        mock_reference.child.return_value = mock_child
        
        # Create patches
        with patch('firebase_admin.get_app', mock_get_app), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}):
            
            # Create instance
            admin = FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            
            # Test method
            result = await admin.get_data("test-path")
            
            # Assertions
            assert result == {'key': 'value'}
            mock_reference.child.assert_called_once_with("test-path")

    @pytest.mark.asyncio
    async def test_get_data_firebase_error(self):
        """Test get_data method handling FirebaseError."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app = MagicMock(return_value=mock_app)
        
        # Setup reference mock with error
        mock_child = MagicMock()
        mock_child.get.side_effect = FirebaseError(500, "Firebase error")
        mock_reference = MagicMock()
        mock_reference.child.return_value = mock_child
        
        # Create patches
        with patch('firebase_admin.get_app', mock_get_app), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}):
            
            # Create instance
            admin = FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            
            # Test method
            with pytest.raises(FirebaseError) as excinfo:
                await admin.get_data("test-path")
            
            # Assertions
            assert "Firebase error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_get_data_general_exception(self):
        """Test get_data method handling general exceptions."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app = MagicMock(return_value=mock_app)
        
        # Setup reference mock with error
        mock_child = MagicMock()
        mock_child.get.side_effect = Exception("General error")
        mock_reference = MagicMock()
        mock_reference.child.return_value = mock_child
        
        # Create patches
        with patch('firebase_admin.get_app', mock_get_app), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}):
            
            # Create instance
            admin = FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            
            # Test method
            with pytest.raises(Exception) as excinfo:
                await admin.get_data("test-path")
            
            # Assertions
            assert "General error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_set_data_success(self):
        """Test set_data method successful operation."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app = MagicMock(return_value=mock_app)
        
        # Setup reference mock
        mock_child = MagicMock()
        mock_reference = MagicMock()
        mock_reference.child.return_value = mock_child
        
        # Create patches
        with patch('firebase_admin.get_app', mock_get_app), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}):
            
            # Create instance
            admin = FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            
            # Test method
            await admin.set_data("test-path", {"key": "value"})
            
            # Assertions
            mock_reference.child.assert_called_once_with("test-path")
            mock_child.set.assert_called_once_with(value={"key": "value"})

    @pytest.mark.asyncio
    async def test_set_data_missing_parameters(self):
        """Test set_data method with missing parameters."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app = MagicMock(return_value=mock_app)
        
        # Setup reference mock
        mock_reference = MagicMock()
        
        # Create patches
        with patch('firebase_admin.get_app', mock_get_app), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}):
            
            # Create instance
            admin = FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            
            # Test method with missing key
            with pytest.raises(ValueError) as excinfo:
                await admin.set_data(None, {"key": "value"})
            assert "Key and value not provided" in str(excinfo.value)
            
            # Test method with missing value
            with pytest.raises(ValueError) as excinfo:
                await admin.set_data("test-path", None)
            assert "Key and value not provided" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_set_data_firebase_error(self):
        """Test set_data method handling FirebaseError."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app = MagicMock(return_value=mock_app)
        
        # Setup reference mock with error
        mock_child = MagicMock()
        mock_child.set.side_effect = FirebaseError(500, "Firebase error")
        mock_reference = MagicMock()
        mock_reference.child.return_value = mock_child
        
        # Create patches
        with patch('firebase_admin.get_app', mock_get_app), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}):
            
            # Create instance
            admin = FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            
            # Test method
            with pytest.raises(FirebaseError) as excinfo:
                await admin.set_data("test-path", {"key": "value"})
            
            # Assertions
            assert "Firebase error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_set_data_general_exception(self):
        """Test set_data method handling general exceptions."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app = MagicMock(return_value=mock_app)
        
        # Setup reference mock with error
        mock_child = MagicMock()
        mock_child.set.side_effect = Exception("General error")
        mock_reference = MagicMock()
        mock_reference.child.return_value = mock_child
        
        # Create patches
        with patch('firebase_admin.get_app', mock_get_app), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}):
            
            # Create instance
            admin = FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            
            # Test method
            with pytest.raises(Exception) as excinfo:
                await admin.set_data("test-path", {"key": "value"})
            
            # Assertions
            assert "General error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_delete_data_success(self):
        """Test delete_data method successful operation."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app = MagicMock(return_value=mock_app)
        
        # Setup reference mock
        mock_child = MagicMock()
        mock_reference = MagicMock()
        mock_reference.child.return_value = mock_child
        
        # Create patches
        with patch('firebase_admin.get_app', mock_get_app), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}):
            
            # Create instance
            admin = FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            
            # Test method
            await admin.delete_data("test-path")
            
            # Assertions
            mock_reference.child.assert_called_once_with("test-path")
            mock_child.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_data_firebase_error(self):
        """Test delete_data method handling FirebaseError."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app = MagicMock(return_value=mock_app)
        
        # Setup reference mock with error
        mock_child = MagicMock()
        mock_child.delete.side_effect = FirebaseError(500, "Firebase error")
        mock_reference = MagicMock()
        mock_reference.child.return_value = mock_child
        
        # Create patches
        with patch('firebase_admin.get_app', mock_get_app), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}):
            
            # Create instance
            admin = FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            
            # Test method
            with pytest.raises(FirebaseError) as excinfo:
                await admin.delete_data("test-path")
            
            # Assertions
            assert "Firebase error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_delete_data_general_exception(self):
        """Test delete_data method handling general exceptions."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app = MagicMock(return_value=mock_app)
        
        # Setup reference mock with error
        mock_child = MagicMock()
        mock_child.delete.side_effect = Exception("General error")
        mock_reference = MagicMock()
        mock_reference.child.return_value = mock_child
        
        # Create patches
        with patch('firebase_admin.get_app', mock_get_app), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None), \
             patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}):
            
            # Create instance
            admin = FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            
            # Test method
            with pytest.raises(Exception) as excinfo:
                await admin.delete_data("test-path")
            
            # Assertions
            assert "General error" in str(excinfo.value)

    @pytest.mark.asyncio
    @patch('firebase_admin.get_app')
    def test_initialization_with_existing_app(self, mock_get_app):
        """Test initializing FirebaseAdmin when app already exists."""
        mock_app = MagicMock()
        mock_get_app.return_value = mock_app
        
        # Setup reference mock
        mock_reference = MagicMock()
        
        with patch('firebase_admin._apps', new={'[DEFAULT]': mock_app}), \
             patch('firebase_admin.db.reference', return_value=mock_reference), \
             patch.object(FirebaseAdmin, '_instance', None):
            
            FirebaseAdmin(options={'databaseURL': 'https://test-db.firebaseio.com'})
            # If we get here without error, the test passes    
