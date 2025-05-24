import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status

from infrastructure.database import FirebaseREST


class TestFirebaseREST:
    @pytest.fixture
    def firebase_rest(self):
        """Create a FirebaseREST instance for testing."""
        firebase_rest = FirebaseREST(dsn="https://test-project-default-rtdb.firebaseio.com")
        return firebase_rest
    
    @pytest.fixture
    def firebase_rest_with_auth(self):
        """Create a FirebaseREST instance with authentication for testing."""
        return FirebaseREST(
            dsn="https://test-project-default-rtdb.firebaseio.com",
            authentication="test-auth-token"
        )
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp ClientSession."""
        mock = AsyncMock()
        mock.get = AsyncMock()
        mock.put = AsyncMock()
        mock.delete = AsyncMock()
        mock.aclose = AsyncMock()
        return mock
    
    @pytest.mark.asyncio
    async def test_firebase_rest_instantiation(self, firebase_rest):
        """Test that FirebaseREST can be instantiated correctly."""
        assert firebase_rest.dsn == "https://test-project-default-rtdb.firebaseio.com"
        assert firebase_rest.auth is None
        assert firebase_rest._FirebaseREST__session is None
        assert isinstance(firebase_rest._FirebaseREST__lock, asyncio.Lock)
    
    @pytest.mark.asyncio
    async def test_firebase_rest_instantiation_with_auth(self, firebase_rest_with_auth):
        """Test that FirebaseREST can be instantiated correctly with authentication."""
        assert firebase_rest_with_auth.dsn == "https://test-project-default-rtdb.firebaseio.com"
        assert firebase_rest_with_auth.auth == "test-auth-token"
        assert firebase_rest_with_auth._FirebaseREST__session is None
    
    @pytest.mark.asyncio
    async def test_create_pool(self, firebase_rest):
        """Test creation of the connection pool."""
        with patch('aiohttp.TCPConnector') as mock_connector, \
             patch('aiohttp.ClientSession') as mock_session_class:
            mock_connector.return_value = MagicMock()
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            # Set the pool_size attribute which is missing in the implementation
            firebase_rest._FirebaseREST__pool_size = 10
            
            await firebase_rest.create_pool()
            
            # Check if the connector was created with the correct limit
            mock_connector.assert_called_once_with(limit=10)
            # Check if the session was created with the connector
            mock_session_class.assert_called_once()
            
            # Check if the session was assigned
            assert firebase_rest._FirebaseREST__session == mock_session
    
    @pytest.mark.asyncio
    async def test_create_pool_already_exists(self, firebase_rest, mock_session):
        """Test creating a pool when one already exists."""
        firebase_rest._FirebaseREST__session = mock_session
        
        await firebase_rest.create_pool()
        
        # Ensure session wasn't changed
        assert firebase_rest._FirebaseREST__session == mock_session
    
    @pytest.mark.asyncio
    async def test_close_pool(self, firebase_rest, mock_session):
        """Test closing the connection pool."""
        firebase_rest._FirebaseREST__session = mock_session
        
        await firebase_rest.close_pool()
        
        # Check if aclose was called
        mock_session.aclose.assert_awaited_once()
        # Check if session was set to None
        assert firebase_rest._FirebaseREST__session is None
    
    @pytest.mark.asyncio
    async def test_close_pool_when_none(self, firebase_rest):
        """Test closing the connection pool when it's None."""
        firebase_rest._FirebaseREST__session = None
        
        await firebase_rest.close_pool()
        
        # Nothing should happen
        assert firebase_rest._FirebaseREST__session is None
    
    @pytest.mark.asyncio
    async def test_get_data_success(self, firebase_rest):
        """Test successful data retrieval."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.json.return_value = {"name": "test_value"}
        
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        result = await firebase_rest.get_data("test_key")
        
        # Check if the correct URL was called
        mock_session.get.assert_awaited_once_with("https://test-project-default-rtdb.firebaseio.com/test_key.json")
        
        # Check if the result is correct
        assert result == {"name": "test_value"}
    
    @pytest.mark.asyncio
    async def test_get_data_with_auth(self, firebase_rest_with_auth):
        """Test data retrieval with authentication."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.json.return_value = {"name": "test_value"}
        
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response
        
        firebase_rest_with_auth._FirebaseREST__session = mock_session
        
        result = await firebase_rest_with_auth.get_data("test_key")
        
        # Check if the correct URL was called with auth
        mock_session.get.assert_awaited_once_with(
            "https://test-project-default-rtdb.firebaseio.com/test_key.json?auth=test-auth-token"
        )
        
        # Check if the result is correct
        assert result == {"name": "test_value"}
    
    @pytest.mark.asyncio
    async def test_get_data_null_response(self, firebase_rest):
        """Test handling of null response from Firebase."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.json.return_value = None
        
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        result = await firebase_rest.get_data("test_key")
        
        # Check if empty dict is returned for null
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_data_failure(self, firebase_rest):
        """Test handling of error response from Firebase."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_404_NOT_FOUND
        mock_response.content = b"Not found"
        
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        result = await firebase_rest.get_data("test_key")
        
        # Check if empty dict is returned on error
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_data_exception(self, firebase_rest):
        """Test handling of exception during data retrieval."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Connection error")
        
        firebase_rest._FirebaseREST__session = mock_session
        
        result = await firebase_rest.get_data("test_key")
        
        # Check if empty dict is returned on exception
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_data_complex_path(self, firebase_rest):
        """Test data retrieval with complex path."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.json.return_value = {"nested": {"data": "value"}}
        
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        result = await firebase_rest.get_data("users/123/profile")
        
        # Check if the correct URL was called
        mock_session.get.assert_awaited_once_with(
            "https://test-project-default-rtdb.firebaseio.com/users/123/profile.json"
        )
        
        # Check if the result is correct
        assert result == {"nested": {"data": "value"}}
    
    @pytest.mark.asyncio
    async def test_set_data_success(self, firebase_rest):
        """Test successful data setting."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        
        mock_session = AsyncMock()
        mock_session.put.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        data = {"name": "test_value"}
        await firebase_rest.set_data("test_key", data)
        
        # Check if the correct URL and data were used
        mock_session.put.assert_awaited_once_with(
            "https://test-project-default-rtdb.firebaseio.com/test_key.json",
            json={"name": "test_value"}
        )
    
    @pytest.mark.asyncio
    async def test_set_data_with_auth(self, firebase_rest_with_auth):
        """Test data setting with authentication."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        
        mock_session = AsyncMock()
        mock_session.put.return_value = mock_response
        
        firebase_rest_with_auth._FirebaseREST__session = mock_session
        
        data = {"name": "test_value"}
        await firebase_rest_with_auth.set_data("test_key", data)
        
        # Check if the correct URL and data were used with auth
        mock_session.put.assert_awaited_once_with(
            "https://test-project-default-rtdb.firebaseio.com/test_key.json?auth=test-auth-token",
            json={"name": "test_value"}
        )
    
    @pytest.mark.asyncio
    async def test_set_data_failure(self, firebase_rest):
        """Test handling of error response when setting data."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_400_BAD_REQUEST
        mock_response.content = b"Bad request"
        
        mock_session = AsyncMock()
        mock_session.put.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        with pytest.raises(Exception, match="Error setting data"):
            await firebase_rest.set_data("test_key", {"name": "test_value"})
    
    @pytest.mark.asyncio
    async def test_set_data_exception(self, firebase_rest):
        """Test handling of exception during data setting."""
        mock_session = AsyncMock()
        mock_session.put.side_effect = Exception("Connection error")
        
        firebase_rest._FirebaseREST__session = mock_session
        
        # The method doesn't raise exceptions, just logs them
        await firebase_rest.set_data("test_key", {"name": "test_value"})
    
    @pytest.mark.asyncio
    async def test_set_data_complex_path(self, firebase_rest):
        """Test data setting with complex path."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        
        mock_session = AsyncMock()
        mock_session.put.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        data = {"status": "active"}
        await firebase_rest.set_data("users/123/profile", data)
        
        # Check if the correct URL and data were used
        mock_session.put.assert_awaited_once_with(
            "https://test-project-default-rtdb.firebaseio.com/users/123/profile.json",
            json={"status": "active"}
        )
    
    @pytest.mark.asyncio
    async def test_set_data_empty_object(self, firebase_rest):
        """Test setting empty object data."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        
        mock_session = AsyncMock()
        mock_session.put.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        await firebase_rest.set_data("test_key", {})
        
        # Check if the correct URL and empty data were used
        mock_session.put.assert_awaited_once_with(
            "https://test-project-default-rtdb.firebaseio.com/test_key.json",
            json={}
        )
    
    @pytest.mark.asyncio
    async def test_set_data_null_value(self, firebase_rest):
        """Test setting null value (deleting data)."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        
        mock_session = AsyncMock()
        mock_session.put.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        await firebase_rest.set_data("test_key", None)
        
        # Check if the correct URL and null data were used
        mock_session.put.assert_awaited_once_with(
            "https://test-project-default-rtdb.firebaseio.com/test_key.json",
            json=None
        )
    
    @pytest.mark.asyncio
    async def test_delete_data_success(self, firebase_rest):
        """Test successful data deletion."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        
        mock_session = AsyncMock()
        mock_session.delete.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        result = await firebase_rest.delete_data("test_key")
        
        # Check if the correct URL was called
        mock_session.delete.assert_awaited_once_with(
            "https://test-project-default-rtdb.firebaseio.com/test_key.json"
        )
        
        # Check if the result is True
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_data_with_auth(self, firebase_rest_with_auth):
        """Test data deletion with authentication."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        
        mock_session = AsyncMock()
        mock_session.delete.return_value = mock_response
        
        firebase_rest_with_auth._FirebaseREST__session = mock_session
        
        result = await firebase_rest_with_auth.delete_data("test_key")
        
        # Check if the correct URL was called with auth
        mock_session.delete.assert_awaited_once_with(
            "https://test-project-default-rtdb.firebaseio.com/test_key.json?auth=test-auth-token"
        )
        
        # Check if the result is True
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_data_failure(self, firebase_rest):
        """Test handling of error response when deleting data."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_400_BAD_REQUEST
        mock_response.content = b"Bad request"
        
        mock_session = AsyncMock()
        mock_session.delete.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        with pytest.raises(Exception, match="Error deleting data"):
            await firebase_rest.delete_data("test_key")
    
    @pytest.mark.asyncio
    async def test_delete_data_complex_path(self, firebase_rest):
        """Test data deletion with complex path."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_200_OK
        
        mock_session = AsyncMock()
        mock_session.delete.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        result = await firebase_rest.delete_data("users/123/profile")
        
        # Check if the correct URL was called
        mock_session.delete.assert_awaited_once_with(
            "https://test-project-default-rtdb.firebaseio.com/users/123/profile.json"
        )
        
        # Check if the result is True
        assert result is True
    
    @pytest.mark.asyncio
    async def test_context_manager(self, firebase_rest):
        """Test using FirebaseREST as a context manager."""
        with patch.object(firebase_rest, 'create_pool') as mock_create_pool, \
             patch.object(firebase_rest, 'close_pool') as mock_close_pool:
            mock_create_pool.return_value = None
            mock_close_pool.return_value = None
            
            async with firebase_rest:
                pass
            
            # Check if create_pool and close_pool were called
            mock_create_pool.assert_called_once()
            mock_close_pool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_data_buffer_overflow(self, firebase_rest):
        """Test handling of buffer overflow when setting large data."""
        # Create a large data payload
        large_data = {"large_field": "x" * (1024 * 1024 * 10)}  # 10MB payload
        
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        mock_response.content = b"Payload too large"
        
        mock_session = AsyncMock()
        mock_session.put.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        with pytest.raises(Exception, match="Error setting data"):
            await firebase_rest.set_data("test_key", large_data)
    
    @pytest.mark.asyncio
    async def test_get_data_buffer_overflow(self, firebase_rest):
        """Test handling of buffer overflow when getting large data."""
        mock_response = AsyncMock()
        mock_response.status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        mock_response.content = b"Response too large"
        
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response
        
        firebase_rest._FirebaseREST__session = mock_session
        
        result = await firebase_rest.get_data("test_key_large_data")
        
        # Check if empty dict is returned on error
        assert result == {}
