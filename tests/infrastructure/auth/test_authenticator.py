import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from jose import jwt

from infrastructure.auth import Authenticator
from infrastructure.auth.token_blacklist_repository import (
    InMemoryTokenBlacklistRepository,
)
from core.entities import (
    UserEntity,
    TokenEntity,
    TokenDataEntity,
    CredentialsEntity,
    ValidationResultEntity,
)


class TestAuthenticator:
    @pytest.fixture
    def user_repository(self):
        """Create a mock user repository for testing."""
        mock = AsyncMock()
        mock.get_user_by_username = AsyncMock()
        mock.get_user_by_id = AsyncMock()
        mock.update_password = AsyncMock()
        mock.check_username_exists = AsyncMock()
        mock.get_all_users = AsyncMock()
        return mock

    @pytest.fixture
    def token_blacklist(self):
        """Create a token blacklist repository for testing."""
        return InMemoryTokenBlacklistRepository()

    @pytest.fixture
    def authenticator(self, user_repository, token_blacklist):
        """Create an Authenticator instance for testing."""
        return Authenticator(
            secret_key="test_secret_key",
            algorithm="HS256",
            access_token_expire_minutes=30,
            user_repository=user_repository,
            token_blacklist_repository=token_blacklist,
        )

    @pytest.fixture
    def test_user(self):
        """Create a test user for authentication tests."""
        return UserEntity(
            id="user123",
            username="testuser",
            email="test@example.com",
            password="$2b$12$tVN1BvCVrCgzlqQIBDvxz.TM7mnrJq3qCaXVUUHgPvCIHWTxCf6K2",  # hashed "Test123$"
            first_name="Test",
            last_name="User",
            phoneNum="+639123456789",  # Added the required phoneNum field
        )

    @pytest.mark.asyncio
    async def test_validate_password_strength_valid(self, authenticator):
        """Test password validation with a valid password."""
        valid_password = "Password1$"
        result = await authenticator.validate_password_strength(valid_password)

        assert result.is_valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_password_strength_too_short(self, authenticator):
        """Test password validation with a too short password."""
        invalid_password = "Pass1$"
        result = await authenticator.validate_password_strength(invalid_password)

        assert result.is_valid is False
        assert "Password must be at least 8 characters long" in result.errors

    @pytest.mark.asyncio
    async def test_validate_password_strength_no_uppercase(self, authenticator):
        """Test password validation with no uppercase letter."""
        invalid_password = "password1$"
        result = await authenticator.validate_password_strength(invalid_password)

        assert result.is_valid is False
        assert "Password must contain at least one uppercase letter" in result.errors

    @pytest.mark.asyncio
    async def test_validate_password_strength_no_digit(self, authenticator):
        """Test password validation with no digit."""
        invalid_password = "Password$"
        result = await authenticator.validate_password_strength(invalid_password)

        assert result.is_valid is False
        assert "Password must contain at least one digit" in result.errors

    @pytest.mark.asyncio
    async def test_validate_password_strength_no_special_char(self, authenticator):
        """Test password validation with no special character."""
        invalid_password = "Password1"
        result = await authenticator.validate_password_strength(invalid_password)

        assert result.is_valid is False
        assert (
            "Password must contain at least one special character (@, $)"
            in result.errors
        )

    @pytest.mark.asyncio
    async def test_hash_and_verify_password(self, authenticator):
        """Test password hashing and verification."""
        test_password = "SecurePass1$"

        # Hash the password
        hashed_password = await authenticator.hash_password(test_password)

        # Verify the password
        is_valid = await authenticator.verify_password(test_password, hashed_password)

        assert is_valid is True

        # Verify with wrong password
        is_valid = await authenticator.verify_password("WrongPass1$", hashed_password)

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_authenticate_success(
        self, authenticator, user_repository, test_user
    ):
        """Test successful user authentication."""
        # Setup the mock to return our test user
        user_repository.get_user_by_username.return_value = test_user

        # Set up a valid password mock
        with patch.object(authenticator, "verify_password", return_value=True):
            credentials = CredentialsEntity(username="testuser", password="Test123$")
            user = await authenticator.authenticate(credentials)

            assert user is not None
            assert user.username == "testuser"
            assert user.id == "user123"

            # Verify the repository was called correctly
            user_repository.get_user_by_username.assert_called_once_with("testuser")

    @pytest.mark.asyncio
    async def test_authenticate_wrong_password(
        self, authenticator, user_repository, test_user
    ):
        """Test authentication with wrong password."""
        # Setup the mock to return our test user
        user_repository.get_user_by_username.return_value = test_user

        # Set up password verification to fail
        with patch.object(authenticator, "verify_password", return_value=False):
            credentials = CredentialsEntity(
                username="testuser", password="WrongPassword"
            )
            user = await authenticator.authenticate(credentials)

            assert user is None

            # Verify the repository was called correctly
            user_repository.get_user_by_username.assert_called_once_with("testuser")

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self, authenticator, user_repository):
        """Test authentication with non-existent user."""
        # Setup the mock to return None (user not found)
        user_repository.get_user_by_username.return_value = None

        credentials = CredentialsEntity(username="nonexistent", password="Password1$")
        user = await authenticator.authenticate(credentials)

        assert user is None

        # Verify the repository was called correctly
        user_repository.get_user_by_username.assert_called_once_with("nonexistent")

    @pytest.mark.asyncio
    async def test_create_token(self, authenticator, test_user):
        """Test token creation."""
        # Create a token for the test user
        token_entity = await authenticator.create_token(test_user)

        assert token_entity is not None
        assert token_entity.access_token is not None
        assert token_entity.token_type == "bearer"
        assert token_entity.expires_at is not None

        # Decode token to verify contents
        payload = jwt.decode(
            token_entity.access_token, "test_secret_key", algorithms=["HS256"]
        )

        assert payload["sub"] == test_user.username
        assert "exp" in payload

    @pytest.mark.asyncio
    async def test_create_token_with_custom_expiry(self, authenticator, test_user):
        """Test token creation with custom expiry time."""
        # Create a token with custom expiry
        custom_delta = timedelta(minutes=5)
        token_entity = await authenticator.create_token(test_user, custom_delta)

        assert token_entity is not None

        # Decode token to verify expiry
        payload = jwt.decode(
            token_entity.access_token, "test_secret_key", algorithms=["HS256"]
        )

        token_exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        now = datetime.now(timezone.utc)

        # Check that expiry is approximately 5 minutes in the future (with 2-second tolerance)
        assert abs((token_exp - now).total_seconds() - 300) < 2

    @pytest.mark.asyncio
    async def test_verify_token_valid(self, authenticator, test_user):
        """Test verification of a valid token."""
        # Create a valid token
        token_entity = await authenticator.create_token(test_user)

        # Verify the token
        token_data = await authenticator.verify_token(token_entity.access_token)

        assert token_data is not None
        assert token_data.username == test_user.username
        assert token_data.exp is not None

    @pytest.mark.asyncio
    async def test_verify_token_invalid(self, authenticator):
        """Test verification of an invalid token."""
        # Try to verify an invalid token
        token_data = await authenticator.verify_token("invalid_token")

        assert token_data is None

    @pytest.mark.asyncio
    async def test_verify_token_blacklisted(
        self, authenticator, test_user, token_blacklist
    ):
        """Test verification of a blacklisted token."""
        # Create a valid token
        token_entity = await authenticator.create_token(test_user)

        # Blacklist the token
        await token_blacklist.add_to_blacklist(
            token_entity.access_token, token_entity.expires_at
        )

        # Verify the token (should fail due to blacklisting)
        token_data = await authenticator.verify_token(token_entity.access_token)

        assert token_data is None

    @pytest.mark.asyncio
    async def test_refresh_token(self, authenticator, user_repository, test_user):
        """Test refreshing a token."""
        # Create an original token
        original_token = await authenticator.create_token(test_user)

        # Mock user repository to return test user
        user_repository.get_user_by_username.return_value = test_user

        # Refresh the token
        new_token = await authenticator.refresh_token(original_token.access_token)

        assert new_token is not None
        assert new_token.access_token != original_token.access_token

        # Verify new token is valid
        token_data = await authenticator.verify_token(new_token.access_token)
        assert token_data is not None
        assert token_data.username == test_user.username

    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self, authenticator):
        """Test refreshing an invalid token."""
        # Try to refresh an invalid token
        new_token = await authenticator.refresh_token("invalid_token")

        assert new_token is None

    @pytest.mark.asyncio
    async def test_change_password_success(
        self, authenticator, user_repository, test_user
    ):
        """Test successful password change."""
        # Mock repository responses
        user_repository.get_user_by_id.return_value = test_user
        user_repository.update_password.return_value = True

        # Mock password verification
        with patch.object(authenticator, "verify_password", return_value=True):
            # Change password
            success, error = await authenticator.change_password(
                user_id="user123", old_password="OldPass1$", new_password="NewPass1$"
            )

            assert success is True
            assert error is None

            # Verify repository was called correctly
            user_repository.get_user_by_id.assert_called_once_with("user123")
            user_repository.update_password.assert_called_once()

    @pytest.mark.asyncio
    async def test_change_password_invalid_old_password(
        self, authenticator, user_repository, test_user
    ):
        """Test password change with invalid old password."""
        # Mock repository responses
        user_repository.get_user_by_id.return_value = test_user

        # Mock password verification to fail
        with patch.object(authenticator, "verify_password", return_value=False):
            # Change password
            success, error = await authenticator.change_password(
                user_id="user123", old_password="WrongPass1$", new_password="NewPass1$"
            )

            assert success is False
            assert error == "Current password is incorrect"

            # Verify repository was called correctly but update was not
            user_repository.get_user_by_id.assert_called_once_with("user123")
            user_repository.update_password.assert_not_called()

    @pytest.mark.asyncio
    async def test_change_password_weak_new_password(
        self, authenticator, user_repository, test_user
    ):
        """Test password change with weak new password."""
        # Mock repository responses
        user_repository.get_user_by_id.return_value = test_user

        # Mock password verification to succeed
        with patch.object(authenticator, "verify_password", return_value=True):
            # Change password
            success, error = await authenticator.change_password(
                user_id="user123",
                old_password="OldPass1$",
                new_password="weak",  # Weak password that will fail validation
            )

            assert success is False
            assert "Password must be at least 8 characters long" in error

            # Verify repository was called correctly but update was not
            user_repository.get_user_by_id.assert_called_once_with("user123")
            user_repository.update_password.assert_not_called()

    @pytest.mark.asyncio
    async def test_logout(self, authenticator, test_user, token_blacklist):
        """Test successful logout (token blacklisting)."""
        # Create a token to blacklist
        token_entity = await authenticator.create_token(test_user)

        # Logout (blacklist the token)
        success = await authenticator.logout(token_entity.access_token)

        assert success is True

        # Verify token is blacklisted
        is_blacklisted = await token_blacklist.is_blacklisted(token_entity.access_token)
        assert is_blacklisted is True

    @pytest.mark.asyncio
    async def test_is_username_taken_cache_hit(self, authenticator):
        """Test username availability check with cache hit."""
        # Add username to cache
        authenticator.username_cache.add("existinguser")

        # Check if username is taken
        result = await authenticator.is_username_taken("existinguser")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_username_taken_database_check(
        self, authenticator, user_repository
    ):
        """Test username availability check with database lookup."""
        # Set up mock to return True (username exists)
        user_repository.check_username_exists.return_value = True

        # Check if username is taken (not in cache)
        result = await authenticator.is_username_taken("newuser")

        assert result is True

        # Verify repository was called
        user_repository.check_username_exists.assert_called_once_with("newuser")

        # Verify username was added to cache
        assert "newuser" in authenticator.username_cache

    @pytest.mark.asyncio
    async def test_validate_user_input_valid(self, authenticator):
        """Test user input validation with valid data."""
        # Mock is_username_taken to return False
        with patch.object(authenticator, "is_username_taken", return_value=False):
            user_data = {
                "username": "newuser",
                "password": "Password1$",
                "email": "user@example.com",
                "phoneNum": "+639123456789",
            }

            result = await authenticator.validate_user_input(user_data)

            assert result.is_valid is True
            assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_user_input_invalid_email(self, authenticator):
        """Test user input validation with invalid email."""
        user_data = {
            "username": "newuser",
            "password": "Password1$",
            "email": "invalid-email",
            "phoneNum": "+639123456789",
        }

        result = await authenticator.validate_user_input(user_data)

        assert result.is_valid is False
        assert "Invalid email address format" in result.errors

    @pytest.mark.asyncio
    async def test_initialize_username_cache(self, authenticator, user_repository):
        """Test initialization of the username cache."""
        # Create mock users
        mock_users = [
            UserEntity(
                id="user1",
                username="user1",
                email="user1@example.com",
                password="hash1",
                first_name="User",
                last_name="One",
                phoneNum="+639123456781",
            ),
            UserEntity(
                id="user2",
                username="user2",
                email="user2@example.com",
                password="hash2",
                first_name="User",
                last_name="Two",
                phoneNum="+639123456782",
            ),
        ]

        # Set up mock to return users
        user_repository.get_all_users.return_value = mock_users

        # Initialize cache
        await authenticator.initialize_username_cache()

        # Verify cache contains usernames
        assert "user1" in authenticator.username_cache
        assert "user2" in authenticator.username_cache
        assert len(authenticator.username_cache) == 2
