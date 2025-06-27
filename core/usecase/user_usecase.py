from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timezone
import logging

from core.entities import (
    UserEntity,
    ValidationResultEntity,
    CredentialsEntity,
    TokenEntity,
)
from core.interface import AuthenticationInterface, NTFInterface
from core.service import generate_user_id, validate_email, validate_password


class UserUseCase:
    """
    Use case class for handling user-related business logic.
    Implements operations like user registration, authentication, and profile management.
    """

    def __init__(
        self, auth_service: AuthenticationInterface, database_service: NTFInterface
    ):
        self.auth_service = auth_service
        self.database_service = database_service
        self.logger = logging.getLogger(__name__)

    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
        phone_number: str,
        profile_picture: Optional[str] = None,
    ) -> Tuple[bool, Optional[UserEntity], Optional[str]]:
        """
        Register a new user in the system.

        Args:
            username: User's chosen username
            email: User's email address
            password: User's password
            first_name: User's first name
            last_name: User's last name
            phone_number: User's phone number
            profile_picture: URL to user's profile picture (optional)

        Returns:
            Tuple containing (success, user_entity, error_message)
        """
        try:
            # Validate user input
            user_data = {
                "username": username,
                "email": email,
                "password": password,
                "first_name": first_name,
                "last_name": last_name,
                "phone_number": phone_number,
                "profile_picture": profile_picture,
            }

            validation_result = await self.auth_service.validate_user_input(user_data)
            if not validation_result.is_valid:
                return False, None, "; ".join(validation_result.errors)

            # Check if username is taken
            if await self.auth_service.is_username_taken(username):
                return False, None, "Username already exists"

            # Create user entity
            hashed_password = await self.auth_service.hash_password(password)

            current_time = datetime.now(timezone.utc)

            user = UserEntity(
                id=generate_user_id(),
                username=username,
                email=email,
                password=hashed_password,
                first_name=first_name,
                last_name=last_name,
                phone_number=phone_number,
                profile_picture=profile_picture,
                created_at=current_time,
            )

            # Serialize user data for database storage
            # Convert datetime to ISO format string to make it JSON serializable
            user_dict = user.model_dump(exclude_none=True)
            if "created_at" in user_dict:
                user_dict["created_at"] = user_dict["created_at"].isoformat()
            if "updated_at" in user_dict:
                user_dict["updated_at"] = user_dict["updated_at"].isoformat()
            if "last_login" in user_dict:
                user_dict["last_login"] = user_dict["last_login"].isoformat()

            # Save user to database
            await self.database_service.set_data(f"users/{user.username}", user_dict)

            return True, user, None
        except Exception as e:
            self.logger.error(f"Error registering user: {str(e)}")
            return False, None, f"Registration error: {str(e)}"

    async def authenticate_user(
        self, credentials: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Authenticate a user and generate access tokens.

        Args:
            credentials: Dictionary containing username_or_email and password

        Returns:
            Dictionary with access and refresh tokens if authentication successful, None otherwise
        """
        try:
            # Convert dictionary to credentials entity
            creds = CredentialsEntity(
                username=credentials.get("username_or_email", ""),
                password=credentials.get("password", ""),
            )

            # Authenticate user
            user = await self.auth_service.authenticate(creds)
            if not user:
                return None

            # Update last login time
            user.last_login = datetime.now(timezone.utc)

            # Serialize user data for database storage
            # Convert datetime to ISO format string to make it JSON serializable
            user_dict = user.model_dump(exclude_none=True)
            if "created_at" in user_dict:
                user_dict["created_at"] = user_dict["created_at"].isoformat()
            if "updated_at" in user_dict:
                user_dict["updated_at"] = user_dict["updated_at"].isoformat()
            if "last_login" in user_dict:
                user_dict["last_login"] = user_dict["last_login"].isoformat()

            await self.database_service.set_data(f"users/{user.username}", user_dict)

            # Generate tokens
            token = await self.auth_service.create_token(user)
            refresh_token = await self.auth_service.refresh_token(token.access_token)

            return {
                "access_token": token.access_token,
                "token_type": token.token_type,
                "refresh_token": refresh_token.access_token if refresh_token else None,
            }
        except Exception as e:
            self.logger.error(f"Error authenticating user: {str(e)}")
            return None

    async def get_user_profile(self, user_id: str) -> Optional[UserEntity]:
        """
        Retrieve a user's profile information.

        Args:
            user_id: User's unique identifier

        Returns:
            UserEntity if user exists, None otherwise
        """
        try:
            user_data = await self.database_service.get_data(f"users/{user_id}")
            if not user_data:
                return None

            return UserEntity(**user_data)
        except Exception as e:
            self.logger.error(f"Error getting user profile: {str(e)}")
            return None

    async def update_user_profile(
        self, user_id: str, update_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[UserEntity], Optional[str]]:
        """
        Update a user's profile information.

        Args:
            user_id: User's unique identifier
            update_data: Dictionary containing fields to update

        Returns:
            Tuple containing (success, updated_user, error_message)
        """
        try:
            # Validate update data
            validation_result = await self.auth_service.validate_user_input(
                update_data, is_update=True
            )
            if not validation_result.is_valid:
                return False, None, "; ".join(validation_result.errors)

            # Get existing user
            user_data = await self.database_service.get_data(f"users/{user_id}")
            if not user_data:
                return False, None, "User not found"

            current_user = UserEntity(**user_data)

            # Update fields if provided
            if "first_name" in update_data:
                current_user.first_name = update_data["first_name"]

            if "last_name" in update_data:
                current_user.last_name = update_data["last_name"]

            if "email" in update_data and update_data["email"] != current_user.email:
                # Email is being changed, verify it's not taken
                if await self.auth_service.is_username_taken(update_data["email"]):
                    return False, None, "Email already in use"

                current_user.email = update_data["email"]
                current_user.is_authenticated = (
                    False  # Require re-verification for new email
                )

            current_user.updated_at = datetime.now(timezone.utc)

            # Serialize user data for database storage
            # Convert datetime to ISO format string to make it JSON serializable
            user_dict = current_user.model_dump(exclude_none=True)
            if "created_at" in user_dict:
                user_dict["created_at"] = user_dict["created_at"].isoformat()
            if "updated_at" in user_dict:
                user_dict["updated_at"] = user_dict["updated_at"].isoformat()
            if "last_login" in user_dict:
                user_dict["last_login"] = user_dict["last_login"].isoformat()

            await self.database_service.set_data(
                f"users/{current_user.username}", user_dict
            )
            return True, current_user, None
        except Exception as e:
            self.logger.error(f"Error updating user profile: {str(e)}")
            return False, None, f"Update error: {str(e)}"

    async def change_user_password(
        self, user_id: str, old_password: str, new_password: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Change a user's password.

        Args:
            user_id: User's unique identifier
            old_password: Current password
            new_password: New password

        Returns:
            Tuple containing (success, error_message)
        """
        try:
            # Validate new password strength
            validation = await self.auth_service.validate_password_strength(
                new_password
            )
            if not validation.is_valid:
                return False, "; ".join(validation.errors)

            # Attempt to change password
            result, error = await self.auth_service.change_password(
                user_id, old_password, new_password
            )
            if not result:
                return False, error or "Failed to change password"

            return True, None
        except Exception as e:
            self.logger.error(f"Error changing user password: {str(e)}")
            return False, f"Password change error: {str(e)}"

    async def delete_user(self, user_id: str) -> Tuple[bool, Optional[str]]:
        """
        Mark a user as deleted in the system.

        Args:
            user_id: User's unique identifier

        Returns:
            Tuple containing (success, error_message)
        """
        try:
            user_data = await self.database_service.get_data(f"users/{user_id}")
            if not user_data:
                return False, "User not found"

            user = UserEntity.model_validate(user_data)
            user.is_deleted = True
            user.updated_at = datetime.now(timezone.utc)

            await self.database_service.set_data(
                f"users/{user.id}", user.model_dump(exclude_none=True)
            )
            return True, None
        except Exception as e:
            self.logger.error(f"Error deleting user: {str(e)}")
            return False, f"Delete error: {str(e)}"
