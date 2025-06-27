from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional


class TokenBlacklistRepository(ABC):
    """
    Abstract interface for token blacklist operations.
    This allows for different implementations (Redis, database, etc.)
    while keeping the same interface for business logic.
    """

    @abstractmethod
    async def add_to_blacklist(self, token: str, expiry: datetime) -> bool:
        """
        Add a token to the blacklist.

        Args:
            token: The token to blacklist
            expiry: When the token expires naturally

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def is_blacklisted(self, token: str) -> bool:
        """
        Check if a token is blacklisted.

        Args:
            token: The token to check

        Returns:
            True if token is blacklisted, False otherwise
        """
        pass

    @abstractmethod
    async def remove_expired(self) -> int:
        """
        Remove expired tokens from the blacklist.
        This should be called periodically to clean up.

        Returns:
            Number of tokens removed
        """
        pass

    @abstractmethod
    async def get_all_blacklisted(self) -> List[dict]:
        """
        Get all blacklisted tokens.
        Mainly for debugging/admin purposes.

        Returns:
            List of token records with their expiry times
        """
        pass
