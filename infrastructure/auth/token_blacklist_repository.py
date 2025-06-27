from datetime import datetime, timezone
from typing import Dict, List

from core.interface.token_blacklist_interface import TokenBlacklistRepository


class InMemoryTokenBlacklistRepository(TokenBlacklistRepository):
    """
    In-memory implementation of token blacklist repository.
    Useful for development and testing.
    Not suitable for production as it doesn't persist across restarts.
    """

    def __init__(self):
        self.blacklisted_tokens: Dict[str, datetime] = {}

    async def add_to_blacklist(self, token: str, expiry: datetime) -> bool:
        """Add a token to the blacklist with its expiry time."""
        self.blacklisted_tokens[token] = expiry
        return True

    async def is_blacklisted(self, token: str) -> bool:
        """Check if a token is in the blacklist and not expired."""
        if token not in self.blacklisted_tokens:
            return False

        # If token is in blacklist but expired, remove it and return False
        if self.blacklisted_tokens[token] < datetime.now(timezone.utc):
            del self.blacklisted_tokens[token]
            return False

        return True

    async def remove_expired(self) -> int:
        """Remove all expired tokens from the blacklist."""
        now = datetime.now(timezone.utc)
        expired_tokens = [
            token for token, expiry in self.blacklisted_tokens.items() if expiry < now
        ]

        for token in expired_tokens:
            del self.blacklisted_tokens[token]

        return len(expired_tokens)

    async def get_all_blacklisted(self) -> List[dict]:
        """Get all blacklisted tokens with their expiry times."""
        return [
            {"token": token, "expiry": expiry}
            for token, expiry in self.blacklisted_tokens.items()
        ]
