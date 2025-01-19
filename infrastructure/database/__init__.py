"""
Database
- Implements data persistence strategies
- Contains database-specific logic
- Provides concrete implementations of repository interfaces
- Handles data access and storage mechanisms
"""

from .factory import DatabaseFactory

__all__ = [
    "DatabaseFactory"
]

