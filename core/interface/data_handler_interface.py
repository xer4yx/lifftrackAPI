from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

T1 = TypeVar("T1")
T2 = TypeVar("T2")


class DataHandlerInterface(ABC, Generic[T1, T2]):
    """
    Interface for data handlers.

    Args:
        T1: return type for load_to_data_model method.
        T2: return type for save_data method.
    """

    @abstractmethod
    def load_to_data_model(self, *args, **kwargs) -> T1:
        """Load data to a data model."""

    @abstractmethod
    def save_data(self, *args, **kwargs) -> Optional[T2]:
        """Save data to a database."""

    @abstractmethod
    def format_date(self, date: str) -> str:
        """Format a date string to a more readable format."""
