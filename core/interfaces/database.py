from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class DatabaseRepository(ABC):
    @abstractmethod
    async def set(self, path: str, data: Dict[str, Any], key: Optional[str] = None) -> str:
        """Set data at specified path"""
        pass
    
    @abstractmethod
    async def push(self, path: str, data: Dict[str, Any]) -> str:
        """Push data to specified path"""
        pass
    
    @abstractmethod
    async def get(self, path: str, key: str) -> Optional[Dict[str, Any]]:
        """Get data from specified path"""
        pass
    
    @abstractmethod
    async def update(self, path: str, key: str, data: Dict[str, Any]) -> bool:
        """Update data at specified path"""
        pass
    
    @abstractmethod
    async def delete(self, path: str, key: str) -> bool:
        """Delete data at specified path"""
        pass
    
    @abstractmethod
    async def query(
        self,
        path: str,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        start_at: Optional[Any] = None,
        end_at: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Query data with filters"""
        pass
