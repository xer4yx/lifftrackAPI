from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class NTFInterface(ABC):
    @abstractmethod
    async def get_data(self, key: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def set_data(self, key: str, value: Any) -> None:
        pass
    
    @abstractmethod
    async def delete_data(self, key: str) -> None:
        pass
