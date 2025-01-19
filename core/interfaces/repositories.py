from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from core.entities import Exercise

class FrameRepository(ABC):
    @abstractmethod
    async def save_frame_data(self, metadata: Dict[str, Any]) -> str:
        """Save frame metadata and return the ID"""
        pass

    @abstractmethod
    async def get_frame_data(self, frame_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve frame data by ID"""
        pass

    @abstractmethod
    async def list_frame_data(
        self, 
        user_id: str, 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """List frame data for a user within a date range"""
        pass
