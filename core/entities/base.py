from datetime import datetime, timezone
from dataclasses import dataclass, field
import uuid

@dataclass
class EntityBase:
    """Base class for all entities"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
        
@dataclass
class EntityDefaultBase:
    """Base class for all entities with default values"""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_deleted: bool = False

    def mark_as_deleted(self) -> None:
        """Mark entity as deleted"""
        self.is_deleted = True
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def update_timestamp(self) -> None:
        """Update the last modified timestamp"""
        self.updated_at = datetime.now(timezone.utc).isoformat() 
