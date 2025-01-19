from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict

@dataclass
class Metric:
    name: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)