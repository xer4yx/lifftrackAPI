from abc import ABC, abstractmethod
from typing import Dict, List
from core.entities.monitoring import Metric

class MetricsExporter(ABC):
    @abstractmethod
    def export(self, metrics: Dict[str, List[Metric]]) -> None:
        pass