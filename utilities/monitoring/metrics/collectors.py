from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone
import time
import threading

from core.entities.monitoring import Metric

class MetricsCollector:
    def __init__(self):
        self._metrics: Dict[str, List[Metric]] = {}
        self._lock = threading.Lock()
    
    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value"""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []
            self._metrics[name].append(
                Metric(name=name, value=value, labels=labels or {})
            )
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, List[Metric]]:
        """Get recorded metrics"""
        with self._lock:
            if name:
                return {name: self._metrics.get(name, [])}
            return self._metrics.copy()
    
    def clear_metrics(self, name: Optional[str] = None) -> None:
        """Clear recorded metrics"""
        with self._lock:
            if name:
                self._metrics.pop(name, None)
            else:
                self._metrics.clear() 