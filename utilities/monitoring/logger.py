import os
import logging
import time
import psutil
import threading
from typing import Optional, Dict, Any
from datetime import datetime

from .logging import setup_logger
from .metrics import MetricsCollector, JSONFileExporter

class MonitoringService:
    def __init__(
        self,
        app_name: str,
        log_dir: str = "logs",
        metrics_dir: str = "metrics"
    ):
        self.app_name = app_name
        self.log_dir = log_dir
        self.metrics_dir = metrics_dir
        self.loggers: Dict[str, logging.Logger] = {}
        self.metrics_collector = MetricsCollector()
        self.metrics_exporter = JSONFileExporter(metrics_dir)
        
        # Set up system metrics collection
        self._setup_system_metrics()
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger for the specified name"""
        if name not in self.loggers:
            log_file = os.path.join(self.log_dir, f"{name}.log")
            self.loggers[name] = setup_logger(
                f"{self.app_name}.{name}",
                log_file
            )
        return self.loggers[name]
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value"""
        self.metrics_collector.record(name, value, labels)
    
    def _setup_system_metrics(self) -> None:
        """Set up system metrics collection"""
        def collect_metrics():
            while True:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.record_metric("system.cpu.usage", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_metric("system.memory.usage", memory.percent)
                self.record_metric(
                    "system.memory.available",
                    memory.available / 1024 / 1024  # MB
                )
                
                # Export metrics every hour
                current_time = datetime.now()
                if current_time.minute == 0 and current_time.second == 0:
                    self.metrics_exporter.export(
                        self.metrics_collector.get_metrics()
                    )
                
                time.sleep(60)  # Collect metrics every minute
        
        thread = threading.Thread(
            target=collect_metrics,
            daemon=True,
            name="system-metrics"
        )
        thread.start() 