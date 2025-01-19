from typing import Optional
from .logger import MonitoringService

class MonitoringFactory:
    _instance: Optional[MonitoringService] = None

    @classmethod
    def get_monitoring_service(cls, app_name: str = "lifttrack", log_dir: str = "logs") -> MonitoringService:
        if not cls._instance:
            cls._instance = MonitoringService(app_name, log_dir)
        return cls._instance

    @classmethod
    def get_logger(cls, module_name: str, app_name: str = "lifttrack", log_dir: str = "logs"):
        monitoring_service = cls.get_monitoring_service(app_name, log_dir)
        return monitoring_service.get_logger(module_name) 