from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class InferenceInterface(ABC):
    @abstractmethod
    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def infer_async(self, image: np.ndarray) -> Dict[str, Any]:
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Clean up resources used by the inference service.
        This method should be called when shutting down the service to properly
        release memory, close connections, and clean up any other resources.
        """
        pass
