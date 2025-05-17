from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class InferenceInterface(ABC):
    @abstractmethod
    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        pass
