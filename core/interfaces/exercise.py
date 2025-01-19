from abc import ABC, abstractmethod
from typing import Optional

class ExerciseRepository(ABC):
    @abstractmethod
    def save(self, data: object) -> None:
        pass

    @abstractmethod
    def load(self, data_type: type) -> Optional[object]:
        pass
