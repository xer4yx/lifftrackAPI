from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy import ndarray, uint8

from cv2 import (
    imdecode, 
    imencode, 
    resize,
    IMREAD_COLOR, 
    IMWRITE_JPEG_QUALITY
)
from cv2.typing import MatLike

# TODO: Implement this class in `services/vision_service.py`
@dataclass
class BytesFrame:
    """Bytes frame entity"""
    data_packet: bytes
    
    @property
    def as_numpy(self) -> ndarray[uint8]:
        return np.frombuffer(self.data_packet, dtype=uint8)

# TODO: Implement this class in `services/vision_service.py`
@dataclass
class Frame:
    """Frame entity"""
    nparray: BytesFrame.as_numpy
    
    @property
    def as_mat(self) -> MatLike:
        return imdecode(self.nparray, IMREAD_COLOR)
    
    @classmethod
    def resize(cls, frame: MatLike, resolution: Tuple[int, int]) -> MatLike:
        """Resize frame to specified resolution"""
        return resize(frame, resolution)
    
    @classmethod
    def encode(cls, frame: MatLike) -> bytes:
        """Encode frame to JPEG format"""
        return imencode('.jpg', frame, [IMWRITE_JPEG_QUALITY, 85])
