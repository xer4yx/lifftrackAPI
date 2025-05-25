from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional


class FrameRepositoryInterface(ABC):
    """
    Abstract interface for frame repositories.
    Defines the contract that all frame repository implementations must fulfill.
    """
    
    @abstractmethod
    def convert_byte_to_numpy(self, byte_data: bytes | Any) -> Any:
        """
        Convert byte data to a processable frame format.
        
        Args:
            byte_data: Byte data to convert
            
        Returns:
            Converted frame data
        """
        pass
        
    @abstractmethod
    def process_frame(self, frame: Any) -> Tuple[Any, Any]:
        """
        Process a frame to prepare it for analysis.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Tuple of (processed_frame, buffer)
        """
        pass
        
    @abstractmethod
    def create_frame_from_planes(self, primary_plane: bytes, 
                               secondary_plane_1: bytes, 
                               secondary_plane_2: bytes, 
                               width: int, height: int) -> Any:
        """
        Create a frame from separate color planes.
        
        Args:
            primary_plane: Primary color plane data
            secondary_plane_1: First secondary color plane data
            secondary_plane_2: Second secondary color plane data
            width: Frame width
            height: Frame height
            
        Returns:
            Processed frame data
        """
        pass
    
    @abstractmethod
    async def parse_frame(self, byte_data: bytes) -> Any:
        """
        Parse a frame from byte data.
        
        Args:
            byte_data: Byte data to parse
            
        Returns:
            Processed frame data
        """
        pass
    
    @abstractmethod
    async def process_frame_async(self, frame: Any, *args, **kwargs) -> Any:
        """
        Process a frame asynchronously.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Processed frame data
        """
        pass
