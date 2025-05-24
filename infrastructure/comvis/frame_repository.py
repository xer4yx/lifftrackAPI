import asyncio
import json
import numpy as np
import cv2
from typing import Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from core.interface.frame_repository_interface import FrameRepositoryInterface
from lifttrack.utils.logging_config import setup_logger

logger = setup_logger("frame-repository", "frame_repository.log")

class FrameRepository(FrameRepositoryInterface):
    """
    Repository for handling frame processing operations.
    This repository is responsible for converting, processing, and creating frames.
    """
    
    def convert_byte_to_numpy(self, byte_data: bytes | Any) -> np.ndarray:
        """
        Convert a byte array or any other type to a numpy array.
        
        Args:
            byte_data: Byte data to convert
            
        Returns:
            Numpy array representation of the byte data
            
        Raises:
            ValueError: If input is not a byte array
        """
        try:
            if not isinstance(byte_data, bytes):
                logger.error(f"Expected byte array, got {type(byte_data)}")
                raise ValueError("Input must be a byte array or numpy array")
            return cv2.imdecode(np.frombuffer(byte_data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Failed to decode frame: {str(e)}")
            raise

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single frame and return resized version in JPEG format.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Tuple of (processed frame, buffer)
            
        Raises:
            Exception: If frame processing fails
        """
        try:
            frame = cv2.resize(frame, (192, 192))
            _, buffer = cv2.imencode(".jpeg", frame, [cv2.IMWRITE_JPEG_OPTIMIZE, 85])
            return frame, buffer
        except Exception as e:
            logger.error(f"Failed to process frame: {str(e)}")
            raise
            
    def create_frame_from_planes(self, primary_plane: bytes, 
                               secondary_plane_1: bytes, 
                               secondary_plane_2: bytes, 
                               width: int, height: int) -> np.ndarray:
        """
        Create a frame from separate color planes.
        
        Args:
            primary_plane: Primary color plane data (Y plane)
            secondary_plane_1: First secondary color plane data (U plane)
            secondary_plane_2: Second secondary color plane data (V plane)
            width: Frame width
            height: Frame height
            
        Returns:
            BGR frame as numpy array
        """
        return self.create_yuv420_frame(primary_plane, secondary_plane_1, secondary_plane_2, width, height)
            
    def create_yuv420_frame(self, y_plane_bytes: bytes, u_plane_bytes: bytes, 
                          v_plane_bytes: bytes, width: int, height: int) -> np.ndarray:
        """
        Create a BGR frame from YUV planes.
        
        Args:
            y_plane_bytes: Y plane data
            u_plane_bytes: U plane data
            v_plane_bytes: V plane data
            width: Frame width
            height: Frame height
            
        Returns:
            BGR frame as numpy array
        """
        try:        
            # Convert Y plane to numpy array
            y = np.frombuffer(y_plane_bytes, dtype=np.uint8).reshape((height, width))
            
            # Create a grayscale image from Y plane as a fallback
            gray_image = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
            
            # Rotate the grayscale image if needed
            if width > height:  # If image is in landscape but should be portrait
                gray_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)
            
            y_size = width * height
            uv_ratio = len(u_plane_bytes) / y_size
            
            try:
                if 0.49 <= uv_ratio <= 0.51:
                    # Calculate dimensions for U and V planes
                    uv_width = width // 2
                    uv_height = height // 2
                    
                    # Reshape U and V planes
                    u = np.frombuffer(u_plane_bytes, dtype=np.uint8)[:uv_width * uv_height].reshape((uv_height, uv_width))
                    v = np.frombuffer(v_plane_bytes, dtype=np.uint8)[:uv_width * uv_height].reshape((uv_height, uv_width))
                    
                    # Try NV12 format conversion first
                    try:
                        # Create NV12 format (YUV420sp)
                        nv12 = np.zeros((height * 3 // 2, width), dtype=np.uint8)
                        nv12[0:height, :] = y
                        
                        # Create and copy interleaved UV plane
                        uv = np.zeros((height//2, width), dtype=np.uint8)
                        for i in range(height//2):
                            for j in range(width//2):
                                uv[i, j*2] = u[i, j]
                                uv[i, j*2+1] = v[i, j]
                        nv12[height:, :] = uv
                        
                        # Convert to BGR
                        nv12 = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
                        return cv2.rotate(nv12, cv2.ROTATE_90_CLOCKWISE) if width > height else nv12
                        
                    except Exception as e:
                        logger.error(f"Error in NV12 conversion: {str(e)}")
                        
                        # Try I420 format as fallback
                        try:
                            # Create I420 format (YUV420p)
                            i420 = np.zeros((height * 3 // 2, width), dtype=np.uint8)
                            i420[0:height, :] = y
                            i420[height:height+height//4, :width] = cv2.resize(u, (width, height//4))
                            i420[height+height//4:, :width] = cv2.resize(v, (width, height//4))
                            
                            # Convert to BGR
                            i420 = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
                            return cv2.rotate(i420, cv2.ROTATE_90_CLOCKWISE) if width > height else i420
                            
                        except Exception as e2:
                            logger.error(f"Error in I420 conversion: {str(e2)}")
                            
                            # Try direct conversion as last resort
                            try:
                                # Resize U and V to full resolution
                                u_full = cv2.resize(u, (width, height))
                                v_full = cv2.resize(v, (width, height))
                                
                                # Convert to float32 and adjust range
                                y_float = y.astype(np.float32)
                                u_float = (u_full.astype(np.float32) - 128) * 0.872
                                v_float = (v_full.astype(np.float32) - 128) * 1.230
                                
                                # Convert to BGR using matrix multiplication
                                b = y_float + 2.032 * u_float
                                g = y_float - 0.395 * u_float - 0.581 * v_float
                                r = y_float + 1.140 * v_float
                                
                                # Clip values and convert back to uint8
                                b = np.clip(b, 0, 255).astype(np.uint8)
                                g = np.clip(g, 0, 255).astype(np.uint8)
                                r = np.clip(r, 0, 255).astype(np.uint8)
                                
                                # Merge channels
                                nv12 = cv2.merge([b, g, r])
                                
                                # Rotate the BGR image if needed
                                if width > height:  # If image is in landscape but should be portrait
                                    nv12 = cv2.rotate(nv12, cv2.ROTATE_90_CLOCKWISE)
                                logger.info("Successfully converted YUV to BGR using direct conversion")
                                return nv12
                            except Exception as e3:
                                logger.error(f"Error in direct conversion: {str(e3)}")
                                return cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE) if width > height else gray_image
                
                logger.warning("Could not convert YUV data, returning grayscale image")
                return cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE) if width > height else gray_image
                
            except Exception as e:
                logger.error(f"Error in YUV conversion: {str(e)}")
                return cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE) if width > height else gray_image
                
        except Exception as e:
            logger.error(f"Error creating YUV frame: {str(e)}")
            # Create a blank image with correct orientation
            if width > height:
                return np.zeros((width, height, 3), dtype=np.uint8)  # Swapped dimensions for portrait
            return np.zeros((height, width, 3), dtype=np.uint8) 
        
    async def parse_frame(self, byte_data: bytes) -> Optional[Any]:
        """
        Parse a frame from byte data.
        
        Args:
            byte_data: Byte data to parse
            
        Returns:
            Processed frame data
        """
        try:
            # Parse the header
            header_length = int.from_bytes(byte_data[0:4], byteorder='big')
            header_bytes = byte_data[4:4+header_length]
            header_json = header_bytes.decode('utf-8')
            header = json.loads(header_json)
            
            # Extract image dimensions and format
            width = header['width']
            height = header['height']
            y_size = header['ySize']
            u_size = header['uSize']
            v_size = header['vSize']
            
            # Extract the YUV planes
            data_start = 4 + header_length
            y_plane = byte_data[data_start:data_start+y_size]
            u_plane = byte_data[data_start+y_size:data_start+y_size+u_size]
            v_plane = byte_data[data_start+y_size+u_size:data_start+y_size+u_size+v_size]
            
            # Create a frame using ComVisUseCase
            frame = self.create_frame_from_planes(
                y_plane, u_plane, v_plane, width, height
            )
            
            if frame is None or frame.size == 0:
                return None
                
            return frame
        except Exception as e:
            logger.error(f"Error parsing frame: {str(e)}")
            return None
        
    async def process_frame_async(self, frame: Any, thread_pool: ThreadPoolExecutor) -> Any:
        """
        Process a frame asynchronously.
        """
        loop = asyncio.get_running_loop()
        processed_frame, buffer = await loop.run_in_executor(
            thread_pool, self.process_frame, frame
        )
        return processed_frame