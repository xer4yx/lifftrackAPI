import unittest
import logging
import os
import base64
import cv2
import numpy as np
from unittest import TestCase
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from unittest.mock import Mock, patch
from main import app
from lifttrack.utils.logging_config import setup_logger

# Configure logging for test_websocket
logger = setup_logger("test_websocket", "test_cases.log")


class TestWebSocketEndpoint(TestCase):
    def setUp(self):
        logger.info(f"Setting up {self.__class__.__name__}")
        self.client = TestClient(app)
        
        # Create a sample frame for testing
        self.test_frame = np.zeros((100, 100, 3), dtype=np.uint8)  # Black image
        _, buffer = cv2.imencode('.jpeg', self.test_frame)
        self.test_frame_bytes = buffer.tobytes()
        
        logger.debug("Test frame created successfully")

    @patch('lifttrackAPI.main.websocket_process_frames')
    async def test_websocket_connection(self, mock_process_frames):
        logger.info(f"Started testing {self.test_websocket_connection.__name__}")
        
        # Mock the frame processing function
        mock_process_frames.return_value = (self.test_frame, {})
        
        try:
            # Connect to websocket
            with self.client.websocket_connect("/ws-tracking") as websocket:
                # Send frame data
                data = {
                    "type": "websocket.receive",
                    "bytes": self.test_frame_bytes
                }
                await websocket.send_json(data)
                logger.debug("Frame data sent successfully")
                
                # Receive response
                response = await websocket.receive_bytes()
                logger.debug("Response received from websocket")
                
                # Verify response can be decoded back to an image
                np_arr = np.frombuffer(base64.b64encode(response), np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                self.assertIsNotNone(img)
                mock_process_frames.assert_called_once()
                logger.debug("Frame processing verified successfully")
                
        except Exception as e:
            logger.error(f"Error in websocket connection test: {str(e)}")
            raise
            
        logger.info(f"{self.test_websocket_connection.__name__} testing completed.")

    async def test_websocket_invalid_data(self):
        logger.info(f"Started testing {self.test_websocket_invalid_data.__name__}")
        
        try:
            with self.client.websocket_connect("/ws-tracking") as websocket:
                # Send invalid data
                data = {
                    "type": "websocket.receive",
                    "bytes": b"invalid_data"
                }
                await websocket.send_json(data)
                logger.debug("Invalid data sent to websocket")
                
                # Connection should close due to error
                with self.assertRaises(Exception):
                    await websocket.receive_bytes()
                logger.debug("Websocket closed as expected with invalid data")
                
        except Exception as e:
            logger.error(f"Error in invalid data test: {str(e)}")
            raise
            
        logger.info(f"{self.test_websocket_invalid_data.__name__} testing completed.")

    async def test_websocket_close(self):
        logger.info(f"Started testing {self.test_websocket_close.__name__}")
        
        try:
            with self.client.websocket_connect("/ws-tracking") as websocket:
                # Send close signal
                data = {
                    "type": "websocket.close"
                }
                await websocket.send_json(data)
                logger.debug("Close signal sent to websocket")
                
                # Verify connection closes
                with self.assertRaises(Exception):
                    await websocket.receive_bytes()
                logger.debug("Websocket closed successfully")
                
        except Exception as e:
            logger.error(f"Error in websocket close test: {str(e)}")
            raise
            
        logger.info(f"{self.test_websocket_close.__name__} testing completed.")

    async def test_websocket_large_frame(self):
        logger.info(f"Started testing {self.test_websocket_large_frame.__name__}")
        
        try:
            # Create a large test frame
            large_frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpeg', large_frame)
            large_frame_bytes = buffer.tobytes()
            logger.debug("Large test frame created")
            
            with self.client.websocket_connect("/ws-tracking") as websocket:
                data = {
                    "type": "websocket.receive",
                    "bytes": large_frame_bytes
                }
                await websocket.send_json(data)
                logger.debug("Large frame sent to websocket")
                
                # Receive response
                response = await websocket.receive_bytes()
                logger.debug("Response received for large frame")
                
                # Verify response
                self.assertIsNotNone(response)
                
        except Exception as e:
            logger.error(f"Error in large frame test: {str(e)}")
            raise
            
        logger.info(f"{self.test_websocket_large_frame.__name__} testing completed.")

    async def test_websocket_multiple_frames(self):
        logger.info(f"Started testing {self.test_websocket_multiple_frames.__name__}")
        
        try:
            with self.client.websocket_connect("/ws-tracking") as websocket:
                # Send multiple frames
                for i in range(3):
                    data = {
                        "type": "websocket.receive",
                        "bytes": self.test_frame_bytes
                    }
                    await websocket.send_json(data)
                    logger.debug(f"Frame {i+1} sent to websocket")
                    
                    response = await websocket.receive_bytes()
                    logger.debug(f"Response {i+1} received from websocket")
                    
                    self.assertIsNotNone(response)
                    
        except Exception as e:
            logger.error(f"Error in multiple frames test: {str(e)}")
            raise
            
        logger.info(f"{self.test_websocket_multiple_frames.__name__} testing completed.")


if __name__ == '__main__':
    unittest.main() 