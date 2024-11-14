import pytest
import numpy as np
import cv2
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from fastapi.websockets import WebSocketDisconnect
from unittest.mock import patch
from routers.WebsocketRouter import router

# Create a FastAPI app for testing
@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

@pytest.fixture
def mock_frame():
    # Create a simple test image (300x300 black image)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    return frame

@pytest.fixture
def encoded_frame(mock_frame):
    # Encode frame to bytes
    _, buffer = cv2.imencode('.jpg', mock_frame)
    return buffer.tobytes()

# ... existing imports ...
from fastapi.websockets import WebSocketDisconnect

@pytest.mark.asyncio
async def test_websocket_endpoint(client, encoded_frame):
    """Test WebSocket endpoint for image processing and response handling."""
    async with client.websocket_connect("/v2/ws-tracking") as websocket:
        # Send encoded image data
        await websocket.send_bytes(encoded_frame)

        # Receive response
        response = await websocket.receive_json()

        # Check if the response contains expected keys
        assert 'predicted_class' in response
        assert 'form_suggestions' in response
        assert 'progress_suggestions' in response
        assert 'frame_count' in response

        # Optionally, check the values of the response
        assert response['predicted_class'] is not None  # Ensure a prediction was made
        assert isinstance(response['form_suggestions'], list)  # Ensure form suggestions is a list
        assert isinstance(response['progress_suggestions'], list)  # Ensure progress suggestions is a list
        assert isinstance(response['frame_count'], int)  # Ensure frame count is an integer