import pytest
from fastapi.testclient import TestClient
from routers.WebsocketRouter import router

client = TestClient(router)

def test_websocket_connection():
    with client.websocket_connect("/ws-tracking") as websocket:
        assert websocket is not None
        assert websocket.client_state.CONNECTED
        websocket.close()