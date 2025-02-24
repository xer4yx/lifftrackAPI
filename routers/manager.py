
import time
import httpx
from contextlib import asynccontextmanager
from typing import Dict, Optional, Generator, AsyncGenerator
from fastapi import WebSocket
from .headers import HEADER_BASED_OPTIMIZERS


class HTTPConnectionPool:
    @staticmethod
    @asynccontextmanager
    async def get_session():
        async with httpx.AsyncClient(
            headers=HEADER_BASED_OPTIMIZERS,
            timeout=httpx.Timeout(timeout=30),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=30)
        ) as session:
            yield session


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict] = {}  # Track user-specific session data

    async def connect(self, websocket: WebSocket, username: str, exercise_name: str):
        await websocket.accept()
        session_id = f"{username}_{exercise_name}_{int(time.time())}"
        self.active_connections[session_id] = websocket
        self.user_sessions[session_id] = {
            "username": username,
            "exercise_name": exercise_name,
            "start_time": time.time(),
            "frame_count": 0
        }
        return session_id

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.user_sessions:
            del self.user_sessions[session_id]

    async def send_personal_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)

    async def send_json_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        return self.user_sessions.get(session_id)

    def update_frame_count(self, session_id: str, count: int):
        if session_id in self.user_sessions:
            self.user_sessions[session_id]["frame_count"] = count
        
    