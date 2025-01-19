from fastapi import WebSocket

from typing import Dict
import asyncio


class ConnectionManager:
    def __init__(self):
        self._active_connections: Dict[str, WebSocket] = {}
        self._user_buffers: Dict[str, list] = {}
        self._user_frame_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        async with self._lock:
            self._active_connections[user_id] = websocket
            self._user_buffers[user_id] = []
            self._user_frame_counts[user_id] = 0
            
    async def disconnect(self, user_id: str):
        async with self._lock:
            if user_id in self._active_connections:
                await self._active_connections[user_id].close()
                self._clear_user_data(user_id)
                
    def _clear_user_data(self, user_id: str):
        """Clean up user data to prevent memory leaks"""
        self._active_connections.pop(user_id, None)
        if user_id in self._user_buffers:
            self._user_buffers[user_id].clear()
            self._user_buffers.pop(user_id, None)
        self._user_frame_counts.pop(user_id, None)
        
    def get_buffer(self, user_id: str) -> list:
        return self._user_buffers.get(user_id, [])
    
    def update_buffer(self, user_id: str, frame):
        if user_id in self._user_buffers:
            self._user_buffers[user_id].append(frame)
            if len(self._user_buffers[user_id]) > 30:  # Keep buffer size limited
                self._user_buffers[user_id].pop(0)
                
    def increment_frame_count(self, user_id: str):
        self._user_frame_counts[user_id] = self._user_frame_counts.get(user_id, 0) + 1
        
    def get_frame_count(self, user_id: str) -> int:
        return self._user_frame_counts.get(user_id, 0)