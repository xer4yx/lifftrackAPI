"""
WebSocket Authentication Utilities

This module provides authentication utilities specifically designed for WebSocket connections,
since standard OAuth2PasswordBearer doesn't work with WebSocket ASGI scope.
"""

from fastapi import WebSocket, HTTPException, status
from typing import Optional, Tuple
import logging

from core.usecase import AuthUseCase

logger = logging.getLogger(__name__)


async def authenticate_websocket_query_param(
    websocket: WebSocket,
    token: str,
    username: str,
    auth_service: AuthUseCase
) -> Tuple[bool, Optional[object], Optional[str]]:
    """
    Authenticate WebSocket connection using token from query parameter.
    
    This is the simplest approach but token may be logged in server logs.
    Use this for development or when logging is properly secured.
    
    Args:
        websocket: FastAPI WebSocket connection
        token: JWT token from query parameter
        username: Expected username from query parameter
        auth_service: Authentication use case service
        
    Returns:
        Tuple of (success, user_object, error_message)
    """
    try:
        with_session, client, error = await auth_service.validate_token(token)
        
        if not with_session:
            logger.warning(f"WebSocket auth failed: {error}")
            return False, None, error or "Invalid token"
            
        if client.username != username:
            logger.warning(f"Username mismatch in WebSocket auth: {client.username} != {username}")
            return False, None, "Username does not match token"
            
        logger.info(f"WebSocket authenticated successfully for user: {username}")
        return True, client, None
        
    except Exception as e:
        logger.error(f"WebSocket authentication error: {str(e)}")
        return False, None, "Authentication failed"


async def authenticate_websocket_subprotocol(
    websocket: WebSocket,
    username: str,
    auth_service: AuthUseCase,
    expected_subprotocol: str = "livestream-v3"
) -> Tuple[bool, Optional[object], Optional[str]]:
    """
    Authenticate WebSocket connection using token from Sec-WebSocket-Protocol header.
    
    This approach is more secure as the token is in headers rather than query params,
    reducing the risk of token exposure in logs. Used by Kubernetes and other systems.
    
    Expected format: ["Authorization", "Bearer TOKEN_HERE", "livestream-v3"]
    or: ["token.jwt.here", "livestream-v3"]
    
    Args:
        websocket: FastAPI WebSocket connection
        username: Expected username
        auth_service: Authentication use case service
        expected_subprotocol: Expected subprotocol name
        
    Returns:
        Tuple of (success, user_object, error_message)
    """
    try:
        # Parse Sec-WebSocket-Protocol header
        protocol_header = websocket.headers.get("sec-websocket-protocol", "")
        protocols = [p.strip() for p in protocol_header.split(",")]
        
        if expected_subprotocol not in protocols:
            return False, None, f"Missing required subprotocol: {expected_subprotocol}"
        
        # Look for token in protocols
        token = None
        
        # Method 1: Authorization Bearer pattern
        if "Authorization" in protocols:
            try:
                auth_index = protocols.index("Authorization")
                if auth_index + 1 < len(protocols):
                    bearer_token = protocols[auth_index + 1]
                    if bearer_token.startswith("Bearer"):
                        token = bearer_token.replace("Bearer", "").strip()
                    else:
                        token = bearer_token
            except (ValueError, IndexError):
                pass
        
        # Method 2: Direct token (not "Authorization" or expected_subprotocol)
        if not token:
            for protocol in protocols:
                if protocol not in ["Authorization", expected_subprotocol] and len(protocol) > 20:
                    # Assume this is the token (JWT tokens are typically long)
                    token = protocol
                    break
        
        if not token:
            return False, None, "No authentication token found in subprotocol"
        
        # Validate token
        with_session, client, error = await auth_service.validate_token(token)
        
        if not with_session:
            logger.warning(f"WebSocket subprotocol auth failed: {error}")
            return False, None, error or "Invalid token"
            
        if client.username != username:
            logger.warning(f"Username mismatch in WebSocket subprotocol auth: {client.username} != {username}")
            return False, None, "Username does not match token"
            
        logger.info(f"WebSocket authenticated successfully via subprotocol for user: {username}")
        return True, client, None
        
    except Exception as e:
        logger.error(f"WebSocket subprotocol authentication error: {str(e)}")
        return False, None, "Authentication failed"


async def close_websocket_with_auth_error(
    websocket: WebSocket,
    error_message: str,
    code: int = status.WS_1008_POLICY_VIOLATION
) -> None:
    """
    Close WebSocket connection with authentication error.
    
    Args:
        websocket: FastAPI WebSocket connection
        error_message: Error message to send
        code: WebSocket close code
    """
    try:
        await websocket.close(code=code, reason=error_message)
        logger.info(f"Closed WebSocket connection: {error_message}")
    except Exception as e:
        logger.error(f"Error closing WebSocket: {str(e)}") 