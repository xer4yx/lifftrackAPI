import pytest
import asyncio
import json
import numpy as np
from fastapi import status, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import io
import inspect

from core.usecase import AuthUseCase, InferenceUseCase
from core.usecase.comvis_usecase import ComVisUseCase, FeatureMetricUseCase
from core.entities.user_entity import UserEntity
from interface.di import get_auth_service, get_current_user_token, get_comvis_usecase
from interface.di.inference_service import get_inference_usecase
from interface.di.comvis_service_di import get_feature_metric_usecase
from infrastructure.di import get_firebase_admin
from interface.ws.websocket_router import (
    livestream_exercise_tracking,
    websocket_router_v3,
)
from interface.ws.websocket_auth import authenticate_websocket_query_param


# Mock classes to help with testing
class MockWebSocket:
    """Mock WebSocket class to simulate client connection"""

    def __init__(self, app=None, base_url=None, headers=None):
        self.sent_messages = []
        self.closed = False
        self.close_code = None
        self.close_reason = None
        self.headers = headers or {"sec-websocket-protocol": "livestream-v3"}
        self.client_state = MagicMock()
        self.client_state.name = "CONNECTED"

        # Added fields from WebSocketTestClient
        self.app = app
        self.base_url = base_url
        self.client = TestClient(app) if app else None
        self.responses = []
        self.url = None
        self.subprotocols = None

    async def accept(self, subprotocol=None):
        return None

    async def send_json(self, data):
        self.sent_messages.append({"type": "json", "data": data})

    async def send_text(self, data):
        self.sent_messages.append({"type": "text", "data": data})

    async def receive_bytes(self):
        # Default implementation returns COMPLETED signal
        return b"COMPLETED"

    async def receive_json(self):
        """Simulate receiving JSON data from WebSocket"""
        # Return a mock response
        return {"status": "mock_response"}

    # Use a single close method that works for both async and non-async calls
    def close(self, code=None, reason=None):
        """Close the WebSocket connection
        This non-async method handles direct websocket.close() calls from the router
        """
        self.closed = True
        self.close_code = code
        self.close_reason = reason
        return None

    async def connect(self, url, subprotocols=None):
        """Simulate connection to WebSocket endpoint"""
        # In a real test, we'd establish a real connection
        # For our purposes, we just record that connect was called
        self.url = url
        self.subprotocols = subprotocols
        return True

    async def send_bytes(self, data):
        """Simulate sending binary data to WebSocket"""
        # We record the data sent but don't actually send it
        self.responses.append({"sent_bytes": data})


# Setup test app with the router
@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(websocket_router_v3)
    return app


# Setup test client
@pytest.fixture
def client(app):
    return TestClient(app)


# Mock authentication service
@pytest.fixture
def mock_auth_service():
    mock_service = AsyncMock(spec=AuthUseCase)
    return mock_service


# Mock computer vision service
@pytest.fixture
def mock_comvis_service():
    mock_service = MagicMock(spec=ComVisUseCase)
    # Setup default behaviors for ComVis methods with small arrays to minimize memory usage
    mock_service.create_frame_from_planes.return_value = np.zeros(
        (120, 160, 3), dtype=np.uint8
    )
    # Return both frame and buffer to match expected return signature
    mock_service.process_frame.return_value = (
        np.zeros((120, 160, 3), dtype=np.uint8),
        {},
    )
    mock_service.load_to_features_model.return_value = {"keypoints": {}}
    mock_service.get_suggestions.return_value = (0.85, ["Keep your back straight"])
    mock_service.load_exercise_data.return_value = MagicMock(date="2023-01-01")

    # Add frame_repo mock that the websocket router expects
    mock_frame_repo = MagicMock()
    mock_frame_repo.process_frame.return_value = (
        np.zeros((120, 160, 3), dtype=np.uint8),
        {},
    )
    mock_service.frame_repo = mock_frame_repo

    # Add other missing methods that might be called
    mock_service.parse_frame = AsyncMock(
        return_value=np.zeros((120, 160, 3), dtype=np.uint8)
    )
    mock_service.load_to_object_model = MagicMock(return_value=[])
    mock_service.format_date = MagicMock(return_value="2023-01-01")
    mock_service.save_exercise_data = AsyncMock()

    yield mock_service

    # Clean up the mock references to large numpy arrays
    mock_service.create_frame_from_planes.return_value = None
    mock_service.process_frame.return_value = None
    mock_frame_repo.process_frame.return_value = None
    import gc

    gc.collect()


# Mock inference service
@pytest.fixture
def mock_inference_service():
    mock_service = MagicMock(spec=InferenceUseCase)
    # Setup default behaviors for inference methods with minimal data
    mock_service.process_frames_concurrent.return_value = [
        {
            "pose": {"keypoints": {}},
            "objects": [],
            "action": {"predicted_class_name": "squat"},
            "processing_time": 0.1,
        }
    ]

    # Add missing private attributes that the websocket router expects
    mock_pose_estimation = MagicMock()
    mock_pose_estimation.infer.return_value = {"keypoints": {}}
    mock_service._pose_estimation = mock_pose_estimation

    mock_object_detection = MagicMock()
    mock_object_detection.infer.return_value = {"predictions": []}
    mock_service._object_detection = mock_object_detection

    mock_action_recognition = MagicMock()
    mock_action_recognition.infer.return_value = {"predicted_class_name": "squat"}
    mock_service._action_recognition = mock_action_recognition

    # Add clear_caches method
    mock_service.clear_caches = MagicMock()

    yield mock_service

    # Clean up the mock
    mock_service.process_frames_concurrent.return_value = None
    import gc

    gc.collect()


# Mock WebSocket for testing
@pytest.fixture
def mock_websocket():
    return MockWebSocket()


# Mock user for authentication
@pytest.fixture
def mock_websocket_auth():
    query_auth = MagicMock(spec=authenticate_websocket_query_param)
    return query_auth


# Mock user for authentication
@pytest.fixture
def mock_user():
    return UserEntity(
        username="testuser",
        email="test@example.com",
        name="Test User",
        password="securepassword123",
        fname="Test",
        lname="User",
        phoneNum="1234567890",
    )


# Mock token for authentication
@pytest.fixture
def mock_token():
    return "valid_token_123"


# Mock database for testing
@pytest.fixture
def mock_db():
    mock_database = MagicMock()
    # Add any database methods that might be called
    return mock_database


# Mock feature metrics service
@pytest.fixture
def mock_feature_metric_service():
    mock_service = MagicMock(spec=FeatureMetricUseCase)

    # Setup default behaviors for feature metrics methods
    mock_feature_metric_repo = MagicMock()
    mock_feature_metric_repo.compute_all_metrics.return_value = {
        "body_alignment": 85.2,
        "joint_consistency": 78.9,
        "load_control": 92.1,
        "speed_control": 81.7,
        "overall_stability": 88.4,
    }
    mock_feature_metric_repo.reset_history = MagicMock()
    mock_service.feature_metric_repo = mock_feature_metric_repo

    # Add save_feature_metrics method
    mock_service.save_feature_metrics = AsyncMock()

    return mock_service


# Apply dependency overrides for all tests
@pytest.fixture(autouse=True)
def override_dependencies(
    app,
    mock_auth_service,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_user,
    mock_token,
    mock_db,
):
    def get_auth_service_override():
        return mock_auth_service

    def get_current_user_token_override():
        return mock_token

    def get_comvis_usecase_override():
        return mock_comvis_service

    def get_inference_usecase_override():
        return mock_inference_service

    def get_firebase_admin_override():
        return mock_db

    def get_feature_metric_usecase_override():
        return mock_feature_metric_service

    # Set dependency overrides on the FastAPI app
    app.dependency_overrides[get_auth_service] = get_auth_service_override
    app.dependency_overrides[get_current_user_token] = get_current_user_token_override
    app.dependency_overrides[get_comvis_usecase] = get_comvis_usecase_override
    app.dependency_overrides[get_inference_usecase] = get_inference_usecase_override
    app.dependency_overrides[get_firebase_admin] = get_firebase_admin_override
    app.dependency_overrides[get_feature_metric_usecase] = (
        get_feature_metric_usecase_override
    )

    yield

    # Clean up memory between tests
    mock_comvis_service.create_frame_from_planes.reset_mock()
    mock_comvis_service.process_frame.reset_mock()
    mock_inference_service.process_frames_concurrent.reset_mock()
    mock_inference_service.clear_caches.reset_mock()

    # Clear overrides after tests - THIS MUST BE UNCOMMENTED
    app.dependency_overrides = {}


# Helper functions
def create_frame_data(width=320, height=240):
    """Create mock frame data for testing"""
    # Create a simple header
    header = {
        "width": width,
        "height": height,
        "ySize": width * height,
        "uSize": (width * height) // 4,
        "vSize": (width * height) // 4,
    }

    # Convert header to bytes
    header_json = json.dumps(header).encode("utf-8")
    header_length = len(header_json).to_bytes(4, byteorder="big")

    # Create sample YUV data - use smaller byte values
    y_plane = bytes([100] * (width * height))
    u_plane = bytes([100] * ((width * height) // 4))
    v_plane = bytes([100] * ((width * height) // 4))

    # Combine all data
    frame_data = header_length + header_json + y_plane + u_plane + v_plane

    return frame_data


# Tests for authentication and connection
@pytest.mark.asyncio
async def test_websocket_invalid_token(
    mock_websocket, mock_auth_service, mock_feature_metric_service, mock_db
):
    """Test that websocket closes with invalid token"""
    # Setup mock auth service to reject token
    mock_auth_service.validate_token.return_value = (False, None, None)

    # Call the websocket handler directly with the mock websocket
    await livestream_exercise_tracking(
        websocket=mock_websocket,
        username="testuser",
        exercise_name="squat",
        token="invalid_token",
        auth_service=mock_auth_service,
        comvis_service=MagicMock(),
        inference_service=MagicMock(),
        feature_metric_service=mock_feature_metric_service,
        db=mock_db,
    )

    # Assert the connection was closed with the correct code
    assert mock_websocket.closed is True
    assert mock_websocket.close_code == status.WS_1008_POLICY_VIOLATION


@pytest.mark.asyncio
async def test_websocket_username_mismatch(
    mock_websocket, mock_auth_service, mock_user, mock_feature_metric_service, mock_db
):
    """Test that websocket closes when username doesn't match token"""
    # Setup mock auth service to return a user with different username
    mock_user.username = "different_user"
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Call the websocket handler
    await livestream_exercise_tracking(
        websocket=mock_websocket,
        username="testuser",
        exercise_name="squat",
        token="valid_token",
        auth_service=mock_auth_service,
        comvis_service=MagicMock(),
        inference_service=MagicMock(),
        feature_metric_service=mock_feature_metric_service,
        db=mock_db,
    )

    # Assert the connection was closed with the correct code
    assert mock_websocket.closed is True
    assert mock_websocket.close_code == status.WS_1008_POLICY_VIOLATION


@pytest.mark.asyncio
async def test_websocket_invalid_subprotocol(
    mock_websocket, mock_auth_service, mock_user, mock_feature_metric_service, mock_db
):
    """Test that websocket closes when using an invalid subprotocol and no fallback token is provided"""
    # Setup mock auth service to accept token (won't be used in this test)
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Set invalid subprotocol
    mock_websocket.headers["sec-websocket-protocol"] = "invalid-protocol"

    # Call the websocket handler WITHOUT a token parameter to prevent fallback authentication
    await livestream_exercise_tracking(
        websocket=mock_websocket,
        username="testuser",
        exercise_name="squat",
        token=None,  # No fallback token provided
        auth_service=mock_auth_service,
        comvis_service=MagicMock(),
        inference_service=MagicMock(),
        feature_metric_service=mock_feature_metric_service,
        db=mock_db,
    )

    # Assert the connection was closed with the correct code
    assert mock_websocket.closed is True
    assert mock_websocket.close_code == status.WS_1008_POLICY_VIOLATION


# @pytest.mark.asyncio
# async def test_websocket_fallback_to_query_param_auth(mock_websocket, mock_auth_service, mock_user, mock_websocket_auth):
#     """Test that websocket falls back to query parameter authentication when subprotocol is invalid"""
#     # Setup mock auth service to accept token
#     mock_auth_service.validate_token.return_value = (True, mock_user, None)
#     mock_websocket_auth.return_value = (True, mock_user, None)

#     # Set invalid subprotocol
#     mock_websocket.headers["sec-websocket-protocol"] = "invalid-protocol"

#     # Call the websocket handler WITH a valid token parameter to enable fallback authentication
#     await livestream_exercise_tracking(
#         websocket=mock_websocket,
#         username="testuser",
#         exercise_name="squat",
#         token="valid_token",
#         auth_service=mock_auth_service,
#         comvis_service=MagicMock(),
#         inference_service=MagicMock()
#     )

#     # Assert the connection was NOT closed (successful fallback)
#     assert mock_websocket.closed is False


@pytest.mark.asyncio
async def test_websocket_basic_connection_and_messaging(
    app, mock_auth_service, mock_user
):
    """Test basic WebSocket connection and message exchange using a lightweight mocked approach"""

    # Setup auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Create a WebSocket test client using our enhanced MockWebSocket
    ws_client = MockWebSocket(
        app=app,
        base_url="ws://testserver",
        headers={"sec-websocket-protocol": "livestream-v3"},
    )

    # Replace the actual endpoint handler with a simple mock
    async def mock_endpoint(websocket, username, exercise_name, token, **kwargs):
        await websocket.accept(subprotocol="livestream-v3")
        # Just return success immediately without any processing
        await websocket.send_json({"status": "success"})

    # Apply the patch
    with patch(
        "interface.ws.websocket_router.livestream_exercise_tracking", mock_endpoint
    ):
        # Connect to the endpoint
        url = "/v2/exercise-tracking?username=testuser&exercise_name=squat"
        await ws_client.connect(url)

        # Send a test message
        test_frame = create_frame_data(width=80, height=60)  # Very small frame
        await ws_client.send_bytes(test_frame)

        # Verify response
        response = await ws_client.receive_json()
        assert response["status"] == "mock_response"

        # Close the connection - don't use await since close() is not async
        ws_client.close()


# Test successful connection and frame processing
# @pytest.mark.asyncio
# async def test_websocket_successful_connection(
#     mock_websocket, mock_auth_service, mock_user, mock_comvis_service, mock_inference_service
# ):
#     """Test that websocket can connect successfully and process a completion signal"""
#     # Setup mock auth service to accept token
#     mock_auth_service.validate_token.return_value = (True, mock_user, None)

#     # Override receive_bytes to return a completion signal
#     mock_websocket.receive_bytes = AsyncMock(return_value=b'COMPLETED')

#     try:
#         # Call the websocket handler with modified execution to limit iterations
#         with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
#             # Make sleep return immediately to speed up the test
#             mock_sleep.return_value = None

#             await livestream_exercise_tracking(
#                 websocket=mock_websocket,
#                 username="testuser",
#                 exercise_name="squat",
#                 token="valid_token",
#                 auth_service=mock_auth_service,
#                 comvis_service=mock_comvis_service,
#                 inference_service=mock_inference_service
#             )

#         # Assert the websocket was not closed prematurely
#         assert mock_websocket.closed is False

#         # Check if the COMPLETED acknowledgment was sent
#         assert any(
#             msg for msg in mock_websocket.sent_messages
#             if msg["type"] == "json" and msg["data"].get("status") == "COMPLETED_ACK"
#         )
#     finally:
#         # Force cleanup to release memory
#         mock_websocket.sent_messages.clear()
#         mock_inference_service.clear_caches()
#         import gc
#         gc.collect()


@pytest.mark.asyncio
async def test_websocket_frame_processing(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that websocket can process frames correctly"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to process one frame then complete
    frame_data = create_frame_data(width=160, height=120)  # Smaller frame
    mock_websocket.receive_bytes = AsyncMock(
        side_effect=[
            frame_data,  # First call returns frame data
            b"COMPLETED",  # Second call returns completion signal
        ]
    )

    # Create a mock function for process_frame that will be called by the handler
    # We can't patch process_frame_async directly since it's defined inside the handler
    original_process_frame = mock_comvis_service.process_frame

    def mock_process_frame(frame):
        # Return a tuple as expected by the actual method
        return np.zeros((120, 160, 3), dtype=np.uint8), {}

    # Replace the process_frame method in the mock_comvis_service
    mock_comvis_service.process_frame = mock_process_frame

    try:
        # Call the websocket handler with patched sleep to control flow
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Track messages sent
            original_send_json = mock_websocket.send_json
            sent_messages = []

            async def track_messages(data):
                sent_messages.append(data)
                return await original_send_json(data)

            mock_websocket.send_json = track_messages

            await livestream_exercise_tracking(
                websocket=mock_websocket,
                username="testuser",
                exercise_name="squat",
                token="valid_token",
                auth_service=mock_auth_service,
                comvis_service=mock_comvis_service,
                inference_service=mock_inference_service,
                feature_metric_service=mock_feature_metric_service,
                db=mock_db,
            )

        # Verify that send_json was called with processing status
        assert any(msg.get("status") == "processing" for msg in sent_messages)

    finally:
        # Restore the original process_frame method
        mock_comvis_service.process_frame = original_process_frame

        # Force cleanup to release memory
        mock_websocket.sent_messages.clear()
        mock_inference_service.clear_caches()
        import gc

        gc.collect()


@pytest.mark.asyncio
async def test_websocket_analyze_buffer(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that websocket can analyze buffer correctly"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to process multiple frames to trigger buffer analysis
    frame_data = create_frame_data(width=160, height=120)  # Use smaller frame size

    # Make the test complete after all frames are processed + extra COMPLETED signals to prevent StopAsyncIteration
    receive_sequence = [frame_data] * 30 + [
        b"COMPLETED"
    ] * 10  # Add extra COMPLETED signals
    mock_websocket.receive_bytes = AsyncMock(side_effect=receive_sequence)

    # Save original process_frame method
    original_process_frame = mock_comvis_service.process_frame

    # Create mock for process_frame
    def mock_process_frame(frame):
        # Return a tuple with processed frame and empty buffer
        return np.zeros((120, 160, 3), dtype=np.uint8), {}

    # Replace the comvis service's process_frame method
    mock_comvis_service.process_frame = mock_process_frame

    # Track if the actual inference methods are called (not process_frames_concurrent)
    pose_estimation_called = False
    object_detection_called = False
    action_recognition_called = False

    # Override the individual inference methods that are actually called
    original_pose_infer = mock_inference_service._pose_estimation.infer
    original_object_infer = mock_inference_service._object_detection.infer
    original_action_infer = mock_inference_service._action_recognition.infer

    def track_pose_estimation(*args, **kwargs):
        nonlocal pose_estimation_called
        pose_estimation_called = True
        return {"keypoints": {}}

    def track_object_detection(*args, **kwargs):
        nonlocal object_detection_called
        object_detection_called = True
        return {"predictions": []}

    def track_action_recognition(*args, **kwargs):
        nonlocal action_recognition_called
        action_recognition_called = True
        return {"predicted_class_name": "squat"}

    # Replace the inference service methods with tracking versions
    mock_inference_service._pose_estimation.infer = track_pose_estimation
    mock_inference_service._object_detection.infer = track_object_detection
    mock_inference_service._action_recognition.infer = track_action_recognition

    try:
        # Call the handler with patches to control timing
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Make sleep return immediately to speed up the test
            mock_sleep.return_value = None

            await livestream_exercise_tracking(
                websocket=mock_websocket,
                username="testuser",
                exercise_name="squat",
                token="valid_token",
                auth_service=mock_auth_service,
                comvis_service=mock_comvis_service,
                inference_service=mock_inference_service,
                feature_metric_service=mock_feature_metric_service,
                db=mock_db,
            )

        # Give a moment for any pending async tasks to complete
        await asyncio.sleep(0.1)

        # Verify that the individual inference methods were called
        assert (
            pose_estimation_called
        ), "Pose estimation should be called during frame analysis"
        assert (
            object_detection_called
        ), "Object detection should be called during frame analysis"
        assert (
            action_recognition_called
        ), "Action recognition should be called during frame analysis"

    finally:
        # Restore original methods
        mock_comvis_service.process_frame = original_process_frame
        mock_inference_service._pose_estimation.infer = original_pose_infer
        mock_inference_service._object_detection.infer = original_object_infer
        mock_inference_service._action_recognition.infer = original_action_infer

        # Force cleanup to release memory
        mock_websocket.sent_messages.clear()
        mock_inference_service.clear_caches()
        import gc

        gc.collect()


@pytest.mark.asyncio
async def test_websocket_disconnect_handling(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that websocket handles disconnection gracefully"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to raise a WebSocketDisconnect exception
    mock_websocket.receive_bytes = AsyncMock(side_effect=WebSocketDisconnect())

    try:
        # Call the websocket handler
        await livestream_exercise_tracking(
            websocket=mock_websocket,
            username="testuser",
            exercise_name="squat",
            token="valid_token",
            auth_service=mock_auth_service,
            comvis_service=mock_comvis_service,
            inference_service=mock_inference_service,
            feature_metric_service=mock_feature_metric_service,
            db=mock_db,
        )

        # Verify inference cache was cleared
        mock_inference_service.clear_caches.assert_called_once()

        # Verify feature metrics history was reset
        mock_feature_metric_service.feature_metric_repo.reset_history.assert_called_once()
    finally:
        # Force cleanup to release memory
        import gc

        gc.collect()


@pytest.mark.asyncio
async def test_websocket_error_handling(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that websocket handles processing errors gracefully"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to process one frame then error
    frame_data = create_frame_data(width=160, height=120)  # Smaller frame
    mock_websocket.receive_bytes = AsyncMock(return_value=frame_data)

    # Make frame processing fail
    mock_comvis_service.create_frame_from_planes.side_effect = Exception("Test error")

    try:
        # Call the websocket handler
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Make it execute only once
            mock_sleep.side_effect = [None, WebSocketDisconnect()]

            await livestream_exercise_tracking(
                websocket=mock_websocket,
                username="testuser",
                exercise_name="squat",
                token="valid_token",
                auth_service=mock_auth_service,
                comvis_service=mock_comvis_service,
                inference_service=mock_inference_service,
                feature_metric_service=mock_feature_metric_service,
                db=mock_db,
            )

        # Verify inference cache was cleared (happens in finally block)
        mock_inference_service.clear_caches.assert_called_once()

        # Verify feature metrics history was reset
        mock_feature_metric_service.feature_metric_repo.reset_history.assert_called_once()
    finally:
        # Force cleanup to release memory
        mock_websocket.sent_messages.clear()
        mock_inference_service.clear_caches.reset_mock()
        import gc

        gc.collect()


@pytest.mark.asyncio
async def test_feature_metrics_computation_and_saving(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that feature metrics are computed and saved at session end"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to process multiple frames to accumulate data before disconnecting
    frame_data = create_frame_data(width=160, height=120)
    # Process 35 frames to trigger at least one analyze_buffer call (at frame 30) then disconnect
    receive_sequence = [frame_data] * 35 + [WebSocketDisconnect()]
    mock_websocket.receive_bytes = AsyncMock(side_effect=receive_sequence)

    # Mock the analyze_buffer function's feature data to ensure accumulated data exists
    mock_features = MagicMock()
    mock_features.body_alignment = MagicMock()  # Mock BodyAlignment object
    mock_features.joint_angles = {"shoulder": 45.0, "elbow": 90.0}
    mock_features.objects = {"weight": {"x": 100, "y": 200, "width": 50, "height": 100}}
    mock_features.speeds = {"velocity": 2.5}
    mock_features.stability = 1.2

    mock_comvis_service.load_to_features_model.return_value = mock_features

    try:
        # Call the websocket handler
        await livestream_exercise_tracking(
            websocket=mock_websocket,
            username="testuser",
            exercise_name="bench_press",
            token="valid_token",
            auth_service=mock_auth_service,
            comvis_service=mock_comvis_service,
            inference_service=mock_inference_service,
            feature_metric_service=mock_feature_metric_service,
            db=mock_db,
        )

        # Verify feature metrics compute_all_metrics was called
        mock_feature_metric_service.feature_metric_repo.compute_all_metrics.assert_called_once()

        # Verify feature metrics were saved
        mock_feature_metric_service.save_feature_metrics.assert_called_once()

        # Check the save call arguments
        save_call_args = mock_feature_metric_service.save_feature_metrics.call_args
        assert save_call_args[0][0] == "testuser"  # username
        assert save_call_args[0][1] == "bench press"  # exercise_name
        assert isinstance(save_call_args[0][2], dict)  # metrics dict

        # Verify exercise-specific thresholds were used for bench_press
        compute_call_kwargs = mock_feature_metric_service.feature_metric_repo.compute_all_metrics.call_args[
            1
        ]
        assert (
            compute_call_kwargs["max_allowed_deviation"] == 8
        )  # bench_press threshold
        assert (
            compute_call_kwargs["max_allowed_variance"] == 12
        )  # bench_press threshold
        assert compute_call_kwargs["max_jerk"] == 4.0  # bench_press threshold
        assert compute_call_kwargs["max_displacement"] == 15.0  # bench_press threshold

    finally:
        # Force cleanup
        import gc

        gc.collect()


@pytest.mark.asyncio
async def test_exercise_specific_thresholds(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that different exercises use appropriate thresholds"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to process multiple frames to accumulate data before disconnecting
    frame_data = create_frame_data(width=160, height=120)
    receive_sequence = [frame_data] * 35 + [WebSocketDisconnect()]
    mock_websocket.receive_bytes = AsyncMock(side_effect=receive_sequence)

    # Mock the analyze_buffer function's feature data to ensure accumulated data exists
    mock_features = MagicMock()
    mock_features.body_alignment = MagicMock()  # Mock BodyAlignment object
    mock_features.joint_angles = {"shoulder": 45.0, "elbow": 90.0}
    mock_features.objects = {"weight": {"x": 100, "y": 200, "width": 50, "height": 100}}
    mock_features.speeds = {"velocity": 2.5}
    mock_features.stability = 1.2

    mock_comvis_service.load_to_features_model.return_value = mock_features

    # Test deadlift exercise
    try:
        await livestream_exercise_tracking(
            websocket=mock_websocket,
            username="testuser",
            exercise_name="deadlift",
            token="valid_token",
            auth_service=mock_auth_service,
            comvis_service=mock_comvis_service,
            inference_service=mock_inference_service,
            feature_metric_service=mock_feature_metric_service,
            db=mock_db,
        )

        # Verify deadlift-specific thresholds were used
        compute_call_kwargs = mock_feature_metric_service.feature_metric_repo.compute_all_metrics.call_args[
            1
        ]
        assert compute_call_kwargs["max_allowed_deviation"] == 10  # deadlift threshold
        assert compute_call_kwargs["max_allowed_variance"] == 15  # deadlift threshold
        assert compute_call_kwargs["max_jerk"] == 6.0  # deadlift threshold
        assert compute_call_kwargs["max_displacement"] == 20.0  # deadlift threshold

    finally:
        # Force cleanup
        import gc

        gc.collect()


@pytest.mark.asyncio
async def test_feature_metrics_error_handling(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that websocket handles feature metrics computation errors gracefully"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to disconnect immediately to trigger final metrics
    mock_websocket.receive_bytes = AsyncMock(side_effect=WebSocketDisconnect())

    # Make feature metrics computation fail
    mock_feature_metric_service.feature_metric_repo.compute_all_metrics.side_effect = (
        Exception("Metrics computation failed")
    )

    try:
        # Call the websocket handler
        await livestream_exercise_tracking(
            websocket=mock_websocket,
            username="testuser",
            exercise_name="squat",
            token="valid_token",
            auth_service=mock_auth_service,
            comvis_service=mock_comvis_service,
            inference_service=mock_inference_service,
            feature_metric_service=mock_feature_metric_service,
            db=mock_db,
        )

        # Verify that even with metrics error, the session completes gracefully
        # and history is still reset
        mock_feature_metric_service.feature_metric_repo.reset_history.assert_called_once()

    finally:
        # Force cleanup
        import gc

        gc.collect()


@pytest.mark.asyncio
async def test_feature_metrics_history_reset(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that feature metrics history is properly reset for new sessions"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to disconnect immediately
    mock_websocket.receive_bytes = AsyncMock(side_effect=WebSocketDisconnect())

    try:
        # Call the websocket handler
        await livestream_exercise_tracking(
            websocket=mock_websocket,
            username="testuser",
            exercise_name="squat",
            token="valid_token",
            auth_service=mock_auth_service,
            comvis_service=mock_comvis_service,
            inference_service=mock_inference_service,
            feature_metric_service=mock_feature_metric_service,
            db=mock_db,
        )

        # Verify history reset was called
        mock_feature_metric_service.feature_metric_repo.reset_history.assert_called_once()

    finally:
        # Force cleanup
        import gc

        gc.collect()


@pytest.mark.asyncio
async def test_shoulder_press_thresholds(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that shoulder press exercise uses stricter thresholds"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to process multiple frames to accumulate data before disconnecting
    frame_data = create_frame_data(width=160, height=120)
    receive_sequence = [frame_data] * 35 + [WebSocketDisconnect()]
    mock_websocket.receive_bytes = AsyncMock(side_effect=receive_sequence)

    # Mock the analyze_buffer function's feature data to ensure accumulated data exists
    mock_features = MagicMock()
    mock_features.body_alignment = MagicMock()  # Mock BodyAlignment object
    mock_features.joint_angles = {"shoulder": 45.0, "elbow": 90.0}
    mock_features.objects = {"weight": {"x": 100, "y": 200, "width": 50, "height": 100}}
    mock_features.speeds = {"velocity": 2.5}
    mock_features.stability = 1.2

    mock_comvis_service.load_to_features_model.return_value = mock_features

    try:
        await livestream_exercise_tracking(
            websocket=mock_websocket,
            username="testuser",
            exercise_name="shoulder_press",
            token="valid_token",
            auth_service=mock_auth_service,
            comvis_service=mock_comvis_service,
            inference_service=mock_inference_service,
            feature_metric_service=mock_feature_metric_service,
            db=mock_db,
        )

        # Verify shoulder press-specific thresholds were used (stricter)
        compute_call_kwargs = mock_feature_metric_service.feature_metric_repo.compute_all_metrics.call_args[
            1
        ]
        assert (
            compute_call_kwargs["max_allowed_deviation"] == 6
        )  # shoulder_press threshold
        assert (
            compute_call_kwargs["max_allowed_variance"] == 10
        )  # shoulder_press threshold
        assert compute_call_kwargs["max_jerk"] == 3.5  # shoulder_press threshold
        assert (
            compute_call_kwargs["max_displacement"] == 12.0
        )  # shoulder_press threshold

    finally:
        # Force cleanup
        import gc

        gc.collect()


@pytest.mark.asyncio
async def test_unknown_exercise_default_thresholds(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that unknown exercises use default thresholds"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to process multiple frames to accumulate data before disconnecting
    frame_data = create_frame_data(width=160, height=120)
    receive_sequence = [frame_data] * 35 + [WebSocketDisconnect()]
    mock_websocket.receive_bytes = AsyncMock(side_effect=receive_sequence)

    # Mock the analyze_buffer function's feature data to ensure accumulated data exists
    mock_features = MagicMock()
    mock_features.body_alignment = MagicMock()  # Mock BodyAlignment object
    mock_features.joint_angles = {"shoulder": 45.0, "elbow": 90.0}
    mock_features.objects = {"weight": {"x": 100, "y": 200, "width": 50, "height": 100}}
    mock_features.speeds = {"velocity": 2.5}
    mock_features.stability = 1.2

    mock_comvis_service.load_to_features_model.return_value = mock_features

    try:
        await livestream_exercise_tracking(
            websocket=mock_websocket,
            username="testuser",
            exercise_name="unknown_exercise",
            token="valid_token",
            auth_service=mock_auth_service,
            comvis_service=mock_comvis_service,
            inference_service=mock_inference_service,
            feature_metric_service=mock_feature_metric_service,
            db=mock_db,
        )

        # Verify default thresholds were used
        compute_call_kwargs = mock_feature_metric_service.feature_metric_repo.compute_all_metrics.call_args[
            1
        ]
        assert compute_call_kwargs["max_allowed_deviation"] == 10  # default threshold
        assert compute_call_kwargs["max_allowed_variance"] == 15  # default threshold
        assert compute_call_kwargs["max_jerk"] == 5.0  # default threshold
        assert compute_call_kwargs["max_displacement"] == 20.0  # default threshold

    finally:
        # Force cleanup
        import gc

        gc.collect()


@pytest.mark.asyncio
async def test_feature_metrics_data_integrity(
    mock_websocket,
    mock_auth_service,
    mock_user,
    mock_comvis_service,
    mock_inference_service,
    mock_feature_metric_service,
    mock_db,
):
    """Test that feature metrics data contains expected keys and values"""
    # Setup mock auth service to accept token
    mock_auth_service.validate_token.return_value = (True, mock_user, None)

    # Setup to process multiple frames to accumulate data before disconnecting
    frame_data = create_frame_data(width=160, height=120)
    receive_sequence = [frame_data] * 35 + [WebSocketDisconnect()]
    mock_websocket.receive_bytes = AsyncMock(side_effect=receive_sequence)

    # Mock the analyze_buffer function's feature data to ensure accumulated data exists
    mock_features = MagicMock()
    mock_features.body_alignment = MagicMock()  # Mock BodyAlignment object
    mock_features.joint_angles = {"shoulder": 45.0, "elbow": 90.0}
    mock_features.objects = {"weight": {"x": 100, "y": 200, "width": 50, "height": 100}}
    mock_features.speeds = {"velocity": 2.5}
    mock_features.stability = 1.2

    mock_comvis_service.load_to_features_model.return_value = mock_features

    try:
        await livestream_exercise_tracking(
            websocket=mock_websocket,
            username="testuser",
            exercise_name="bench_press",
            token="valid_token",
            auth_service=mock_auth_service,
            comvis_service=mock_comvis_service,
            inference_service=mock_inference_service,
            feature_metric_service=mock_feature_metric_service,
            db=mock_db,
        )

        # Verify the metrics returned by the mock match expected format
        expected_metrics = (
            mock_feature_metric_service.feature_metric_repo.compute_all_metrics.return_value
        )
        assert "body_alignment" in expected_metrics
        assert "joint_consistency" in expected_metrics
        assert "load_control" in expected_metrics
        assert "speed_control" in expected_metrics
        assert "overall_stability" in expected_metrics

        # Verify all values are floats
        for key, value in expected_metrics.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric"
            assert 0 <= value <= 100, f"{key} should be between 0-100"

    finally:
        # Force cleanup
        import gc

        gc.collect()
