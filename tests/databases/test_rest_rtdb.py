from datetime import datetime, timezone
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.entities import User as UserEntity
from core.entities import (
    Exercise, 
    ExerciseFeature, 
    Object, 
    JointAngles, 
    BodyAlignment, 
    ExerciseType
)

from lifttrack.dbhandler.rest_rtdb import RTDBHelper


@pytest.fixture
def mock_db():
    """Create a mock database instance"""
    db = MagicMock()
    # Create AsyncMock instances for async operations
    db.put = AsyncMock()
    db.get = AsyncMock()
    db.post = AsyncMock()
    db.delete = AsyncMock()
    return db

@pytest.fixture
def rtdb_handler(mock_db):
    """Create a RTDBHelper instance with mocked connection"""
    handler = RTDBHelper()
    handler._get_connection = MagicMock(return_value=mock_db)
    return handler

@pytest.fixture
def mock_user_dict_instance():
    """Create valid user data fixture"""
    return {
        "username": "testuser",
        "email": "testuser@mail.com",
        "password": "testpassword",
        "first_name": "Test",
        "last_name": "User",
        "phone_number": "09123456789"
    }
    
@pytest.fixture
def mock_user_entity_instance():
    """Create valid user entity fixture"""
    return UserEntity(
        username="testuser",
        email="testuser@mail.com",
        password="testpassword",
        first_name="Test",
        last_name="User",
        phone_number="09123456789"
    ).to_dict()
    
@pytest.fixture
def mock_exercise_dict_instance():
    """Create valid progress data fixture"""
    return {
        "testuser": {
            "rdl": {
                "2024-12-29T12-34-14": {
                    "second_1": {
                    "date": "2024-12-29T12:34:14.133761",
                    "features": {
                        "body_alignment": [
                        12.745002523141531,
                        162.52817313759746
                        ],
                        "joint_angles": {
                        "left_hip_left_knee_left_ankle": 177.72239987686729,
                        "left_shoulder_left_elbow_left_wrist": 160.67208312283574,
                        "left_shoulder_left_hip_left_knee": 83.76644909556005,
                        "right_hip_right_knee_right_ankle": 152.74201614481927,
                        "right_shoulder_right_elbow_right_wrist": 169.80205586571006,
                        "right_shoulder_right_hip_right_knee": 72.61091964478223
                        },
                        "movement_pattern": "romanian_deadlift",
                        "objects": {
                        "classs_id": 0,
                        "confidence": 0.8848945498466492,
                        "height": 48,
                        "type": "barbell",
                        "width": 63,
                        "x": 137.5,
                        "y": 153
                        },
                        "speeds": {
                        "left_ankle": 1,
                        "left_ear": 3.605551275463989,
                        "left_elbow": 1,
                        "left_eye": 3.605551275463989,
                        "left_hip": 2,
                        "left_knee": 1.4142135623730951,
                        "left_shoulder": 4.123105625617661,
                        "left_wrist": 4,
                        "nose": 3.1622776601683795,
                        "right_ankle": 0,
                        "right_ear": 2.23606797749979,
                        "right_elbow": 2,
                        "right_eye": 2.23606797749979,
                        "right_hip": 1,
                        "right_knee": 3.605551275463989,
                        "right_shoulder": 0,
                        "right_wrist": 9.486832980505138
                        },
                        "stability": 44.47521961005582
                    },
                    "frame": "frame_30",
                    "suggestion": "No suggestions"
                    }
                }
            }
        }
    }
    
@pytest.fixture
def mock_exercise_entity_instance():
    """Create valid exercise entity fixture"""
    return Exercise(
        username="testuser",
        exercise_type=ExerciseType.RDL,
        date_performed=datetime.now(timezone.utc).strftime("%d-%m-%Y"),
        time_frame=datetime.now(timezone.utc).strftime("%H:%M:%S"),
        suggestion=["No suggestions"],
        features=ExerciseFeature(
            body_alignment=BodyAlignment(
                vertical_alignment=12.745002523141531,
                horizontal_alignment=162.52817313759746
            ),
            joint_angles=JointAngles(
                left_sew=177.72239987686729,
                left_hka=160.67208312283574,
                left_shk=83.76644909556005,
                right_sew=152.74201614481927,
                right_hka=169.80205586571006,
                right_shk=72.61091964478223
            ),
            movement_pattern="romanian_deadlift",
            objects=Object(
                object_type="barbell",
                object_height=48,
                object_width=63,
                object_x_coordinate=137.5,
                object_y_coordinate=153,
            ),
            movement_speed=1,
            stability=44.47521961005582
        )
    ).to_dict()

@pytest.mark.asyncio
async def test_connection_error(rtdb_handler):
    """Test handling of connection errors"""
    rtdb_handler._get_connection = MagicMock(side_effect=Exception("Connection failed"))
    
    with pytest.raises(Exception) as exc_info:
        await rtdb_handler.get("test/path", "test_key")
    assert str(exc_info.value) == "Connection failed"

@pytest.mark.asyncio
async def test_set_with_dict(rtdb_handler, mock_db, mock_user_dict_instance):
    """Test set operation"""
    rtdb_handler._run_in_executor = AsyncMock(return_value="test_key")
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.set("test/path", mock_user_dict_instance, "test_key")
    assert result == "test_key"
    rtdb_handler._run_in_executor.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_set_with_entity(rtdb_handler, mock_db, mock_user_entity_instance):
    """Test set operation with entity"""
    rtdb_handler._run_in_executor = AsyncMock(return_value="test_key")
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.set("test/path", mock_user_entity_instance, "test_key")
    assert result == "test_key"
    rtdb_handler._run_in_executor.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_set_with_progress(rtdb_handler, mock_db, mock_exercise_dict_instance):
    """Test set operation with progress"""
    rtdb_handler._run_in_executor = AsyncMock(return_value="test_key")
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.set("test/path", mock_exercise_dict_instance, "test_key")
    assert result == "test_key"
    rtdb_handler._run_in_executor.assert_awaited_once()

@pytest.mark.asyncio
async def test_set_with_exercise_entity(rtdb_handler, mock_db, mock_exercise_entity_instance):
    """Test set operation with exercise entity"""
    rtdb_handler._run_in_executor = AsyncMock(return_value="test_key")
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.set("test/path", mock_exercise_entity_instance, "test_key")
    assert result == "test_key"
    rtdb_handler._run_in_executor.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_with_dict(rtdb_handler, mock_db, mock_user_dict_instance):
    """Test get operation"""
    rtdb_handler._run_in_executor = AsyncMock(return_value=mock_user_dict_instance)
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.get("test/path", "test_key")
    assert result == mock_user_dict_instance
    rtdb_handler._run_in_executor.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_with_entity(rtdb_handler, mock_db, mock_user_entity_instance):
    """Test get operation with entity"""
    rtdb_handler._run_in_executor = AsyncMock(return_value=mock_user_entity_instance)
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.get("test/path", "test_key")
    assert result == mock_user_entity_instance
    rtdb_handler._run_in_executor.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_get_with_progress(rtdb_handler, mock_db, mock_exercise_dict_instance):
    """Test get operation with progress"""
    rtdb_handler._run_in_executor = AsyncMock(return_value=mock_exercise_dict_instance)
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.get("test/path", "test_key")
    assert result == mock_exercise_dict_instance
    rtdb_handler._run_in_executor.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_get_with_exercise_entity(rtdb_handler, mock_db, mock_exercise_entity_instance):
    """Test get operation with exercise entity"""
    rtdb_handler._run_in_executor = AsyncMock(return_value=mock_exercise_entity_instance)
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.get("test/path", "test_key")
    assert result == mock_exercise_entity_instance
    rtdb_handler._run_in_executor.assert_awaited_once()

@pytest.mark.asyncio
async def test_query_with_dict(rtdb_handler, mock_db, mock_user_dict_instance):
    """Test query operation"""
    rtdb_handler._run_in_executor = AsyncMock(return_value=[mock_user_dict_instance])
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.query("test/path")
    assert result == [mock_user_dict_instance]
    rtdb_handler._run_in_executor.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_query_with_entity(rtdb_handler, mock_db, mock_user_entity_instance):
    """Test query operation with entity"""
    rtdb_handler._run_in_executor = AsyncMock(return_value=[mock_user_entity_instance])
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.query("test/path")
    assert result == [mock_user_entity_instance]
    rtdb_handler._run_in_executor.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_query_with_progress(rtdb_handler, mock_db, mock_exercise_dict_instance):
    """Test query operation with progress"""
    rtdb_handler._run_in_executor = AsyncMock(return_value=[mock_exercise_dict_instance])
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.query("test/path")
    assert result == [mock_exercise_dict_instance]
    rtdb_handler._run_in_executor.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_query_with_exercise_entity(rtdb_handler, mock_db, mock_exercise_entity_instance):
    """Test query operation with exercise entity"""
    rtdb_handler._run_in_executor = AsyncMock(return_value=[mock_exercise_entity_instance])
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.query("test/path")
    assert result == [mock_exercise_entity_instance]
    rtdb_handler._run_in_executor.assert_awaited_once()

@pytest.mark.asyncio
async def test_push_with_dict(rtdb_handler, mock_db, mock_user_dict_instance):
    """Test push operation"""
    rtdb_handler._run_in_executor = AsyncMock(return_value={"name": "new_key"})
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.push("test/path", mock_user_dict_instance)
    assert result == "new_key"
    rtdb_handler._run_in_executor.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_push_with_entity(rtdb_handler, mock_db, mock_user_entity_instance):
    """Test push operation with entity"""
    rtdb_handler._run_in_executor = AsyncMock(return_value={"name": "new_key"})
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.push("test/path", mock_user_entity_instance)
    assert result == "new_key"
    rtdb_handler._run_in_executor.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_push_with_progress(rtdb_handler, mock_db, mock_exercise_dict_instance):
    """Test push operation with progress"""
    rtdb_handler._run_in_executor = AsyncMock(return_value={"name": "new_key"})
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.push("test/path", mock_exercise_dict_instance)
    assert result == "new_key"
    rtdb_handler._run_in_executor.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_push_with_exercise_entity(rtdb_handler, mock_db, mock_exercise_entity_instance):
    """Test push operation with exercise entity"""
    rtdb_handler._run_in_executor = AsyncMock(return_value={"name": "new_key"})
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)
    
    result = await rtdb_handler.push("test/path", mock_exercise_entity_instance)
    assert result == "new_key"
    rtdb_handler._run_in_executor.assert_awaited_once()

@pytest.mark.asyncio
async def test_update_with_dict(rtdb_handler, mock_db, mock_user_dict_instance):
    """Test update operation"""
    get_result = mock_user_dict_instance.copy()
    # Mock the update operation to return None (Firebase's behavior)
    mock_db.child().child().update.return_value = None
    rtdb_handler._run_in_executor = AsyncMock(side_effect=[get_result, None])
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)

    result = await rtdb_handler.update("test/path", "test_key", {"email": "new@example.com"})
    assert result is True

@pytest.mark.asyncio
async def test_update_with_entity(rtdb_handler, mock_db, mock_user_entity_instance):
    """Test update operation with entity"""
    get_result = mock_user_entity_instance.copy()
    # Mock the update operation to return None
    mock_db.child().child().update.return_value = None
    rtdb_handler._run_in_executor = AsyncMock(side_effect=[get_result, None])
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)

    result = await rtdb_handler.update("test/path", "test_key", {"email": "new@example.com"})
    assert result is True

@pytest.mark.asyncio
async def test_delete_with_dict(rtdb_handler, mock_db):
    """Test delete operation"""
    # Mock the delete operation to return None
    mock_db.child().child().remove.return_value = None
    rtdb_handler._run_in_executor = AsyncMock(return_value=None)
    rtdb_handler._get_connection = MagicMock(return_value=mock_db)

    result = await rtdb_handler.delete("test/path", "test_key")
    assert result is True
