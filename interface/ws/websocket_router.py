import json
import pathlib
from datetime import datetime

import jsonschema
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.encoders import jsonable_encoder

from .constant import EXERCISE_THRESHOLDS

from core.interface import NTFInterface
from core.usecase import AuthUseCase, ComVisUseCase, FeatureMetricUseCase

from infrastructure.di import get_firebase_admin

from interface.di import get_auth_service, get_comvis_usecase
from interface.di.comvis_service_di import get_feature_metric_usecase
from interface.ws.live_protocol import build_bands_message, build_coaching_message
from interface.ws.websocket_auth import (
    authenticate_websocket_query_param,
    authenticate_websocket_subprotocol,
    close_websocket_with_auth_error,
)

from lifttrack.utils.logging_config import setup_logger

logger = setup_logger("interface.ws.router", "websocket.log")
websocket_router_v3 = APIRouter(prefix="/v2", tags=["v2-websocket"])

# The client streams keypoints (ADR 0001); the server validates each frame
# against the shared wire contract before scoring it.
_KEYPOINTS_SCHEMA = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[2]
        / "contract"
        / "schemas"
        / "keypoints.schema.json"
    ).read_text(encoding="utf-8")
)

# Coaching + persistence run once per this many keypoint frames (~1/sec at 15fps),
# sharing a single scoring pass (ADR 0011). Bands are sent on every frame.
COACHING_EVERY = 15
MAX_FRAMES = 27000  # ~30 min at 15fps


@websocket_router_v3.websocket("/exercise-tracking")
async def livestream_exercise_tracking(
    websocket: WebSocket,
    username: str = Query(...),
    exercise_name: str = Query(...),
    token: str = Query(
        None, description="JWT authentication token (alternative to subprotocol auth)"
    ),
    auth_service: AuthUseCase = Depends(get_auth_service),
    comvis_service: ComVisUseCase = Depends(get_comvis_usecase),
    feature_metric_service: FeatureMetricUseCase = Depends(get_feature_metric_usecase),
    db: NTFInterface = Depends(get_firebase_admin),
):
    """WebSocket endpoint for real-time exercise tracking (keypoints-in / bands-out).

    The client streams ``keypoints`` messages (17 COCO, normalized [0,1]); the
    server returns ``bands`` per frame and a ~1/sec ``coaching`` message, and
    persists scores to Firebase in the same 1/sec pass (ADR 0011). No pixels and
    no positions cross the wire.

    Authentication (in order of preference):
    1. Sec-WebSocket-Protocol header: ["<jwt>", "livestream-v3"]
    2. Query parameter: ?token=<jwt>
    """
    auth_success, client, error_msg = await authenticate_websocket_subprotocol(
        websocket, username, auth_service, "livestream-v3"
    )
    if not auth_success and token:
        logger.info("Subprotocol auth failed, trying query parameter auth")
        auth_success, client, error_msg = await authenticate_websocket_query_param(
            websocket, token, username, auth_service
        )
    if not auth_success:
        await close_websocket_with_auth_error(
            websocket, error_msg or "Authentication failed"
        )
        return

    await websocket.accept(subprotocol="livestream-v3")

    exercise_name = exercise_name.lower().replace(" ", "_")
    current_thresholds = EXERCISE_THRESHOLDS.get(
        exercise_name, EXERCISE_THRESHOLDS["default"]
    )

    frame_count = 0
    connection_active = True

    # Latest per-frame signals, used for the periodic + final metric passes.
    last_body_alignment = None
    last_joint_angles: dict = {}
    last_speeds: dict = {}
    last_stability = 0.0

    def _connected() -> bool:
        return connection_active and websocket.client_state.name == "CONNECTED"

    try:
        while connection_active and frame_count < MAX_FRAMES:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type == "complete":
                await websocket.send_json({"type": "complete_ack"})
                break
            if msg_type != "keypoints":
                await websocket.send_json(
                    {"type": "error", "error": "expected a 'keypoints' message"}
                )
                continue

            try:
                jsonschema.validate(message, _KEYPOINTS_SCHEMA)
            except jsonschema.ValidationError as ve:
                await websocket.send_json(
                    {"type": "error", "error": f"invalid keypoints: {ve.message}"}
                )
                continue

            t = message["t"]
            keypoints_dict = {
                name: (kp["x"], kp["y"], kp["confidence"])
                for name, kp in message["keypoints"].items()
            }

            # Scoring is pure Python (no ML) and fast; run it inline.
            features = comvis_service.process_keypoints(keypoints_dict)
            joint_angles = features.joint_angles or {}
            frame_count += 1

            # Bands lane: every frame.
            if _connected():
                try:
                    await websocket.send_json(
                        build_bands_message(joint_angles, exercise_name, t)
                    )
                except RuntimeError as e:
                    logger.warning(f"Could not send bands: {e}")
                    connection_active = False
                    break

            # Track latest signals for the metric passes.
            if joint_angles:
                last_joint_angles = joint_angles
            if features.body_alignment:
                last_body_alignment = features.body_alignment
            if features.speeds:
                last_speeds = features.speeds
            if features.stability is not None:
                last_stability = features.stability

            # Coaching + persistence lane: ~1/sec.
            if frame_count % COACHING_EVERY == 0:
                coaching = build_coaching_message(joint_angles, exercise_name)
                if _connected():
                    try:
                        await websocket.send_json(coaching)
                    except RuntimeError as e:
                        logger.warning(f"Could not send coaching: {e}")

                try:
                    if last_body_alignment:
                        feature_metric_service.compute_body_alignment(
                            last_body_alignment,
                            current_thresholds["max_allowed_deviation"],
                        )
                        feature_metric_service.compute_joint_consistency(
                            last_joint_angles,
                            current_thresholds["max_allowed_variance"],
                        )
                        feature_metric_service.compute_speed_control(
                            last_speeds, current_thresholds["max_jerk"]
                        )
                        feature_metric_service.compute_overall_stability(
                            last_stability, current_thresholds["max_displacement"]
                        )
                except Exception as metrics_error:
                    logger.error(f"Error updating metric history: {metrics_error}")

                try:
                    second_number = frame_count // COACHING_EVERY
                    suggestions_text = (
                        " ".join(coaching["suggestions"])
                        if coaching["suggestions"]
                        else "Form looks good! Keep it up!"
                    )
                    exercise_data_model = comvis_service.load_exercise_data(
                        frame_index=str(frame_count),
                        features=features,
                        suggestions=suggestions_text,
                        frame_id=f"frame_{frame_count}",
                    )
                    await comvis_service.save_exercise_data(
                        username=username,
                        exercise_name=exercise_name.replace("_", " "),
                        date=comvis_service.format_date(datetime.now().isoformat()),
                        time_frame=f"second_{second_number}",
                        exercise_data=jsonable_encoder(exercise_data_model),
                        db=db,
                    )
                except Exception as save_error:
                    logger.error(f"Failed to save analysis: {save_error}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected after {frame_count} keypoint frames")
        connection_active = False
    except Exception as e:
        logger.error(f"Error in websocket handler: {e}", exc_info=True)
        connection_active = False
    finally:
        connection_active = False

        # Persist the session's final feature metrics from the last real frame.
        try:
            if last_body_alignment:
                final_metrics = (
                    feature_metric_service.feature_metric_repo.compute_all_metrics(
                        body_alignment=last_body_alignment,
                        joint_angles=last_joint_angles,
                        objects={},
                        speeds=last_speeds,
                        stability_raw=last_stability,
                        max_allowed_deviation=current_thresholds[
                            "max_allowed_deviation"
                        ],
                        max_allowed_variance=current_thresholds["max_allowed_variance"],
                        max_jerk=current_thresholds["max_jerk"],
                        max_displacement=current_thresholds["max_displacement"],
                    )
                )
                await feature_metric_service.save_feature_metrics(
                    username, exercise_name.replace("_", " "), final_metrics
                )
                logger.info(f"Final feature metrics saved for {username}")
            else:
                logger.warning("No frames scored; skipping final metrics")
        except Exception as e:
            logger.error(f"Error saving final feature metrics: {e}")

        try:
            feature_metric_service.feature_metric_repo.reset_history()
        except Exception as e:
            logger.error(f"Error resetting feature metrics history: {e}")
