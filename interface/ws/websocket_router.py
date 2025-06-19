import asyncio
import concurrent.futures
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from functools import partial
from fastapi.encoders import jsonable_encoder
from typing import Any, List

from core.interface import NTFInterface
from core.usecase import AuthUseCase
from core.usecase.comvis_usecase import ComVisUseCase
from core.usecase import InferenceUseCase
from infrastructure.di import get_firebase_admin
from interface.di import get_auth_service, get_comvis_usecase
from interface.di.inference_service import get_inference_usecase
from interface.ws.websocket_auth import (
    authenticate_websocket_query_param,
    authenticate_websocket_subprotocol,
    close_websocket_with_auth_error,
)

from lifttrack.utils.logging_config import setup_logger

logger = setup_logger("interface.ws.router", "websocket.log")
websocket_router_v3 = APIRouter(prefix="/v2", tags=["v2-websocket"])

# Optimize thread pool for 4-core CPU: Use only 2 workers to prevent saturation
# This leaves room for the main event loop and other system processes
thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="ws-shared-pool"
)


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
    inference_service: InferenceUseCase = Depends(get_inference_usecase),
    db: NTFInterface = Depends(get_firebase_admin),
):
    """
    WebSocket endpoint for real-time exercise tracking with authentication.

    Authentication Methods (in order of preference):
    1. Sec-WebSocket-Protocol header: ["your_jwt_token_here", "livestream-v3"]
    2. Query parameter: ?token=your_jwt_token_here

    The subprotocol method is more secure as tokens won't appear in server logs.
    """

    # Try subprotocol authentication first (more secure)
    auth_success, client, error_msg = await authenticate_websocket_subprotocol(
        websocket, username, auth_service, "livestream-v3"
    )

    # Fall back to query parameter authentication if subprotocol auth failed and token is provided
    if not auth_success and token:
        logger.info("Subprotocol auth failed, trying query parameter auth")
        auth_success, client, error_msg = await authenticate_websocket_query_param(
            websocket, token, username, auth_service
        )

    # If both methods failed, close connection
    if not auth_success:
        await close_websocket_with_auth_error(
            websocket, error_msg or "Authentication failed"
        )
        return

    # Accept the WebSocket connection with the proper subprotocol
    await websocket.accept(subprotocol="livestream-v3")

    # Initialize variables
    frame_count = 0
    max_frames = 1800
    frame_buffer_size = 30
    frames_buffer = []
    pending_analyses = set()
    connection_active = True
    saved_seconds = set()

    # Preprocess exercise name
    exercise_name = exercise_name.lower().replace(" ", "_")

    async def analyze_buffer(
        frames_buffer: List[Any], current_frame_count: int, second_number: int
    ):
        """Analyze the buffer of frames and send results using optimized sequential processing"""
        loop = asyncio.get_running_loop()

        try:
            # Use sequential processing similar to the working WebsocketRouter.py
            # This prevents thread pool saturation by avoiding concurrent inference
            def sequential_analysis():
                """Sequential frame analysis to prevent thread saturation"""
                if len(frames_buffer) < 2:
                    return None, None, None, None

                # Use the last two frames for analysis
                current_frame = frames_buffer[-1]
                previous_frame = (
                    frames_buffer[-2] if len(frames_buffer) > 1 else frames_buffer[-1]
                )

                # Process frames sequentially instead of concurrently
                # Get pose keypoints for current and previous frames
                current_pose_result = inference_service._pose_estimation.infer(
                    current_frame
                )
                previous_pose_result = inference_service._pose_estimation.infer(
                    previous_frame
                )

                # Get object detections from current frame
                objects_result = inference_service._object_detection.infer(
                    current_frame
                )

                # Get action recognition from the full buffer
                action_result = inference_service._action_recognition.infer(
                    frames_buffer
                )

                # Extract data
                current_pose = current_pose_result.get("keypoints", {})
                previous_pose = previous_pose_result.get("keypoints", {})
                objects = objects_result.get("predictions", [])
                predicted_class_name = action_result.get(
                    "predicted_class_name", exercise_name
                )

                return current_pose, previous_pose, objects, predicted_class_name

            # Run the sequential analysis in the shared thread pool
            current_pose, previous_pose, objects, predicted_class_name = (
                await loop.run_in_executor(thread_pool, sequential_analysis)
            )

            if current_pose is None:
                logger.warning("Sequential analysis returned no results")
                return None, None

            # Process with ComVisUseCase using the same thread pool
            features = comvis_service.load_to_features_model(
                previous_pose=previous_pose,
                current_pose=current_pose,
                object_inference=comvis_service.load_to_object_model(objects),
                class_name=predicted_class_name,
            )

            # Get suggestions and accuracy
            accuracy, suggestions = comvis_service.get_suggestions(
                features, exercise_name
            )

            # Create exercise data model
            exercise_data_base_model = comvis_service.load_exercise_data(
                frame_index=str(current_frame_count),
                features=features,
                suggestions=suggestions,
                frame_id=f"frame_{current_frame_count}",
            )

            # Use exact second number for the time frame key
            time_frame = f"second_{second_number}"

            # Database operations - properly await the async save operation
            try:
                await comvis_service.save_exercise_data(
                    username=username,
                    exercise_name=exercise_name.replace("_", " "),
                    date=comvis_service.format_date(datetime.now().isoformat()),
                    time_frame=time_frame,
                    exercise_data=jsonable_encoder(exercise_data_base_model),
                    db=db,
                )
                logger.info(
                    f"Saved analysis for {time_frame} at frame {current_frame_count}"
                )
            except Exception as save_error:
                logger.error(
                    f"Failed to save analysis for {time_frame}: {str(save_error)}"
                )

            # Only send message to client if connection is still active
            if connection_active and websocket.client_state.name == "CONNECTED":
                try:
                    await websocket.send_json(
                        {
                            "suggestions": suggestions,
                            "accuracy": str(accuracy),
                            "predicted_class": predicted_class_name,
                            "processing_frame": current_frame_count,
                        }
                    )
                except RuntimeError as e:
                    logger.warning(f"Could not send analysis results: {str(e)}")
                    # Don't raise the exception, just log it

            return suggestions, accuracy
        except Exception as e:
            logger.error(f"Analysis task failed: {str(e)}", exc_info=True)
            return None, None

    try:
        while connection_active and frame_count < max_frames:
            # Receive frame data
            byte_data = await websocket.receive_bytes()

            # Check for completion signal
            if byte_data == b"COMPLETED":
                logger.info(f"Received completion signal from client")
                await websocket.send_json({"status": "COMPLETED_ACK"})
                continue  # Skip the rest of the loop

            # Parse frame (non-blocking)
            frame = await comvis_service.parse_frame(byte_data)
            if frame is not None:
                # Get the current event loop
                loop = asyncio.get_running_loop()

                # Process frame asynchronously using the shared thread pool
                # Remove the thread_pool parameter to prevent additional thread creation
                processed_frame = await loop.run_in_executor(
                    thread_pool,
                    lambda: comvis_service.frame_repo.process_frame(frame)[
                        0
                    ],  # Only need the processed frame
                )

                # Add to buffer
                frames_buffer.append(processed_frame)
                if len(frames_buffer) > frame_buffer_size:
                    frames_buffer.pop(0)

                # Increment frame count
                frame_count += 1

                # Calculate the current second based on frame count
                current_second = frame_count // 30

                # Check if we need to save this second and haven't saved it yet
                if frame_count % 30 == 0 and current_second not in saved_seconds:
                    # Mark this second as saved
                    saved_seconds.add(current_second)

                    # Schedule analysis and don't wait for it
                    analysis_task = asyncio.create_task(
                        analyze_buffer(
                            frames_buffer.copy(), frame_count, current_second
                        )
                    )

                    # Properly track the task
                    pending_analyses.add(analysis_task)

                    # Use a proper callback that captures the task
                    def task_done_callback(completed_task, task_ref=analysis_task):
                        pending_analyses.discard(task_ref)
                        if completed_task.exception():
                            logger.error(
                                f"Analysis task failed: {completed_task.exception()}"
                            )

                    analysis_task.add_done_callback(task_done_callback)

                # Return feedback to client about frame processing
                try:
                    await websocket.send_json(
                        {
                            "status": "processing",
                            "frame_count": frame_count,
                            "current_second": current_second,
                        }
                    )
                except RuntimeError as e:
                    logger.warning(f"Could not send processing status: {str(e)}")
                    connection_active = False
                    break

            # Short yield to allow other tasks to run
            await asyncio.sleep(0.001)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected after processing {frame_count} frames")
        logger.info(f"Client state: {websocket.client_state.name}")
        connection_active = False
    except Exception as e:
        logger.error(f"Error in websocket handler: {str(e)}", exc_info=True)
        connection_active = False
    finally:
        # Mark connection as inactive to prevent new messages
        connection_active = False

        # Wait for pending analyses to complete but don't send messages
        if pending_analyses:
            try:
                # Set a longer timeout to allow database operations to complete
                done, pending = await asyncio.wait(pending_analyses, timeout=10.0)

                # Log any incomplete tasks and cancel them
                if pending:
                    logger.warning(
                        f"{len(pending)} analysis tasks did not complete in time, cancelling them"
                    )
                    for task in pending:
                        task.cancel()
                    # Wait a bit for cancellation to complete
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error waiting for pending analyses: {str(e)}")

        # Cleanup resources
        frames_buffer.clear()

        # Try to tear down the inference service to release resources
        try:
            inference_service.clear_caches()
        except Exception as e:
            logger.error(f"Error clearing inference caches: {str(e)}")
