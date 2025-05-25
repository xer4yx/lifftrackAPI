import asyncio
import time
import cv2
import numpy as np
import concurrent.futures
from typing import Dict, Any, List, Optional

from lifttrack.utils.logging_config import setup_logger
from core.interface import InferenceInterface

# Setup logging
logger = setup_logger("inference-usecase", "inference_usecase.log")

class InferenceUseCase:
    """
    A use case that combines multiple inference services to analyze exercises.
    Implements concurrent processing to run different inference models in parallel.
    """

    def __init__(
        self,
        object_detection_service: Optional[InferenceInterface] = None,
        pose_estimation_service: Optional[InferenceInterface] = None,
        action_recognition_service: Optional[InferenceInterface] = None,
        max_workers: int = 4
    ) -> None:
        """
        Initialize the Inference Use Case with specific inference services.

        Args:
            object_detection_service: Service for object detection
            pose_estimation_service: Service for pose estimation
            action_recognition_service: Service for action recognition
            max_workers: Maximum number of workers for concurrent operations
        """
        # Initialize services using factory if not provided
        self._object_detection = object_detection_service
        self._pose_estimation = pose_estimation_service
        self._action_recognition = action_recognition_service
        self._max_workers = max_workers
        
        # Frame buffer for action recognition
        self._frame_buffer = []
        self._buffer_size = 30  # Default buffer size for VideoActionInferenceService
        
        logger.info("Initialized InferenceUseCase with all three inference services")

    async def process_frame_async(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame using all three inference services asynchronously.

        Args:
            frame: Input frame as numpy array

        Returns:
            Combined results from all three inference services
        """
        try:
            start_time = time.time()
            
            # Create tasks for concurrent execution
            loop = asyncio.get_event_loop()
            object_task = loop.run_in_executor(None, self._object_detection.infer, frame)
            pose_task = loop.run_in_executor(None, self._pose_estimation.infer, frame)
            
            # Update frame buffer for action recognition
            if len(self._frame_buffer) >= self._buffer_size:
                self._frame_buffer.pop(0)
            self._frame_buffer.append(frame)
            
            # Only run action recognition if we have enough frames
            if len(self._frame_buffer) == self._buffer_size:
                action_task = loop.run_in_executor(None, self._action_recognition.infer, self._frame_buffer)
            else:
                action_task = asyncio.create_task(asyncio.sleep(0))  # Dummy task
            
            # Wait for all tasks to complete
            objects_result, pose_result, action_result = await asyncio.gather(
                object_task, pose_task, action_task
            )
            
            # Process action result based on whether we had enough frames
            if len(self._frame_buffer) < self._buffer_size:
                action_result = {"status": "buffering", "frames_needed": self._buffer_size - len(self._frame_buffer)}
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Combined inference completed in {elapsed_time:.4f}s")
            
            # Combine results
            return {
                "objects": objects_result.get("predictions", []),
                "pose": pose_result.get("keypoints", {}),
                "action": action_result,
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Error during combined inference: {e}", exc_info=True)
            return {"error": str(e)}

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame using all three inference services sequentially.
        This is a synchronous version of process_frame_async.

        Args:
            frame: Input frame as numpy array

        Returns:
            Combined results from all three inference services
        """
        try:
            start_time = time.time()
            
            # Run object detection
            objects_result = self._object_detection.infer(frame)
            
            # Run pose estimation
            pose_result = self._pose_estimation.infer(frame)
            
            # Update frame buffer for action recognition
            if len(self._frame_buffer) >= self._buffer_size:
                self._frame_buffer.pop(0)
            self._frame_buffer.append(frame)
            
            # Run action recognition if we have enough frames
            if len(self._frame_buffer) == self._buffer_size:
                action_result = self._action_recognition.infer(self._frame_buffer)
            else:
                action_result = {"status": "buffering", "frames_needed": self._buffer_size - len(self._frame_buffer)}
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Sequential inference completed in {elapsed_time:.4f}s")
            
            # Combine results
            return {
                "objects": objects_result.get("predictions", []),
                "pose": pose_result.get("keypoints", {}),
                "action": action_result,
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Error during sequential inference: {e}", exc_info=True)
            return {"error": str(e)}

    def process_video(self, video_path: str, output_path: Optional[str] = None, max_frames: int = 300) -> Dict[str, Any]:
        """
        Process a video file using all three inference services.

        Args:
            video_path: Path to the input video file
            output_path: Path to save the annotated video (optional)
            max_frames: Maximum number of frames to process

        Returns:
            Summary of processed results
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize video writer if output path is provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process frames
            results = []
            frame_index = 0
            
            while cap.isOpened() and frame_index < min(frame_count, max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_result = self.process_frame(frame)
                results.append(frame_result)
                
                # Annotate and save frame if output path is provided
                if writer:
                    annotated_frame = self._annotate_frame(frame, frame_result)
                    writer.write(annotated_frame)
                
                frame_index += 1
                if frame_index % 10 == 0:
                    logger.info(f"Processed {frame_index}/{min(frame_count, max_frames)} frames")
            
            # Clean up
            cap.release()
            if writer:
                writer.release()
            
            # Summarize results
            return self._summarize_results(results)
            
        except Exception as e:
            logger.error(f"Error during video processing: {e}", exc_info=True)
            return {"error": str(e)}

    async def process_video_async(self, video_path: str, output_path: Optional[str] = None, max_frames: int = 300) -> Dict[str, Any]:
        """
        Process a video file asynchronously using all three inference services.

        Args:
            video_path: Path to the input video file
            output_path: Path to save the annotated video (optional)
            max_frames: Maximum number of frames to process

        Returns:
            Summary of processed results
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize video writer if output path is provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process frames
            results = []
            frame_index = 0
            
            while cap.isOpened() and frame_index < min(frame_count, max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame asynchronously
                frame_result = await self.process_frame_async(frame)
                results.append(frame_result)
                
                # Annotate and save frame if output path is provided
                if writer:
                    annotated_frame = self._annotate_frame(frame, frame_result)
                    writer.write(annotated_frame)
                
                frame_index += 1
                if frame_index % 10 == 0:
                    logger.info(f"Processed {frame_index}/{min(frame_count, max_frames)} frames")
            
            # Clean up
            cap.release()
            if writer:
                writer.release()
            
            # Summarize results
            return self._summarize_results(results)
            
        except Exception as e:
            logger.error(f"Error during async video processing: {e}", exc_info=True)
            return {"error": str(e)}

    def process_frames_concurrent(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process multiple frames using concurrent execution.

        Args:
            frames: List of input frames

        Returns:
            List of combined results from all three inference services
        """
        if not frames:
            return []
            
        try:
            start_time = time.time()
            logger.debug(f"Starting concurrent processing of {len(frames)} frames")

            # Use ThreadPoolExecutor for concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                # Submit processing tasks
                futures = [executor.submit(self.process_frame, frame) for frame in frames]
                
                # Collect results in order
                results = [future.result() for future in futures]
                
            elapsed_time = time.time() - start_time
            logger.debug(f"Concurrent processing completed in {elapsed_time:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during concurrent processing: {e}", exc_info=True)
            return [{"error": str(e)} for _ in range(len(frames))]

    def _annotate_frame(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Annotate a frame with inference results.

        Args:
            frame: Input frame
            result: Inference results

        Returns:
            Annotated frame
        """
        try:
            annotated_frame = frame.copy()
            
            # Annotate object detections
            if "objects" in result and result["objects"]:
                annotated_frame = self._object_detection.visualize_detections(
                    annotated_frame, result["objects"]
                )
            
            # Annotate pose keypoints
            if "pose" in result and result["pose"]:
                annotated_frame = self._pose_estimation.visualize_pose(
                    annotated_frame, result["pose"]
                )
            
            # Add action recognition label
            if "action" in result and "predicted_class_name" in result["action"]:
                action_name = result["action"]["predicted_class_name"]
                confidence = max(result["action"].get("confidence_scores", [0]))
                
                cv2.putText(
                    annotated_frame,
                    f"Action: {action_name} ({confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error annotating frame: {e}")
            return frame

    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize results from processing a video.

        Args:
            results: List of frame results

        Returns:
            Summary of results
        """
        if not results:
            return {"error": "No results to summarize"}
            
        try:
            # Count detected objects by class
            object_counts = {}
            for result in results:
                if "objects" in result:
                    for obj in result["objects"]:
                        obj_class = obj.get("class", "unknown")
                        if obj_class in object_counts:
                            object_counts[obj_class] += 1
                        else:
                            object_counts[obj_class] = 1
            
            # Calculate average processing time
            avg_processing_time = sum(result.get("processing_time", 0) for result in results) / len(results)
            
            # Count action predictions
            action_counts = {}
            for result in results:
                if "action" in result and "predicted_class_name" in result["action"]:
                    action_name = result["action"]["predicted_class_name"]
                    if action_name in action_counts:
                        action_counts[action_name] += 1
                    else:
                        action_counts[action_name] = 1
            
            # Determine most common action
            most_common_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else "unknown"
            
            return {
                "frames_processed": len(results),
                "object_counts": object_counts,
                "action_counts": action_counts,
                "most_common_action": most_common_action,
                "average_processing_time": avg_processing_time
            }
            
        except Exception as e:
            logger.error(f"Error summarizing results: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of all inference services.

        Returns:
            Health status of all services
        """
        try:
            # Check each service
            object_health = self._object_detection.health_check()
            pose_health = self._pose_estimation.health_check()
            action_health = self._action_recognition.health_check()
            
            # Determine overall health
            services_healthy = (
                object_health.get("status") == "healthy" and
                pose_health.get("status") == "healthy" and
                action_health.get("status") == "healthy"
            )
            
            return {
                "status": "healthy" if services_healthy else "unhealthy",
                "object_detection": object_health,
                "pose_estimation": pose_health,
                "action_recognition": action_health,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def clear_caches(self) -> None:
        """
        Clear caches in all inference services to free memory.
        """
        try:
            # Clear frame buffer
            self._frame_buffer.clear()
            
            # Clear service caches
            if hasattr(self._pose_estimation, "clear_cache"):
                self._pose_estimation.clear_cache()
                
            logger.debug("All caches cleared")
            
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
            
    def teardown(self) -> None:
        """
        Clean up resources used by the inference use case and all its services.
        Properly teardown all the inference services to release memory and resources.
        """
        try:
            logger.info("Starting InferenceUseCase teardown process")
            
            # Clear all caches first
            self.clear_caches()
            
            # Teardown individual services
            services = [
                self._object_detection,
                self._pose_estimation,
                self._action_recognition
            ]
            
            for service in services:
                if service and hasattr(service, "teardown"):
                    try:
                        service.teardown()
                        logger.debug(f"Successfully torn down {service.__class__.__name__}")
                    except Exception as service_error:
                        logger.error(f"Error tearing down {service.__class__.__name__}: {service_error}")
            
            # Set services to None to help garbage collection
            self._object_detection = None
            self._pose_estimation = None
            self._action_recognition = None
            
            # Final garbage collection
            import gc
            gc.collect()
            
            # Clear TensorFlow session at the end for final cleanup
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                logger.debug("TensorFlow session cleared")
            except ImportError:
                logger.debug("TensorFlow not available, skipping session clearing")
            
            logger.info("InferenceUseCase teardown completed successfully")
        except Exception as e:
            logger.error(f"Error during InferenceUseCase teardown: {e}", exc_info=True) 