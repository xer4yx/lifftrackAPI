import asyncio
import os
import time
import numpy as np
import cv2
import tensorflow as tf
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple, Union

from .constants import CLASS_NAMES
from core.interface import InferenceInterface
from lifttrack import config
from lifttrack.utils.logging_config import setup_logger
from lifttrack.v2.comvis.analyze_features import analyze_annotations
from lifttrack.v2.comvis.utils import resize_to_128x128

# Setup logging
logger = setup_logger("action-inference", "inference.log")


class VideoActionInferenceService(InferenceInterface):
    """
    A service that implements the InferenceInterface for video action recognition.
    This service identifies exercise types from video frames and provides form suggestions.
    Optimized for performance with memory caching, concurrent processing, and TensorFlow best practices.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        fallback_model_path: Optional[str] = None,
        num_frames: int = 30,
        max_workers: int = 4,
        enable_gpu: bool = True,
    ) -> None:
        """
        Initialize the Video Action Inference Service.

        Args:
            model_path: Path to the primary TensorFlow model for action recognition
            fallback_model_path: Path to a fallback model (lighter version)
            num_frames: Number of frames to use for inference
            max_workers: Maximum number of workers for concurrent operations
            enable_gpu: Whether to enable GPU acceleration
        """
        # Performance optimization: Configure TensorFlow
        self._configure_tensorflow(enable_gpu)

        # Get configuration from config file if not provided
        self._model_path = model_path or config.get("CNN", "lifttrack_cnn")
        self._fallback_model_path = fallback_model_path or config.get(
            "CNN", "lifttrack_cnn_lite"
        )
        self._num_frames = num_frames
        self._max_workers = max_workers

        # Initialize the model
        self._model = self._load_model_optimized()

        # Cache for already processed frames to avoid redundant computations
        self._prediction_cache = {}
        self._cache_max_size = 100  # Limit cache size to avoid memory issues

    def _configure_tensorflow(self, enable_gpu: bool) -> None:
        """
        Configure TensorFlow for optimal performance.

        Args:
            enable_gpu: Whether to enable GPU acceleration
        """
        # Performance optimization: Configure memory growth instead of pre-allocating all GPU memory
        if enable_gpu:
            try:
                physical_devices = tf.config.list_physical_devices("GPU")
                if physical_devices:
                    logger.info(f"Found {len(physical_devices)} GPU(s)")
                    for device in physical_devices:
                        tf.config.experimental.set_memory_growth(device, True)

                    # Set better thread configuration
                    tf.config.threading.set_intra_op_parallelism_threads(4)
                    tf.config.threading.set_inter_op_parallelism_threads(4)

                    # Set TensorFlow thread mode
                    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
                    os.environ["TF_GPU_THREAD_COUNT"] = "1"
                else:
                    logger.warning("No GPU found. Using CPU.")
            except Exception as e:
                logger.warning(f"Error configuring GPU: {e}. Falling back to CPU.")

        # Optimize TensorFlow operations
        tf.config.optimizer.set_jit(True)  # Enable XLA compilation for faster execution

    def _load_model_optimized(self) -> tf.keras.Model:
        """
        Load the 3D CNN model with optimizations and fallback options.

        Returns:
            Loaded and optimized TensorFlow model

        Raises:
            RuntimeError: If both primary and fallback models fail to load
        """
        try:
            # Load model without optimizer for faster loading and less memory usage
            model = tf.keras.models.load_model(self._model_path, compile=False)
            logger.info("Primary action recognition model loaded successfully")

            # Convert model to TensorFlow Lite format for inference optimization
            if (
                tf.config.list_physical_devices("GPU")
                and self._check_tensorrt_availability()
            ):
                try:
                    # Optimize with JIT compilation
                    model_fn = tf.function(model)
                    model = model_fn.get_concrete_function(
                        tf.TensorSpec([1, self._num_frames, 128, 128, 3], tf.float32)
                    )
                    logger.info("Applied JIT compilation to model")
                except Exception as e:
                    logger.warning(f"Failed to apply optimization: {e}")

            return model

        except Exception as primary_error:
            logger.error(f"Error loading primary model: {primary_error}")
            try:
                # Try loading fallback model
                model = tf.keras.models.load_model(
                    self._fallback_model_path, compile=False
                )
                logger.info(
                    f"Loaded fallback model successfully from {self._fallback_model_path}"
                )
                return model
            except Exception as fallback_error:
                logger.error(f"Error loading fallback model: {fallback_error}")
                raise RuntimeError("Failed to load both primary and fallback models")

    def _check_tensorrt_availability(self) -> bool:
        """
        Check if TensorRT is available.

        Returns:
            True if TensorRT is available, False otherwise
        """
        try:
            return hasattr(tf, "experimental") and hasattr(tf.experimental, "tensorrt")
        except:
            return False

    def prepare_frames_for_input(self, frame_list: List[np.ndarray]) -> np.ndarray:
        """
        Prepare frames for model input by resizing and stacking them.
        Uses parallel processing for faster frame preparation.

        Args:
            frame_list: List of frames to process

        Returns:
            numpy array with shape (num_frames, height, width, 3)

        Raises:
            ValueError: If frame_list is empty or has invalid content
        """
        if len(frame_list) == 0 or not isinstance(frame_list[0], np.ndarray):
            raise ValueError("frame_list must be a non-empty list of numpy arrays")

        # Create a frame hash for caching
        frames_hash = hash(frame_list[0].tobytes())
        if frames_hash in self._prediction_cache:
            return self._prediction_cache[frames_hash]

        # Handle temporal dimension
        if len(frame_list) != self._num_frames:
            if len(frame_list) > self._num_frames:
                frame_list = frame_list[: self._num_frames]
            else:
                last_frame = frame_list[-1]
                padding = [last_frame] * (self._num_frames - len(frame_list))
                frame_list = list(frame_list) + padding

        # Use parallel processing for frame preparation
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            frames_resized = list(executor.map(self._process_single_frame, frame_list))

        # Stack frames
        frames_stack = np.stack(
            frames_resized, axis=0
        )  # Shape: (num_frames, 128, 128, 3)

        # Verify final shape
        expected_shape = (self._num_frames, 128, 128, 3)
        if frames_stack.shape != expected_shape:
            raise ValueError(
                f"Unexpected output dimensions. Expected {expected_shape}, got {frames_stack.shape}"
            )

        # Cache the result
        if len(self._prediction_cache) >= self._cache_max_size:
            # Remove a random item if cache is full
            self._prediction_cache.pop(next(iter(self._prediction_cache)))
        self._prediction_cache[frames_hash] = frames_stack

        return frames_stack

    def _process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for model input.

        Args:
            frame: Input frame

        Returns:
            Processed frame
        """
        # Convert grayscale to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # Convert BGR to RGB if needed (OpenCV uses BGR by default)
        elif frame.shape[-1] == 3 and not np.array_equal(
            frame[:, :, 0], frame[:, :, 2]
        ):  # Check if BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Handle other cases (like RGBA)
        elif frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Resize to 128x128
        frame_resized = resize_to_128x128(frame)
        return frame_resized

    def infer(self, image: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, Any]:
        """
        Run inference on image data (implements InferenceInterface).
        Can accept either a single frame or a list of frames.

        Args:
            image: Input image or list of images as numpy array

        Returns:
            Dictionary containing inference results including predicted class
        """
        start_time = time.time()

        try:
            # Handle either single frame or list of frames
            if isinstance(image, np.ndarray):
                # For a single frame, repeat it to create temporal dimension
                frames = [image] * self._num_frames
            else:
                frames = image

            # Prepare frames for model input
            frames_input = self.prepare_frames_for_input(frames)
            frames_input_batch = np.expand_dims(frames_input, axis=0)

            # Run inference
            predictions = self._model.predict(frames_input_batch, verbose=0)

            # Get predicted class
            predicted_class_index = tf.compat.v1.argmax(predictions[0]).numpy()
            predicted_class_name = CLASS_NAMES.get(predicted_class_index, "unknown")

            # Log and return results
            elapsed_time = time.time() - start_time
            logger.debug(
                f"Inference completed in {elapsed_time:.4f}s with class: {predicted_class_name}"
            )

            return {
                "predicted_class_index": int(predicted_class_index),
                "predicted_class_name": predicted_class_name,
                "confidence_scores": predictions[0].tolist(),
                "processing_time": elapsed_time,
            }

        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            return {"error": str(e)}

    def predict_class(self, frame_list: List[np.ndarray]) -> str:
        """
        Predict the exercise class from a list of frames.

        Args:
            frame_list: List of video frames

        Returns:
            Predicted class name
        """
        result = self.infer(frame_list)
        if "error" in result:
            raise RuntimeError(f"Inference failed: {result['error']}")
        return result["predicted_class_name"]

    def provide_form_suggestions(
        self, predicted_class_name: str, features: Dict[str, Any]
    ) -> str:
        """
        Provides suggestions for improving form based on the predicted class name and extracted features.

        Args:
            predicted_class_name: The predicted class name for the current frame
            features: A dictionary containing extracted features (joint_angles, movement_patterns, speeds, etc.)

        Returns:
            A string containing suggestions for form improvement
        """
        suggestions = []

        # Get features with proper keys and default values
        joint_angles = features.get("joint_angles", {})
        body_alignment = features.get("body_alignment", {})
        speeds = features.get("speeds", {})
        stability = features.get("stability", 0.0)

        if predicted_class_name == "benchpress":
            if (
                joint_angles.get("left_shoulder_left_elbow_left_wrist") is not None
                and joint_angles.get("left_shoulder_left_elbow_left_wrist") > 45
            ):
                suggestions.append(
                    "Keep your elbows tucked at a 45-degree angle to protect your shoulders."
                )

            if (
                body_alignment.get("shoulder_angle") is not None
                and body_alignment.get("shoulder_angle") > 10
            ):
                suggestions.append("Ensure your shoulders are level during the lift.")

            if stability > 0.1:
                suggestions.append("Maintain a stable base to enhance your lift.")

        elif predicted_class_name == "deadlift":
            if stability > 0.1:
                suggestions.append(
                    "Focus on maintaining a stable position throughout the lift."
                )

            if (
                joint_angles.get("left_hip_left_knee_left_ankle") is not None
                and joint_angles.get("left_hip_left_knee_left_ankle") < 170
            ):
                suggestions.append(
                    "Keep your hips lower than your shoulders to maintain proper form."
                )

            if speeds.get("left_hip", 0.0) > 2.0:
                suggestions.append("Control your speed to avoid injury.")

        elif predicted_class_name == "romanian_deadlift":
            if (
                joint_angles.get("left_hip_left_knee_left_ankle") is not None
                and joint_angles.get("left_hip_left_knee_left_ankle") < 160
            ):
                suggestions.append("Ensure your back is flat and hinge at the hips.")

            if stability > 0.1:
                suggestions.append("Maintain stability to prevent rounding your back.")

        elif predicted_class_name == "shoulder_press":
            if speeds.get("left_shoulder", 0.0) > 2.0:
                suggestions.append("Control the speed of your lift to avoid injury.")

            if (
                joint_angles.get("left_shoulder_left_elbow_left_wrist") is not None
                and joint_angles.get("left_shoulder_left_elbow_left_wrist") > 90
            ):
                suggestions.append("Keep your wrists straight and elbows aligned.")

            if stability > 0.1:
                suggestions.append("Maintain a stable core throughout the movement.")

        return "\n".join(suggestions)

    async def analyze_frames(
        self, annotations: List[Dict[str, Any]], object_tracker: Any
    ) -> Dict[str, Any]:
        """
        Analyze frames to extract form suggestions. Uses asyncio for concurrent processing.

        Args:
            annotations: List of annotations with predicted classes
            object_tracker: Object tracker for analyzing movements

        Returns:
            Dictionary containing analysis results and frame count
        """
        analysis_results = []
        max_frames = 1800

        # Process frames in chunks to enable parallel processing
        chunk_size = 30  # Process 30 frames at a time
        tasks = []

        for frame_index in range(0, min(len(annotations), max_frames), chunk_size):
            end_idx = min(frame_index + chunk_size, len(annotations), max_frames)
            chunk = annotations[frame_index:end_idx]
            tasks.append(self._process_frame_chunk(chunk, frame_index, object_tracker))

        # Process chunks in parallel using asyncio
        chunk_results = await asyncio.gather(*tasks)

        # Flatten results
        for chunk_result in chunk_results:
            analysis_results.extend(chunk_result)

        return {
            "frames": analysis_results,
            "total_frames": min(len(annotations), max_frames),
        }

    async def _process_frame_chunk(
        self, chunk: List[Dict[str, Any]], start_index: int, object_tracker: Any
    ) -> List[Dict[str, Any]]:
        """
        Process a chunk of frames asynchronously.

        Args:
            chunk: Chunk of annotations to process
            start_index: Starting index for this chunk
            object_tracker: Object tracker for analyzing movements

        Returns:
            List of analysis results for this chunk
        """
        chunk_results = []

        for i, annotation in enumerate(chunk):
            try:
                frame_index = start_index + i
                if frame_index % 30 == 0:  # Still analyzing every 30th frame
                    predicted_class_index = annotation["predicted_class_index"]
                    predicted_class_name = CLASS_NAMES.get(predicted_class_index)

                    if predicted_class_name is None:
                        raise ValueError(
                            f"Invalid class index: {predicted_class_index}"
                        )

                    # Extract features from the annotation using analyze_features
                    features, _ = analyze_annotations([annotation], object_tracker)

                    # Ensure features are structured correctly for provide_form_suggestions
                    suggestions = self.provide_form_suggestions(
                        predicted_class_name, features[0]
                    )

                    chunk_results.append(
                        {
                            "frame_index": frame_index,
                            "class_name": predicted_class_name,
                            "suggestions": suggestions,
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing frame {start_index + i}: {str(e)}")
                continue

        return chunk_results

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the inference service is healthy and operational.

        Returns:
            Dictionary with health status
        """
        try:
            # Create a test input to check model
            test_frame = np.zeros((128, 128, 3), dtype=np.uint8)
            test_frames = [test_frame] * self._num_frames

            # Run inference on test input
            self.infer(test_frames)

            return {
                "status": "healthy",
                "model_path": self._model_path,
                "timestamp": time.time(),
                "num_frames": self._num_frames,
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "timestamp": time.time()}

    async def infer_async(
        self, image: Union[np.ndarray, List[np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Run inference asynchronously on image data.
        Can accept either a single frame or a list of frames.

        Args:
            image: Input image or list of images as numpy array

        Returns:
            Dictionary containing inference results including predicted class
        """
        try:
            # Use event loop to offload the computation to a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.infer, image)
        except Exception as e:
            logger.error(f"Error during async inference: {e}", exc_info=True)
            return {"error": str(e)}

    def teardown(self) -> None:
        """
        Clean up resources used by the inference service.
        """
        try:
            # Clear prediction cache
            if hasattr(self, "_prediction_cache"):
                self._prediction_cache.clear()

            # This assumes that the GPU is used to run the model
            # Clear TensorFlow session to release GPU memory
            if len(tf.config.list_physical_devices("GPU")) >= 1:
                tf.keras.backend.clear_session()

            # Set model variables to None to help garbage collection
            self._model = None

            # Call garbage collector
            import gc

            gc.collect()

            logger.info("VideoActionInferenceService successfully torn down")
        except Exception as e:
            logger.error(f"Error during VideoActionInferenceService teardown: {e}")
