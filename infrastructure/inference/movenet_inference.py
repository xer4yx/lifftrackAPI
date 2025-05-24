import asyncio
import os
import cv2
import time
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple

from .constants import KEYPOINT_DICT
from lifttrack import config
from lifttrack.v2.comvis.utils import resize_to_192x192
from lifttrack.utils.logging_config import setup_logger
from core.interface import InferenceInterface

# Setup logging
logger = setup_logger("movenet-inference", "inference.log")

class PoseNetInferenceService(InferenceInterface):
    """
    A service that implements the InferenceInterface using TensorFlow's MoveNet model.
    This service handles human pose estimation using a TensorFlow Hub model.
    Optimized for performance with memory caching, concurrent processing, and TensorFlow best practices.
    """

    def __init__(self, 
                 model_url: Optional[str] = None,
                 confidence_threshold: float = 0.1,
                 max_workers: int = 4,
                 enable_gpu: bool = True) -> None:
        """
        Initialize the PoseNet Inference Service.

        Args:
            model_url: URL for the TensorFlow Hub MoveNet model
            confidence_threshold: Minimum confidence threshold for keypoints
            max_workers: Maximum number of workers for concurrent operations
            enable_gpu: Whether to enable GPU acceleration
        """
        # Performance optimization: Configure TensorFlow
        self._configure_tensorflow(enable_gpu)
        
        # Get configuration from config file if not provided
        self._model_url = model_url or config.get(section="TensorHub", option="MOVENET_MODEL")
        self._confidence_threshold = confidence_threshold
        self._max_workers = max_workers
        
        # Initialize the model
        try:
            # Performance optimization: Load model with caching and optimizations
            self._model = self._load_model_optimized()
            self._movenet = self._model.signatures["serving_default"]
            logger.info(f"Initialized PoseNet Inference Service with model: {self._model_url}")
        except Exception as e:
            logger.error(f"Failed to initialize PoseNet Inference Service: {e}")
            raise RuntimeError(f"Failed to load MoveNet model: {e}")
        
        # Cache for already processed frames to avoid redundant computations
        self._prediction_cache = {}
        self._cache_max_size = 100  # Limit cache size to avoid memory issues

        # Define tf.function once during initialization to avoid retracing
        self._preprocess_tf = tf.function(
            self._preprocess_tf_impl,
            input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8)]
        )

    def _configure_tensorflow(self, enable_gpu: bool) -> None:
        """
        Configure TensorFlow for optimal performance.
        
        Args:
            enable_gpu: Whether to enable GPU acceleration
        """
        # Performance optimization: Configure memory growth instead of pre-allocating all GPU memory
        if enable_gpu:
            try:
                physical_devices = tf.config.list_physical_devices('GPU')
                if physical_devices:
                    logger.info(f"Found {len(physical_devices)} GPU(s)")
                    for device in physical_devices:
                        tf.config.experimental.set_memory_growth(device, True)
                    
                    # Set a better thread configuration
                    tf.config.threading.set_intra_op_parallelism_threads(4)
                    tf.config.threading.set_inter_op_parallelism_threads(4)
                    
                    # Set TensorFlow thread mode
                    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
                    os.environ['TF_GPU_THREAD_COUNT'] = '1'
                else:
                    logger.warning("No GPU found. Using CPU.")
            except Exception as e:
                logger.warning(f"Error configuring GPU: {e}. Falling back to CPU.")
        
        # Optimize TensorFlow operations
        tf.config.optimizer.set_jit(True)  # Enable XLA compilation for faster execution

    def _load_model_optimized(self) -> tf.keras.Model:
        """
        Load the MoveNet model with optimizations.
        
        Returns:
            Loaded and optimized TensorFlow model
        """
        # Performance optimization: Use tf.keras.utils.custom_object_scope for custom operations
        model = hub.load(self._model_url)
        
        # Optimize model with TensorRT if available (for NVIDIA GPUs)
        if tf.config.list_physical_devices('GPU') and self._check_tensorrt_availability():
            try:
                # Save and convert model to optimize with TensorRT
                model = tf.function(model, jit_compile=True)
                logger.info("Applied JIT compilation to model")
            except Exception as e:
                logger.warning(f"Failed to apply TensorRT optimization: {e}")
        
        return model

    def _check_tensorrt_availability(self) -> bool:
        """
        Check if TensorRT is available.
        
        Returns:
            True if TensorRT is available, False otherwise
        """
        try:
            return hasattr(tf, 'experimental') and hasattr(tf.experimental, 'tensorrt')
        except:
            return False

    def _preprocess_tf_impl(self, img: tf.Tensor) -> tf.Tensor:
        """
        TensorFlow implementation of image preprocessing.
        
        Args:
            img: Input image tensor
            
        Returns:
            Preprocessed tensor
        """
        # Resize and normalize in one step
        img = tf.image.resize_with_pad(img, 192, 192)
        img = tf.cast(img, dtype=tf.int32)
        return img

    def _preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """
        Preprocess image for the model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed TensorFlow tensor
        """
        # Create a unique hash for the image to use for caching
        image_hash = hash(image.tobytes())
        
        # Check if preprocessed image is in cache
        if image_hash in self._prediction_cache and 'preprocessed' in self._prediction_cache[image_hash]:
            return self._prediction_cache[image_hash]['preprocessed']
        
        # Preprocess image using the pre-compiled tf.function
        resized_image = resize_to_192x192(image)
        preprocessed = self._preprocess_tf(resized_image)
        preprocessed = tf.expand_dims(preprocessed, axis=0)
        
        # Cache the preprocessed image
        if image_hash not in self._prediction_cache:
            self._prediction_cache[image_hash] = {}
        self._prediction_cache[image_hash]['preprocessed'] = preprocessed
        
        # Manage cache size
        if len(self._prediction_cache) > self._cache_max_size:
            # Remove oldest item
            oldest_key = next(iter(self._prediction_cache))
            del self._prediction_cache[oldest_key]
        
        return preprocessed

    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary containing keypoints and other inference results
        """
        try:
            start_time = time.time()
            
            # Create image hash for caching
            image_hash = hash(image.tobytes())
            
            # Check if prediction is in cache
            if image_hash in self._prediction_cache and 'result' in self._prediction_cache[image_hash]:
                keypoints = self._prediction_cache[image_hash]['result']
                logger.debug("Using cached prediction result")
            else:
                # Preprocess the image
                input_img = self._preprocess_image(image)
                
                # Perform inference
                results = self._movenet(input_img)
                keypoints_data = results["output_0"].numpy()[0, 0, :, :3]
                
                # Process keypoints
                keypoints = self._process_keypoints(keypoints_data, image.shape)
                
                # Cache the results
                if image_hash not in self._prediction_cache:
                    self._prediction_cache[image_hash] = {}
                self._prediction_cache[image_hash]['result'] = keypoints
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Inference completed in {elapsed_time:.4f}s")
            
            return {
                "keypoints": keypoints,
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            return {"keypoints": {}, "error": str(e)}
        

    def infer_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Run inference on multiple images concurrently.

        Args:
            images: List of input images as numpy arrays

        Returns:
            List of dictionaries containing keypoints and other inference results
        """
        if not images:
            return []
            
        try:
            start_time = time.time()
            logger.debug(f"Starting batch inference with {len(images)} images")

            # Performance optimization: Use ThreadPoolExecutor for concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                # Submit inference tasks
                futures = [executor.submit(self.infer, image) for image in images]
                
                # Collect results in order
                results = [future.result() for future in futures]
                
            elapsed_time = time.time() - start_time
            logger.debug(f"Batch inference completed in {elapsed_time:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch inference: {e}", exc_info=True)
            return [{"keypoints": {}, "error": str(e)} for _ in range(len(images))]

    def _process_keypoints(self, keypoints_data: np.ndarray, image_shape: Tuple[int, int, int]) -> Dict[str, Tuple[int, int, float]]:
        """
        Process keypoints data from the model.
        
        Args:
            keypoints_data: Raw keypoints data from the model
            image_shape: Shape of the original image (height, width, channels)
            
        Returns:
            Dictionary of keypoints with their coordinates and confidence scores
        """
        y, x, _ = image_shape
        shaped_keypoints = {}

        for name, index in KEYPOINT_DICT.items():
            ky, kx, kp_conf = keypoints_data[index]
            cx, cy = int(kx * x), int(ky * y)
            shaped_keypoints[name] = (cx, cy, float(kp_conf))

        return shaped_keypoints

    def visualize_pose(self, frame: np.ndarray, keypoints: Dict[str, Tuple[int, int, float]]) -> np.ndarray:
        """
        Draw keypoints and connections on the frame.
        
        Args:
            frame: Input frame
            keypoints: Dictionary of keypoints with their coordinates and confidence scores
            
        Returns:
            Frame with keypoints and connections drawn
        """
        try:
            annotated_frame = frame.copy()
            
            # Define connections between keypoints for skeleton visualization
            connections = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('right_shoulder', 'right_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'),
                ('right_hip', 'right_knee'),
                ('left_knee', 'left_ankle'),
                ('right_knee', 'right_ankle'),
            ]
            
            # Draw connections
            for connection in connections:
                start_point, end_point = connection
                if (start_point in keypoints and end_point in keypoints and 
                    keypoints[start_point][2] > self._confidence_threshold and 
                    keypoints[end_point][2] > self._confidence_threshold):
                    
                    cv2.line(annotated_frame, 
                             (keypoints[start_point][0], keypoints[start_point][1]),
                             (keypoints[end_point][0], keypoints[end_point][1]),
                             (0, 255, 255), 2)
            
            # Draw keypoints
            for name, (x, y, confidence) in keypoints.items():
                if confidence > self._confidence_threshold:
                    # Draw keypoint as a small circle
                    cv2.circle(annotated_frame, (x, y), 4, (0, 255, 0), -1)
                    
                    # Add text label
                    cv2.putText(annotated_frame, f"{name} ({confidence:.2f})", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.3, (255, 0, 0), 1)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error visualizing pose: {e}")
            return frame
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the inference service is healthy and operational.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Create a small test image
            test_image = np.zeros((192, 192, 3), dtype=np.uint8)
            
            # Try to run inference
            start_time = time.time()
            _ = self.infer(test_image)
            elapsed_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "model_url": self._model_url,
                "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                "test_inference_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def infer_async(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single image asynchronously.

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary containing keypoints and other inference results
        """
        try:
            # Use event loop to offload the computation to a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.infer, image)
        except Exception as e:
            logger.error(f"Error during async inference: {e}", exc_info=True)
            return {"keypoints": {}, "error": str(e)}
    
    def clear_cache(self) -> None:
        """
        Clear the prediction cache to free memory.
        """
        self._prediction_cache.clear()
        logger.debug("Prediction cache cleared")
        
    def teardown(self) -> None:
        """
        Clean up resources used by the inference service.
        """
        try:
            # Clear cache to free memory
            self.clear_cache()
            
            # Clear TensorFlow session to release GPU memory
            if len(tf.config.list_physical_devices('GPU')) >= 1:
                tf.keras.backend.clear_session()
            
            # Set model variables to None to help garbage collection
            self._movenet = None
            self._model = None
            
            # Call garbage collector
            import gc
            gc.collect()
            
            logger.info("PoseNetInferenceService successfully torn down")
        except Exception as e:
            logger.error(f"Error during PoseNetInferenceService teardown: {e}") 