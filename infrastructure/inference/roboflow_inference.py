import os
import cv2
import time
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from inference_sdk import InferenceHTTPClient

from lifttrack import config
from lifttrack.utils.logging_config import setup_logger
from lifttrack.v2.comvis.utils import resize_to_192x192  # Reusing existing utility
from lifttrack.models import Object

from core.interface import InferenceInterface
from utils import InferenceSdkSettings

# Setup logging
inference_logger = setup_logger("roboflow-inference", "inference.log")


class RoboflowInferenceService(InferenceInterface):
    """
    A service that implements the InferenceInterface using Roboflow's HTTP client.
    This service handles object detection using a Roboflow model running in a Docker container.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        model_version: Optional[int] = None,
        max_workers: int = 4,
    ) -> None:
        """
        Initialize the Roboflow Inference Service.

        Args:
            api_url: URL for the Roboflow Inference Client
            api_key: API key for authentication
            project_id: Roboflow project ID
            model_version: Roboflow model version
            max_workers: Maximum number of workers for concurrent operations
        """
        inference_settings = InferenceSdkSettings()

        # Get configuration from config file if not provided
        self._api_url = api_url or inference_settings.api_url
        self._api_key = api_key or inference_settings.api_key
        self._project_id = project_id or inference_settings.project_id
        self._model_version = model_version or inference_settings.model_version
        self._max_workers = max_workers

        # Initialize the client
        try:
            self._client = InferenceHTTPClient(
                api_url=self._api_url,
                api_key=self._api_key,
            )
            inference_logger.info(
                f"Initialized Roboflow Inference Service at {self._api_url}"
            )
        except Exception as e:
            inference_logger.error(
                f"Failed to initialize Roboflow Inference Service: {e}"
            )
            raise ConnectionError(
                f"Failed to connect to Roboflow Inference service: {e}"
            )

    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary containing inference results
        """
        try:
            start_time = time.time()

            # Prepare the image
            resized_image = resize_to_192x192(image)

            # Run inference
            model_id = f"{self._project_id}/{self._model_version}"
            inference_logger.debug(f"Running inference with model {model_id}")

            roboflow_results = self._client.infer(resized_image, model_id=model_id)

            # Process and validate results
            if not roboflow_results or not isinstance(roboflow_results, dict):
                inference_logger.warning("Invalid inference results received")
                return {"predictions": []}

            # Format predictions as standardized objects
            predictions = self._format_predictions(roboflow_results)

            elapsed_time = time.time() - start_time
            inference_logger.debug(
                f"Inference completed in {elapsed_time:.4f}s with {len(predictions)} objects detected"
            )

            return {"predictions": predictions}

        except Exception as e:
            inference_logger.error(f"Error during inference: {e}", exc_info=True)
            return {"predictions": [], "error": str(e)}

    async def infer_async(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference asynchronously on a single image.

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary containing inference results
        """
        try:
            start_time = time.time()

            # Prepare the image
            resized_image = resize_to_192x192(image)

            # Run inference asynchronously
            model_id = f"{self._project_id}/{self._model_version}"
            inference_logger.debug(f"Running async inference with model {model_id}")

            roboflow_results = await self._client.infer_async(
                resized_image, model_id=model_id
            )

            # Process and validate results
            if not roboflow_results or not isinstance(roboflow_results, dict):
                inference_logger.warning("Invalid async inference results received")
                return {"predictions": []}

            # Format predictions as standardized objects
            predictions = self._format_predictions(roboflow_results)

            elapsed_time = time.time() - start_time
            inference_logger.debug(
                f"Async inference completed in {elapsed_time:.4f}s with {len(predictions)} objects detected"
            )

            return {"predictions": predictions}

        except Exception as e:
            inference_logger.error(f"Error during async inference: {e}", exc_info=True)
            return {"predictions": [], "error": str(e)}

    def infer_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Run inference on multiple images concurrently.

        Args:
            images: List of input images as numpy arrays

        Returns:
            List of dictionaries containing inference results
        """
        if not images:
            return []

        try:
            start_time = time.time()
            inference_logger.debug(
                f"Starting batch inference with {len(images)} images"
            )

            # Use ThreadPoolExecutor for concurrent processing
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers
            ) as executor:
                # Submit inference tasks
                futures = [executor.submit(self.infer, image) for image in images]

                # Collect results in order
                results = [future.result() for future in futures]

            elapsed_time = time.time() - start_time
            inference_logger.debug(f"Batch inference completed in {elapsed_time:.4f}s")

            return results

        except Exception as e:
            inference_logger.error(f"Error during batch inference: {e}", exc_info=True)
            return [{"predictions": [], "error": str(e)} for _ in range(len(images))]

    def _format_predictions(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format raw prediction results into standardized objects.

        Args:
            raw_results: Raw inference results from Roboflow

        Returns:
            List of formatted prediction objects
        """
        try:
            predictions = []

            if "predictions" not in raw_results:
                return predictions

            for pred in raw_results["predictions"]:
                formatted_pred = {
                    "x": pred.get("x", 0),
                    "y": pred.get("y", 0),
                    "width": pred.get("width", 0),
                    "height": pred.get("height", 0),
                    "confidence": pred.get("confidence", 0),
                    "class_id": pred.get("class_id", 0),
                    "class": pred.get("class", ""),
                }
                predictions.append(formatted_pred)

            return predictions

        except Exception as e:
            inference_logger.error(f"Error formatting predictions: {e}")
            return []

    def visualize_detections(
        self,
        frame: np.ndarray,
        predictions: List[Dict[str, Any]],
        conf_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Draw bounding boxes on a frame based on detection results.

        Args:
            frame: Input frame
            predictions: List of predictions from inference
            conf_threshold: Confidence threshold for visualization

        Returns:
            Frame with bounding boxes drawn
        """
        try:
            annotated_frame = frame.copy()

            for pred in predictions:
                # Extract object properties
                x, y = pred.get("x", 0), pred.get("y", 0)
                width, height = pred.get("width", 0), pred.get("height", 0)
                confidence = pred.get("confidence", 0)
                label = pred.get("class", "")

                # Skip low confidence detections
                if confidence < conf_threshold:
                    continue

                # Calculate coordinates for rectangle
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + width), int(y + height)

                # Draw rectangle and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {confidence:.2f}"
                cv2.putText(
                    annotated_frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            return annotated_frame

        except Exception as e:
            inference_logger.error(f"Error visualizing detections: {e}")
            return frame

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the inference service is healthy and operational.

        Returns:
            Dictionary with health status
        """
        try:
            # Get server info to check connection
            server_info = self._client.get_server_info()

            # Run a simple inference with a blank image to check model
            test_image = np.zeros((192, 192, 3), dtype=np.uint8)
            self._client.infer(
                test_image, model_id=f"{self._project_id}/{self._model_version}"
            )

            return {
                "status": "healthy",
                "server_info": server_info,
                "model_id": f"{self._project_id}/{self._model_version}",
                "timestamp": time.time(),
            }
        except Exception as e:
            inference_logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "timestamp": time.time()}

    def teardown(self) -> None:
        """
        Clean up resources used by the inference service.
        """
        try:
            # Close any client connections
            if hasattr(self._client, "close") and callable(self._client.close):
                self._client.close()

            # Set variables to None to help garbage collection
            self._client = None

            inference_logger.info("RoboflowInferenceService successfully torn down")
        except Exception as e:
            inference_logger.error(
                f"Error during RoboflowInferenceService teardown: {e}"
            )
