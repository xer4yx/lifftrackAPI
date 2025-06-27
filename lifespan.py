from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI

from lifttrack.utils.logging_config import setup_logger

from core.interface import InferenceInterface
from infrastructure.inference.factory import InferenceServiceFactory

# Setup logger
logger = setup_logger("lifespan", "lifespan.log")

# Global registry to store inference service instances
inference_services: Dict[str, InferenceInterface] = {}


def get_inference_service(service_name: str) -> Optional[InferenceInterface]:
    """
    Get an inference service by name.

    Args:
        service_name: Name of the inference service to retrieve

    Returns:
        The requested inference service instance or None if not found
    """
    return inference_services.get(service_name.lower())


async def teardown_services():
    """
    Clean up all inference services during application shutdown.
    """
    for service_name, service in inference_services.items():
        try:
            logger.info(f"Tearing down {service_name} inference service...")
            service.teardown()
            logger.info(f"{service_name} inference service torn down successfully")
        except Exception as e:
            logger.error(f"Error tearing down {service_name} inference service: {e}")

    # Clear the services dictionary
    inference_services.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.

    This handles the initialization and teardown of inference services,
    ensuring proper resource management throughout the application lifecycle.

    Args:
        app: FastAPI application instance
    """
    # Startup: Initialize inference services
    try:
        logger.info("Initializing inference services...")

        # Initialize PoseNet service
        inference_services["posenet"] = (
            InferenceServiceFactory.create_inference_service(
                service_type="posenet", config={"enable_gpu": False, "max_workers": 1}
            )
        )
        logger.info("PoseNet inference service initialized")

        # Initialize Roboflow service
        inference_services["roboflow"] = (
            InferenceServiceFactory.create_inference_service(
                service_type="roboflow", config={"max_workers": 1}
            )
        )
        logger.info("Roboflow inference service initialized")

        # Initialize VideoAction service
        inference_services["videoaction"] = (
            InferenceServiceFactory.create_inference_service(
                service_type="videoaction",
                config={"enable_gpu": False, "max_workers": 1, "num_frames": 30},
            )
        )
        logger.info("VideoAction inference service initialized")

        # Make services available through the app state
        app.state.inference_services = inference_services

        logger.info("All inference services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing inference services: {e}")
        # Clean up any partially initialized services
        await teardown_services()
        raise

    yield  # Application runs here

    # Shutdown: Clean up resources
    logger.info("Shutting down inference services...")
    await teardown_services()
    logger.info("All inference services shut down successfully")
