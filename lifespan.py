from contextlib import asynccontextmanager

from fastapi import FastAPI

from lifttrack.utils.logging_config import setup_logger

# Setup logger
logger = setup_logger("lifespan", "lifespan.log")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan.

    Server-side inference services were removed at the keypoints-in cutover
    (ADR 0001/0002/0009): pose runs on-device and the client streams keypoints,
    so there are no heavy models to load or tear down here.
    """
    logger.info("Application startup: no server-side inference to initialize")
    yield
    logger.info("Application shutdown")
