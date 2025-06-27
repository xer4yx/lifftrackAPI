from fastapi import Depends
from functools import lru_cache

from core.interface import (
    PoseFeatureInterface,
    FormAnalysisInterface,
    NTFInterface,
    FeatureMetricRepositoryInterface,
)
from infrastructure.comvis import (
    PoseFeatureRepository,
    FormAnalysisRepository,
    FrameRepository,
    WeightliftDataRepository,
    FeatureRepository,
    FeatureMetricRepository,
    FeatureMetricsDataRepository,
)
from infrastructure.di import get_firebase_admin


@lru_cache(maxsize=1)
def get_pose_feature_repository() -> PoseFeatureInterface:
    """
    Get an instance of PoseFeatureRepository.

    Returns:
        PoseFeatureInterface implementation
    """
    return PoseFeatureRepository()


@lru_cache(maxsize=1)
def get_form_analysis_repository() -> FormAnalysisInterface:
    """
    Get an instance of FormAnalysisRepository.

    Returns:
        FormAnalysisInterface implementation
    """
    return FormAnalysisRepository()


@lru_cache(maxsize=1)
def get_frame_repository() -> FrameRepository:
    """
    Get an instance of FrameRepository.
    """
    return FrameRepository()


@lru_cache(maxsize=1)
def get_data_repository(
    database_repository: NTFInterface = Depends(get_firebase_admin),
) -> WeightliftDataRepository:
    """
    Get an instance of DataRepository.
    """
    return WeightliftDataRepository(database_repository=database_repository)


@lru_cache(maxsize=1)
def get_feature_metrics_data_repository(
    database_repository: NTFInterface = Depends(get_firebase_admin),
) -> FeatureMetricsDataRepository:
    """
    Get an instance of FeatureMetricsDataRepository.
    """
    return FeatureMetricsDataRepository(database_repository=database_repository)


@lru_cache(maxsize=1)
def get_feature_repository() -> FeatureRepository:
    """
    Get an instance of FeatureRepository.
    """
    return FeatureRepository()


@lru_cache(maxsize=1)
def get_feature_metric_repository() -> FeatureMetricRepositoryInterface:
    """
    Get an instance of FeatureMetricRepository.

    Returns:
        FeatureMetricRepositoryInterface implementation
    """
    return FeatureMetricRepository()
