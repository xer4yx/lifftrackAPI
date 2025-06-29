from fastapi import Depends
from functools import lru_cache
from typing import Optional

from core.interface import (
    PoseFeatureInterface,
    FormAnalysisInterface,
    FrameRepositoryInterface,
    FeatureRepositoryInterface,
    FeatureMetricRepositoryInterface,
    DataHandlerInterface,
)
from core.usecase.comvis_usecase import ComVisUseCase, FeatureMetricUseCase
from infrastructure.di import (
    get_form_analysis_repository,
    get_pose_feature_repository,
    get_frame_repository,
    get_feature_repository,
    get_feature_metric_repository,
    get_data_repository,
    get_feature_metrics_data_repository,
)


@lru_cache(maxsize=1)
def get_comvis_usecase(
    pose_feature_repository: PoseFeatureInterface = Depends(
        get_pose_feature_repository
    ),
    form_analysis_repository: FormAnalysisInterface = Depends(
        get_form_analysis_repository
    ),
    frame_repository: FrameRepositoryInterface = Depends(get_frame_repository),
    feature_repository: FeatureRepositoryInterface = Depends(get_feature_repository),
    data_repository: DataHandlerInterface = Depends(get_data_repository),
) -> ComVisUseCase:
    """
    Get an instance of ComVisUseCase.

    Args:
        pose_feature_repository: Optional PoseFeatureInterface implementation
        form_analysis_repository: Optional FormAnalysisInterface implementation

    Returns:
        ComVisUseCase instance
    """
    return ComVisUseCase(
        pose_feature_repository=pose_feature_repository,
        form_analysis_repository=form_analysis_repository,
        frame_repository=frame_repository,
        feature_repository=feature_repository,
        data_handler=data_repository,
    )


@lru_cache(maxsize=1)
def get_feature_metric_usecase(
    feature_metric_repository: FeatureMetricRepositoryInterface = Depends(
        get_feature_metric_repository
    ),
    data_handler: DataHandlerInterface = Depends(get_feature_metrics_data_repository),
) -> FeatureMetricUseCase:
    """
    Get an instance of FeatureMetricUseCase.
    """
    return FeatureMetricUseCase(
        feature_metric_repository=feature_metric_repository,
        data_handler=data_handler,
    )
