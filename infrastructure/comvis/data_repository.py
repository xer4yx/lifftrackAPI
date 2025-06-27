from typing import Dict, Any, Optional
from core.entities import PoseFeatures, ExerciseData, FeatureMetrics
from lifttrack.utils.logging_config import setup_logger

from core.interface import NTFInterface, DataHandlerInterface

logger = setup_logger("data-repository", "data_repository.log")


class WeightliftDataRepository(DataHandlerInterface[ExerciseData, None]):
    """
    Repository for handling data operations related to exercise data.
    This repository is responsible for formatting and saving exercise data.
    """

    def __init__(self, database_repository: NTFInterface):
        self.database_repository = database_repository

    def load_to_data_model(
        self,
        features: PoseFeatures,
        suggestions: str,
        frame_index: str,
        frame_id: Optional[str] = None,
    ) -> ExerciseData:
        """
        Save exercise data in an ExerciseData model.

        Args:
            features: PoseFeatures model
            suggestions: Form suggestions
            frame_index: Frame index
            frame_id: Optional frame ID

        Returns:
            ExerciseData model

        Raises:
            Exception: If loading fails
        """
        try:
            return ExerciseData(
                frame=frame_index,
                suggestion=suggestions,
                features=features,
                frame_id=frame_id,
            )
        except Exception as e:
            logger.error(f"Failed to load exercise data: {str(e)}")
            raise

    def format_date(self, date: str) -> str:
        """
        Format a date string.

        Args:
            date: Date string to format

        Returns:
            Formatted date string
        """
        exercise_datetime = date.split(".")[0]
        return exercise_datetime.replace(":", "-")

    async def save_data(
        self,
        username: str,
        exercise_name: str,
        date: str,
        time_frame: str,
        exercise_data: Dict[str, Any],
        db: NTFInterface,
    ) -> None:
        """
        Save progress to the database.

        Args:
            username: User's username
            exercise_name: Exercise name
            date: Date string
            time_frame: Time frame string
            exercise_data: Exercise data to save
            db: NTFInterface instance

        Raises:
            Exception: If saving fails
        """
        try:
            await db.set_data(
                key=f"progress/{username}/{exercise_name.lower()}/{date}/{time_frame}",
                value=exercise_data,
            )
        except Exception as e:
            logger.error(f"Failed to save progress: {str(e)}")
            raise


class FeatureMetricsDataRepository(DataHandlerInterface[FeatureMetrics, None]):
    """
    Repository for handling data operations related to feature metrics.
    This repository is responsible for formatting and saving feature metrics.
    """

    def __init__(self, database_repository: NTFInterface):
        self.data_repository = database_repository

    def load_to_data_model(
        self,
        body_alignment: float,
        joint_consistency: float,
        load_control: float,
        speed_control: float,
        overall_stability: float,
    ) -> FeatureMetrics:
        """
        Load feature metrics into a FeatureMetrics model.

        Args:
            body_alignment: Body alignment
            joint_consistency: Joint consistency
            load_control: Load control
            speed_control: Speed control
            overall_stability: Overall stability

        Returns:
            FeatureMetrics model

        Raises:
            Exception: If loading fails
        """
        return FeatureMetrics(
            body_alignment=body_alignment,
            joint_consistency=joint_consistency,
            load_control=load_control,
            speed_control=speed_control,
            overall_stability=overall_stability,
        )

    def format_date(self, date: str) -> str:
        """
        Format a date string.

        Args:
            date: Date string to format

        Returns:
            Formatted date string
        """
        exercise_datetime = date.split(".")[0]
        return exercise_datetime.replace(":", "-")

    async def save_data(
        self, username: str, exercise_name: str, feature_metrics: Dict[str, Any]
    ) -> None:
        """
        Save feature metrics to the database.

        Args:
            username: User's username
            exercise_name: Exercise name
            feature_metrics: Feature metrics to save
            db: NTFInterface instance

        Raises:
            Exception: If saving fails
        """
        try:
            await self.data_repository.set_data(
                key=f"metrics/{username}/{exercise_name.lower()}",
                value=feature_metrics,
            )
        except Exception as e:
            logger.error(f"Failed to save feature metrics: {str(e)}")
            raise
