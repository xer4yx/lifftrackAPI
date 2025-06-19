from typing import Dict, Any, Optional
from core.entities import PoseFeatures, ExerciseData
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
