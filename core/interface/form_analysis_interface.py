from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

from core.entities.pose_entity import PoseFeatures, FormAnalysis


class FormAnalysisInterface(ABC):
    @abstractmethod
    def analyze_bench_press_form(
        self, features: PoseFeatures, exercise_name: str
    ) -> FormAnalysis:
        """
        Analyze bench press form using pose features.

        Args:
            features: Extracted pose features
            exercise_name: Name of the exercise variant

        Returns:
            FormAnalysis with accuracy score and suggestions
        """
        pass

    @abstractmethod
    def analyze_deadlift_form(
        self, features: PoseFeatures, exercise_name: str
    ) -> FormAnalysis:
        """
        Analyze deadlift form using pose features.

        Args:
            features: Extracted pose features
            exercise_name: Name of the exercise variant

        Returns:
            FormAnalysis with accuracy score and suggestions
        """
        pass

    @abstractmethod
    def analyze_rdl_form(
        self, features: PoseFeatures, exercise_name: str
    ) -> FormAnalysis:
        """
        Analyze Romanian deadlift form using pose features.

        Args:
            features: Extracted pose features
            exercise_name: Name of the exercise variant

        Returns:
            FormAnalysis with accuracy score and suggestions
        """
        pass

    @abstractmethod
    def analyze_shoulder_press_form(
        self, features: PoseFeatures, exercise_name: str
    ) -> FormAnalysis:
        """
        Analyze shoulder press form using pose features.

        Args:
            features: Extracted pose features
            exercise_name: Name of the exercise variant

        Returns:
            FormAnalysis with accuracy score and suggestions
        """
        pass

    @abstractmethod
    def analyze_form(self, features: PoseFeatures, exercise_name: str) -> FormAnalysis:
        """
        Calculate form accuracy and provide feedback using pose features.

        Args:
            features: Extracted pose features
            exercise_name: Name of the exercise being performed

        Returns:
            FormAnalysis with accuracy score and suggestions
        """
        pass
