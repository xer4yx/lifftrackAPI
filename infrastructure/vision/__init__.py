"""
ML Models
- Implements machine learning model-specific logic
- Provides concrete implementations of ML model interfaces
- Handles model loading, training, and inference
- Integrates with specific ML frameworks (TensorFlow, PyTorch)
"""

from .cnn.model import CNNExerciseClassifier

__all__ = ['CNNExerciseClassifier']

"""
Vision module for computer vision related implementations
"""
