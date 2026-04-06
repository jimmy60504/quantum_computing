"""Model definitions for HW1 Problem 3."""

from .baseline import CNNMLPClassifier
from .hybrid import CNNQNNClassifier

__all__ = ["CNNMLPClassifier", "CNNQNNClassifier"]
