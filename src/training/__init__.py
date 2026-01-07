"""
Training module for model optimization and validation.

This module provides the training engine for cross-validation, hyperparameter
optimization, and model validation workflows.

Classes:
    TrainingEngine: Orchestrates model training and validation
    FoldResults: Results from a single CV fold
    CVResults: Results from complete CV run

See docs/SYSTEM_DESIGN.md for detailed component architecture.
"""

from src.training.engine import (
    TrainingEngine,
    FoldResults,
    CVResults,
)

__all__ = [
    "TrainingEngine",
    "FoldResults",
    "CVResults",
]
