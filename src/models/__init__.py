"""
Models module for machine learning implementations.

This module provides abstract base classes and concrete implementations for
various machine learning models (logistic regression, tree-based, ensemble methods).

Classes:
    BaseModel: Abstract base class for model implementations
    LogisticRegressionModel: Linear logistic regression model
    XGBoostModel: XGBoost gradient boosting model
    LightGBMModel: LightGBM gradient boosting model
    RandomForestModel: Random Forest ensemble model
    ModelFactory: Factory for creating model instances by type

See docs/SYSTEM_DESIGN.md for detailed component architecture.
"""

from src.models.factory import (
    BaseModel,
    LogisticRegressionModel,
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    ModelFactory,
)

__all__ = [
    "BaseModel",
    "LogisticRegressionModel",
    "XGBoostModel",
    "LightGBMModel",
    "RandomForestModel",
    "ModelFactory",
]
