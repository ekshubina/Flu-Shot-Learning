"""
Evaluation module for metrics computation and visualization.

This module provides tools for evaluating model performance including metric
computation, visualization of results, and diagnostics.

Classes:
    Evaluator: Computes and reports evaluation metrics

Functions:
    plot_roc_curves: Plot ROC curves for predictions
    plot_calibration_curve: Plot calibration diagnostics
    plot_feature_importance: Plot feature importance
    plot_prediction_confidence: Plot prediction confidence distribution

See docs/SYSTEM_DESIGN.md for detailed component architecture.
"""

__all__ = [
    "Evaluator",
]
