"""
Metric computation utilities for the ML pipeline.

Provides helper functions for computing common evaluation metrics.
These are used by the Evaluator class and for experiment tracking.

Metrics include:
- AUROC (Area Under ROC Curve)
- Calibration error (ECE, MCE, Brier score)
- Per-vaccine metrics and aggregates

Reference: SYSTEM_DESIGN.md - Component 9: Utilities
"""

from typing import Dict, Tuple
import numpy as np


def compute_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute ROC AUC score for binary classification.
    
    Parameters:
        y_true (np.ndarray): True binary labels (0 or 1)
        y_pred (np.ndarray): Predicted probabilities in [0, 1]
    
    Returns:
        float: AUC score in [0.5, 1.0] (0.5 = random, 1.0 = perfect)
    
    Implementation notes:
        - TODO: Use sklearn.metrics.roc_auc_score()
        - TODO: Validate inputs
        - TODO: Return AUC score
    """
    # TODO: Implement
    raise NotImplementedError("compute_auroc() not yet implemented")


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Measures mean absolute difference between predicted probability
    and empirical frequency across bins.
    
    Parameters:
        y_true (np.ndarray): True binary labels
        y_pred (np.ndarray): Predicted probabilities
        n_bins (int): Number of bins. Default: 10
    
    Returns:
        float: ECE score in [0, 1] (lower is better)
    
    Implementation notes:
        - TODO: Bin predictions
        - TODO: For each bin, compute confidence and accuracy
        - TODO: Weight by bin size
        - TODO: Return weighted average difference
    """
    # TODO: Implement
    raise NotImplementedError("compute_expected_calibration_error() not yet implemented")


def compute_brier_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Brier Score.
    
    Mean squared difference between predicted probability and true label.
    
    Parameters:
        y_true (np.ndarray): True binary labels (0 or 1)
        y_pred (np.ndarray): Predicted probabilities [0, 1]
    
    Returns:
        float: Brier score in [0, 1] (lower is better)
    
    Implementation notes:
        - TODO: Compute mean((y_pred - y_true)^2)
        - TODO: Return float
    """
    # TODO: Implement
    raise NotImplementedError("compute_brier_score() not yet implemented")


def compute_per_vaccine_auroc(
    y_true_h1n1: np.ndarray,
    y_true_seasonal: np.ndarray,
    y_pred_h1n1: np.ndarray,
    y_pred_seasonal: np.ndarray,
) -> Dict[str, float]:
    """
    Compute AUC for both vaccines and their mean.
    
    Competition evaluation metric is mean(AUC_h1n1, AUC_seasonal).
    
    Parameters:
        y_true_h1n1: True H1N1 labels
        y_true_seasonal: True seasonal labels
        y_pred_h1n1: Predicted H1N1 probabilities
        y_pred_seasonal: Predicted seasonal probabilities
    
    Returns:
        Dict with keys:
            - 'h1n1': AUC for H1N1
            - 'seasonal': AUC for seasonal
            - 'mean': Mean AUC (competition metric)
    
    Implementation notes:
        - TODO: Compute AUC for each vaccine
        - TODO: Compute mean
        - TODO: Return dictionary
    """
    # TODO: Implement
    raise NotImplementedError("compute_per_vaccine_auroc() not yet implemented")


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute sensitivity (TPR) and specificity (TNR) at threshold.
    
    Parameters:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        threshold (float): Classification threshold. Default: 0.5
    
    Returns:
        Tuple[float, float]: (sensitivity, specificity)
    
    Implementation notes:
        - TODO: Classify at threshold
        - TODO: Compute TP, FP, TN, FN
        - TODO: Compute TPR = TP/(TP+FN)
        - TODO: Compute TNR = TN/(TN+FP)
        - TODO: Handle divide by zero
        - TODO: Return tuple
    """
    # TODO: Implement
    raise NotImplementedError("compute_sensitivity_specificity() not yet implemented")


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray = np.arange(0.0, 1.01, 0.1),
) -> Dict[str, list]:
    """
    Compute metrics at multiple thresholds.
    
    Useful for threshold optimization and understanding performance-specificity
    tradeoff.
    
    Parameters:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        thresholds (np.ndarray): Thresholds to evaluate. Default: 0.0 to 1.0 by 0.1
    
    Returns:
        Dict with lists of metrics per threshold:
            - 'thresholds': The evaluated thresholds
            - 'sensitivity': Sensitivity at each threshold
            - 'specificity': Specificity at each threshold
            - 'ppv': Positive Predictive Value at each threshold
            - 'npv': Negative Predictive Value at each threshold
    
    Implementation notes:
        - TODO: For each threshold, compute all metrics
        - TODO: Store in lists
        - TODO: Return dictionary with lists
    """
    # TODO: Implement
    raise NotImplementedError("compute_threshold_metrics() not yet implemented")


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'f1',
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Searches for threshold that maximizes specified metric.
    
    Parameters:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        metric (str): Metric to optimize ('f1', 'accuracy', 'balanced_accuracy')
            Default: 'f1'
    
    Returns:
        Tuple[float, float]: (optimal_threshold, metric_value)
    
    Implementation notes:
        - TODO: Try multiple thresholds (0.0 to 1.0)
        - TODO: For each, compute specified metric
        - TODO: Find threshold with maximum metric
        - TODO: Return threshold and metric value
    """
    # TODO: Implement
    raise NotImplementedError("find_optimal_threshold() not yet implemented")
