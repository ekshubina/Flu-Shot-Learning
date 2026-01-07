"""
Data validation utilities for the ML pipeline.

Provides helper functions for validating features, labels, and predictions
at various pipeline stages. Enables early detection of data issues.

Validation includes:
- Feature shape and dtype validation
- Label value and class balance checks
- Prediction probability range validation
- Missing value detection

Reference: SYSTEM_DESIGN.md - Component 9: Utilities
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def validate_features(
    X: pd.DataFrame,
    expected_shape: Optional[Tuple[int, int]] = None,
    expected_columns: Optional[list] = None,
) -> bool:
    """
    Validate feature matrix shape, columns, and data types.
    
    Checks:
    - X is pandas DataFrame or numpy array
    - Shape is valid (not empty, correct dimensions)
    - If expected_shape provided, matches expected
    - If expected_columns provided, all columns present and correct order
    - No all-NaN columns
    - Numeric columns are numeric dtype
    
    Parameters:
        X (pd.DataFrame): Feature matrix to validate
        expected_shape (Optional[Tuple[int, int]]): Expected shape (n_samples, n_features)
            If None, only checks shape is 2D and not empty
        expected_columns (Optional[list]): Expected column names in order
            If None, only checks columns exist
    
    Returns:
        bool: True if validation passes
    
    Raises:
        ValueError: If validation fails, with descriptive message
        
    Implementation notes:
        - TODO: Check X is DataFrame
        - TODO: Check shape is 2D and not empty
        - TODO: If expected_shape provided, check matches
        - TODO: If expected_columns provided, check present and in order
        - TODO: Check no all-NaN columns
        - TODO: Raise ValueError with clear message on failure
        - TODO: Return True if all checks pass
    """
    # TODO: Implement
    raise NotImplementedError("validate_features() not yet implemented")


def validate_labels(
    y: np.ndarray,
    num_classes: int = 2,
    class_values: Optional[list] = None,
) -> bool:
    """
    Validate binary/multiclass labels.
    
    Checks:
    - y is numpy array (1D)
    - Shape is (n_samples,)
    - Contains only expected class values (0 and 1 for binary)
    - No missing values
    - Sufficient samples per class (at least 1 sample per class)
    - Class balance warning if highly imbalanced
    
    Parameters:
        y (np.ndarray): Label array to validate
        num_classes (int): Expected number of classes (2 for binary). Default: 2
        class_values (Optional[list]): Expected class values. Default: [0, 1]
    
    Returns:
        bool: True if validation passes
    
    Raises:
        ValueError: If validation fails
        Warning: If class imbalance detected (via logger)
        
    Implementation notes:
        - TODO: Check y is 1D numpy array
        - TODO: Check no missing values
        - TODO: If class_values not provided, default to [0, 1]
        - TODO: Check all values in class_values
        - TODO: Check at least 1 sample per class
        - TODO: Warn if class imbalance > 0.7 or < 0.3 prevalence
        - TODO: Raise ValueError on failure
        - TODO: Return True if all checks pass
    """
    # TODO: Implement
    raise NotImplementedError("validate_labels() not yet implemented")


def validate_predictions(
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    prob_range: Tuple[float, float] = (0.0, 1.0),
) -> bool:
    """
    Validate predicted probabilities or class labels.
    
    Checks:
    - y_pred is numpy array (1D)
    - Shape matches y_true if provided
    - If prob_range specified, all values in range (e.g., [0, 1] for probabilities)
    - No NaN or infinite values
    - For probabilities, values are between 0 and 1
    
    Parameters:
        y_pred (np.ndarray): Predictions to validate (1D array)
        y_true (Optional[np.ndarray]): True labels for shape validation
        prob_range (Tuple[float, float]): Expected range for probability values.
            Default: (0.0, 1.0)
    
    Returns:
        bool: True if validation passes
    
    Raises:
        ValueError: If validation fails with descriptive message
        
    Implementation notes:
        - TODO: Check y_pred is 1D numpy array
        - TODO: Check shape is (n_samples,)
        - TODO: If y_true provided, check shapes match
        - TODO: Check no NaN or infinite values
        - TODO: Check all values in prob_range
        - TODO: Raise ValueError on failure with clear message
        - TODO: Return True if all checks pass
    """
    # TODO: Implement
    raise NotImplementedError("validate_predictions() not yet implemented")


def check_missing_values(
    X: pd.DataFrame,
    verbose: bool = True,
) -> pd.Series:
    """
    Check and report missing values in feature matrix.
    
    Computes missing value counts and percentages per column.
    
    Parameters:
        X (pd.DataFrame): Feature matrix
        verbose (bool): If True, print summary. Default: True
    
    Returns:
        pd.Series: Missing value counts per column (sorted descending)
        
    Example:
        ```python
        missing = check_missing_values(X_train)
        # Output:
        # employment_industry    2843
        # employment_occupation  2843
        # ...
        ```
    
    Implementation notes:
        - TODO: Count missing values per column
        - TODO: Sort by count descending
        - TODO: If verbose, print summary with percentages
        - TODO: Return Series with column names and missing counts
    """
    # TODO: Implement
    raise NotImplementedError("check_missing_values() not yet implemented")


def report_class_imbalance(
    y: np.ndarray,
    vaccine_name: str = "Target",
) -> dict:
    """
    Report class balance information for binary target.
    
    Computes proportion of positive class, prevalence ratio, etc.
    
    Parameters:
        y (np.ndarray): Binary labels (0 or 1)
        vaccine_name (str): Name of target (for reporting). Default: "Target"
    
    Returns:
        dict: Class balance metrics:
            - 'negative_count': Count of 0 labels
            - 'positive_count': Count of 1 labels
            - 'negative_proportion': Proportion of 0 labels
            - 'positive_proportion': Proportion of 1 labels
            - 'prevalence': Proportion of positive class
            - 'imbalance_ratio': (majority / minority) class count ratio
    
    Implementation notes:
        - TODO: Count positive and negative samples
        - TODO: Compute proportions
        - TODO: Compute imbalance ratio
        - TODO: Return dictionary
    """
    # TODO: Implement
    raise NotImplementedError("report_class_imbalance() not yet implemented")
