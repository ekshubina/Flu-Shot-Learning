"""
Plotting utility functions for the ML pipeline.

Provides wrapper functions for common visualizations.
These are simpler versions than the evaluation/plots.py functions,
used throughout the pipeline for quick diagnostics.

Includes:
- Confusion matrix heatmaps
- Class distribution histograms
- Feature correlation matrices
- Model comparison plots

Reference: SYSTEM_DESIGN.md - Component 9: Utilities
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(
    y: np.ndarray,
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Plot histogram of class distribution.
    
    Parameters:
        y (np.ndarray): Binary labels (0 or 1)
        title (str): Plot title. Default: "Class Distribution"
        figsize (Tuple[int, int]): Figure size. Default: (8, 5)
    
    Returns:
        plt.Figure: Matplotlib figure object
    
    Implementation notes:
        - TODO: Count 0s and 1s
        - TODO: Create bar plot
        - TODO: Add counts and percentages on bars
        - TODO: Set title and labels
        - TODO: Return figure
    """
    # TODO: Implement
    raise NotImplementedError("plot_class_distribution() not yet implemented")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Parameters:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        threshold (float): Classification threshold. Default: 0.5
        title (str): Plot title. Default: "Confusion Matrix"
        figsize (Tuple[int, int]): Figure size. Default: (6, 5)
    
    Returns:
        plt.Figure: Matplotlib figure object
    
    Implementation notes:
        - TODO: Apply threshold to get binary predictions
        - TODO: Compute confusion matrix
        - TODO: Create heatmap with counts and percentages
        - TODO: Add labels and title
        - TODO: Return figure
    """
    # TODO: Implement
    raise NotImplementedError("plot_confusion_matrix() not yet implemented")


def plot_feature_distribution(
    X: pd.DataFrame,
    feature_name: str,
    y: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot distribution of a single feature.
    
    If y provided, separate distributions by class.
    
    Parameters:
        X (pd.DataFrame): Feature matrix
        feature_name (str): Name of feature to plot
        y (Optional[np.ndarray]): Binary labels for class separation
        title (Optional[str]): Plot title. If None, use feature name
        figsize (Tuple[int, int]): Figure size. Default: (10, 5)
    
    Returns:
        plt.Figure: Matplotlib figure object
    
    Implementation notes:
        - TODO: Extract feature column
        - TODO: If y provided, separate by class
        - TODO: Create histogram or KDE plot
        - TODO: Add title and labels
        - TODO: Return figure
    """
    # TODO: Implement
    raise NotImplementedError("plot_feature_distribution() not yet implemented")


def plot_feature_correlation(
    X: pd.DataFrame,
    max_features: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Feature Correlation Matrix",
) -> plt.Figure:
    """
    Plot correlation heatmap of features.
    
    Useful for identifying multicollinearity and feature relationships.
    
    Parameters:
        X (pd.DataFrame): Feature matrix
        max_features (int): Limit to top N features. Default: 20
        figsize (Tuple[int, int]): Figure size. Default: (10, 8)
        title (str): Plot title. Default: "Feature Correlation Matrix"
    
    Returns:
        plt.Figure: Matplotlib figure object
    
    Implementation notes:
        - TODO: Select top max_features features (by variance or first N)
        - TODO: Compute correlation matrix
        - TODO: Create heatmap with seaborn
        - TODO: Add colorbar and labels
        - TODO: Return figure
    """
    # TODO: Implement
    raise NotImplementedError("plot_feature_correlation() not yet implemented")


def plot_roc_curve_simple(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Simple ROC curve plot (single vaccine).
    
    Parameters:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        title (str): Plot title. Default: "ROC Curve"
        figsize (Tuple[int, int]): Figure size. Default: (8, 6)
    
    Returns:
        plt.Figure: Matplotlib figure object
    
    Implementation notes:
        - TODO: Compute ROC curve
        - TODO: Plot ROC line and diagonal
        - TODO: Add AUC to legend
        - TODO: Set labels and title
        - TODO: Return figure
    """
    # TODO: Implement
    raise NotImplementedError("plot_roc_curve_simple() not yet implemented")


def create_comparison_table(
    results_dict: dict,
) -> pd.DataFrame:
    """
    Create comparison table from results dictionary.
    
    Useful for comparing multiple model runs.
    
    Parameters:
        results_dict (dict): Dictionary with run names as keys,
            metric dicts as values
    
    Returns:
        pd.DataFrame: Comparison table with runs as rows, metrics as columns
    
    Implementation notes:
        - TODO: Convert dict to DataFrame
        - TODO: Sort by relevant metric (e.g., auroc_mean)
        - TODO: Format numbers to 4 decimal places
        - TODO: Return DataFrame
    """
    # TODO: Implement
    raise NotImplementedError("create_comparison_table() not yet implemented")
