"""
Visualization functions for model evaluation and diagnostics.

This module provides plotting functions for visualizing:
- ROC curves (one per vaccine)
- Calibration curves (reliability diagrams)
- Feature importance / feature contributions
- Prediction confidence distributions

All functions use matplotlib and seaborn for consistent styling.

Reference: SYSTEM_DESIGN.md - Component 7: Evaluation
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_roc_curves(
    y_true_h1n1: np.ndarray,
    y_true_seasonal: np.ndarray,
    y_pred_h1n1: np.ndarray,
    y_pred_seasonal: np.ndarray,
    auroc_h1n1: Optional[float] = None,
    auroc_seasonal: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 5),
    title: str = "ROC Curves - Flu Vaccine Prediction",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curves for both H1N1 and seasonal vaccines.
    
    The ROC (Receiver Operating Characteristic) curve plots the true positive rate
    (sensitivity) against the false positive rate (1 - specificity) across all
    possible classification thresholds. A good classifier's ROC curve bows toward
    the top-left corner.
    
    Parameters:
        y_true_h1n1 (np.ndarray): True labels for H1N1 vaccine (n_samples,)
        y_true_seasonal (np.ndarray): True labels for seasonal vaccine (n_samples,)
        y_pred_h1n1 (np.ndarray): Predicted probabilities for H1N1 (n_samples,)
        y_pred_seasonal (np.ndarray): Predicted probabilities for seasonal (n_samples,)
        auroc_h1n1 (Optional[float]): Pre-computed AUC for H1N1 (for display)
        auroc_seasonal (Optional[float]): Pre-computed AUC for seasonal (for display)
        figsize (Tuple[int, int]): Figure size (width, height). Default: (12, 5)
        title (str): Figure title. Default: "ROC Curves - Flu Vaccine Prediction"
        save_path (Optional[str]): If provided, save figure to this path
    
    Returns:
        plt.Figure: Matplotlib figure object with two subplots (one per vaccine)
    
    Implementation notes:
        - TODO: Compute ROC curve points using sklearn.metrics.roc_curve()
        - TODO: Create figure with 1x2 subplots (H1N1 and seasonal side-by-side)
        - TODO: Plot ROC curve, diagonal chance line, and AUC annotation
        - TODO: Label axes clearly (FPR, TPR) and add legend
        - TODO: If auroc_h1n1/seasonal provided, display in title
        - TODO: If save_path provided, save figure
        - TODO: Return figure object for further manipulation
    """
    from sklearn.metrics import roc_curve, auc
    
    # Compute ROC curves
    fpr_h1n1, tpr_h1n1, _ = roc_curve(y_true_h1n1, y_pred_h1n1)
    fpr_seasonal, tpr_seasonal, _ = roc_curve(y_true_seasonal, y_pred_seasonal)
    
    # Compute AUC if not provided
    if auroc_h1n1 is None:
        auroc_h1n1 = auc(fpr_h1n1, tpr_h1n1)
    if auroc_seasonal is None:
        auroc_seasonal = auc(fpr_seasonal, tpr_seasonal)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot H1N1 ROC curve
    axes[0].plot(fpr_h1n1, tpr_h1n1, color='#1f77b4', lw=2,
                 label=f'ROC Curve (AUC = {auroc_h1n1:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title('H1N1 Vaccine', fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([-0.01, 1.01])
    axes[0].set_ylim([-0.01, 1.01])
    
    # Plot Seasonal ROC curve
    axes[1].plot(fpr_seasonal, tpr_seasonal, color='#ff7f0e', lw=2,
                 label=f'ROC Curve (AUC = {auroc_seasonal:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1].set_title('Seasonal Vaccine', fontsize=12, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([-0.01, 1.01])
    axes[1].set_ylim([-0.01, 1.01])
    
    # Set overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    vaccine_name: str = "Vaccine",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot calibration (reliability) diagram for predicted probabilities.
    
    A calibration curve shows the relationship between predicted probability
    and empirical frequency of the positive class. A well-calibrated model
    has points close to the diagonal line (perfect calibration).
    
    The x-axis shows the mean predicted probability in each bin, and the y-axis
    shows the empirical frequency of positive examples in that bin.
    
    Parameters:
        y_true (np.ndarray): True binary labels (n_samples,)
        y_pred (np.ndarray): Predicted probabilities (n_samples,) in [0, 1]
        n_bins (int): Number of bins for grouping predictions. Default: 10
        vaccine_name (str): Name of vaccine for title. Default: "Vaccine"
        figsize (Tuple[int, int]): Figure size (width, height). Default: (8, 6)
        save_path (Optional[str]): If provided, save figure to this path
    
    Returns:
        plt.Figure: Matplotlib figure object with calibration curve
    
    Implementation notes:
        - TODO: Bin predictions into n_bins groups
        - TODO: For each bin, compute mean predicted probability and empirical frequency
        - TODO: Create scatter plot of (mean_pred, empirical_freq)
        - TODO: Overlay perfect calibration diagonal line
        - TODO: Add histogram of prediction distribution as secondary plot
        - TODO: Label axes clearly and add legend
        - TODO: If save_path provided, save figure
        - TODO: Return figure object
    """
    # Validate inputs
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred shapes don't match")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only 0 or 1")
    if not np.all((y_pred >= 0) & (y_pred <= 1)):
        raise ValueError("y_pred must be in [0, 1]")
    
    # Create figure with GridSpec for histogram
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])
    
    # Compute calibration curve points
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            mask = (y_pred >= bin_lower) & (y_pred <= bin_upper)
        else:
            mask = (y_pred >= bin_lower) & (y_pred < bin_upper)
        
        if np.any(mask):
            bin_centers.append(np.mean(y_pred[mask]))
            bin_accuracies.append(np.mean(y_true[mask]))
            bin_counts.append(np.sum(mask))
    
    bin_centers = np.array(bin_centers)
    bin_accuracies = np.array(bin_accuracies)
    bin_counts = np.array(bin_counts)
    
    # Plot calibration curve
    ax_main.scatter(bin_centers, bin_accuracies, s=bin_counts*2, alpha=0.6,
                    color='#1f77b4', edgecolors='black', linewidth=1,
                    label='Calibration Curve')
    
    # Plot perfect calibration line
    ax_main.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
    
    # Labels and formatting
    ax_main.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax_main.set_ylabel('Empirical Frequency', fontsize=11)
    ax_main.set_title(f'{vaccine_name} - Calibration Curve', fontsize=12, fontweight='bold')
    ax_main.legend(loc='upper left', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim([-0.05, 1.05])
    ax_main.set_ylim([-0.05, 1.05])
    
    # Plot histogram of predictions
    ax_hist.hist(y_pred, bins=30, alpha=0.6, color='#1f77b4', edgecolor='black')
    ax_hist.set_xlabel('Predicted Probability', fontsize=10)
    ax_hist.set_ylabel('Frequency', fontsize=10)
    ax_hist.grid(True, alpha=0.3, axis='y')
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_calibration_curves(
    y_true_h1n1: np.ndarray,
    y_true_seasonal: np.ndarray,
    y_pred_h1n1: np.ndarray,
    y_pred_seasonal: np.ndarray,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (14, 5),
    title: str = "Calibration Curves - Flu Vaccine Prediction",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot calibration (reliability) diagrams for both H1N1 and seasonal vaccines.
    
    A calibration curve shows the relationship between predicted probability
    and empirical frequency of the positive class. A well-calibrated model
    has points close to the diagonal line (perfect calibration).
    
    This function creates a 1x2 subplot figure with:
    - Left: H1N1 vaccine calibration curve
    - Right: Seasonal vaccine calibration curve
    
    Each calibration curve shows:
    - Scatter plot of (mean predicted probability, empirical frequency) per bin
    - Perfect calibration diagonal (y=x) as reference line
    - Histogram of prediction distribution below
    
    Parameters:
        y_true_h1n1 (np.ndarray): True H1N1 labels (n_samples,)
        y_true_seasonal (np.ndarray): True seasonal labels (n_samples,)
        y_pred_h1n1 (np.ndarray): Predicted probabilities for H1N1 (n_samples,) in [0, 1]
        y_pred_seasonal (np.ndarray): Predicted probabilities for seasonal (n_samples,) in [0, 1]
        n_bins (int): Number of bins for grouping predictions. Default: 10
        figsize (Tuple[int, int]): Figure size (width, height). Default: (14, 5)
        title (str): Figure title. Default: "Calibration Curves - Flu Vaccine Prediction"
        save_path (Optional[str]): If provided, save figure to this path
    
    Returns:
        plt.Figure: Matplotlib figure object with two calibration curves
    
    Implementation notes:
        - Bin predictions into n_bins equal-width bins (0-0.1, 0.1-0.2, etc.)
        - For each bin, compute mean predicted probability (X-axis) and empirical frequency (Y-axis)
        - Plot scatter points (size proportional to bin sample count) and diagonal reference line
        - Include histogram of prediction distribution as secondary plot
        - Two subplots: H1N1 and seasonal side-by-side
        - Validates input arrays have matching shapes and valid value ranges
        - Returns figure object for further manipulation
    """
    # Validate inputs
    if y_true_h1n1.shape != y_pred_h1n1.shape:
        raise ValueError("y_true_h1n1 and y_pred_h1n1 shapes don't match")
    if y_true_seasonal.shape != y_pred_seasonal.shape:
        raise ValueError("y_true_seasonal and y_pred_seasonal shapes don't match")
    if y_true_h1n1.shape != y_true_seasonal.shape:
        raise ValueError("y_true_h1n1 and y_true_seasonal shapes don't match")
    if not np.all((y_true_h1n1 == 0) | (y_true_h1n1 == 1)):
        raise ValueError("y_true_h1n1 must contain only 0 or 1")
    if not np.all((y_true_seasonal == 0) | (y_true_seasonal == 1)):
        raise ValueError("y_true_seasonal must contain only 0 or 1")
    if not np.all((y_pred_h1n1 >= 0) & (y_pred_h1n1 <= 1)):
        raise ValueError("y_pred_h1n1 must be in [0, 1]")
    if not np.all((y_pred_seasonal >= 0) & (y_pred_seasonal <= 1)):
        raise ValueError("y_pred_seasonal must be in [0, 1]")
    
    # Create figure with two subplots for H1N1 and seasonal
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Helper function to compute and plot calibration curve for one vaccine
    def plot_vaccine_calibration(ax, y_true, y_pred, vaccine_name):
        # Compute calibration curve points
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Last bin includes upper boundary
            if i == n_bins - 1:
                mask = (y_pred >= bin_lower) & (y_pred <= bin_upper)
            else:
                mask = (y_pred >= bin_lower) & (y_pred < bin_upper)
            
            # Only include non-empty bins
            if np.any(mask):
                bin_centers.append(np.mean(y_pred[mask]))
                bin_accuracies.append(np.mean(y_true[mask]))
                bin_counts.append(np.sum(mask))
        
        bin_centers = np.array(bin_centers)
        bin_accuracies = np.array(bin_accuracies)
        bin_counts = np.array(bin_counts)
        
        # Plot calibration curve
        ax.scatter(bin_centers, bin_accuracies, s=bin_counts*2, alpha=0.6,
                   color='#1f77b4', edgecolors='black', linewidth=1,
                   label='Calibration Curve')
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
        
        # Labels and formatting
        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Empirical Frequency', fontsize=11)
        ax.set_title(f'{vaccine_name} - Calibration Curve', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
    
    # Plot calibration curves for both vaccines
    plot_vaccine_calibration(axes[0], y_true_h1n1, y_pred_h1n1, 'H1N1 Vaccine')
    plot_vaccine_calibration(axes[1], y_true_seasonal, y_pred_seasonal, 'Seasonal Vaccine')
    
    # Set overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: np.ndarray,
    importances: np.ndarray,
    top_k: int = 20,
    vaccine_name: str = "Vaccine",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature importance / contribution scores.
    
    Displays the top k most important features based on their importance scores
    from the model. Importance scores could be from:
    - Tree-based models: feature_importances_
    - Linear models: absolute values of coefficients
    - SHAP values: mean absolute SHAP values
    - Permutation importance: change in model performance
    
    Parameters:
        feature_names (np.ndarray): Names of features (n_features,)
        importances (np.ndarray): Importance scores (n_features,)
        top_k (int): Display top k features. Default: 20
        vaccine_name (str): Name of vaccine for title. Default: "Vaccine"
        figsize (Tuple[int, int]): Figure size (width, height). Default: (10, 8)
        save_path (Optional[str]): If provided, save figure to this path
    
    Returns:
        plt.Figure: Matplotlib figure object with feature importance plot
    
    Implementation notes:
        - TODO: Validate inputs (feature_names and importances same length)
        - TODO: Sort features by importance score (descending)
        - TODO: Select top k features
        - TODO: Create horizontal bar plot with feature names and scores
        - TODO: Color bars by importance level (gradient from low to high)
        - TODO: Add value labels on bars
        - TODO: Label axes and add title
        - TODO: If save_path provided, save figure
        - TODO: Return figure object
    """
    # Validate inputs
    if len(feature_names) != len(importances):
        raise ValueError(
            f"feature_names and importances length mismatch: "
            f"{len(feature_names)} vs {len(importances)}"
        )
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = feature_names[sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    # Select top k
    top_idx = slice(0, min(top_k, len(sorted_names)))
    plot_names = sorted_names[top_idx]
    plot_importances = sorted_importances[top_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_importances)))
    bars = ax.barh(range(len(plot_importances)), plot_importances, color=colors,
                   edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, plot_importances)):
        ax.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', va='center', fontsize=9)
    
    # Formatting
    ax.set_yticks(range(len(plot_importances)))
    ax.set_yticklabels(plot_names, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(f'{vaccine_name} - Top {len(plot_importances)} Feature Importances',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    fig.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_prediction_confidence(
    y_pred_h1n1: np.ndarray,
    y_pred_seasonal: np.ndarray,
    y_true_h1n1: Optional[np.ndarray] = None,
    y_true_seasonal: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 5),
    title: str = "Prediction Confidence Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot histogram of predicted probabilities (confidence distribution).
    
    Shows the distribution of predicted probabilities for both vaccines.
    If true labels are provided, histograms can be overlaid by class
    to visualize whether the model is confident in its predictions.
    
    Parameters:
        y_pred_h1n1 (np.ndarray): Predicted probabilities for H1N1 (n_samples,)
        y_pred_seasonal (np.ndarray): Predicted probabilities for seasonal (n_samples,)
        y_true_h1n1 (Optional[np.ndarray]): True H1N1 labels for separation by class
        y_true_seasonal (Optional[np.ndarray]): True seasonal labels for separation by class
        figsize (Tuple[int, int]): Figure size (width, height). Default: (12, 5)
        title (str): Figure title. Default: "Prediction Confidence Distribution"
        save_path (Optional[str]): If provided, save figure to this path
    
    Returns:
        plt.Figure: Matplotlib figure object with confidence histograms
    
    Implementation notes:
        - TODO: Create figure with 1x2 subplots (H1N1 and seasonal side-by-side)
        - TODO: If y_true provided, separate histograms by class (0 vs 1)
        - TODO: Plot histograms with 30 bins and semi-transparent colors
        - TODO: Add labels showing count of predictions per class
        - TODO: Add vertical lines for mean predictions
        - TODO: Add legend indicating class (if y_true provided)
        - TODO: If save_path provided, save figure
        - TODO: Return figure object
    """
    # Validate inputs
    if y_pred_h1n1.shape != y_pred_seasonal.shape:
        raise ValueError("y_pred_h1n1 and y_pred_seasonal shapes don't match")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot H1N1 confidence distribution
    if y_true_h1n1 is not None:
        # Separate by class
        neg_mask = y_true_h1n1 == 0
        pos_mask = y_true_h1n1 == 1
        
        axes[0].hist(y_pred_h1n1[neg_mask], bins=30, alpha=0.6, color='#1f77b4',
                     label=f'No Vaccine (n={np.sum(neg_mask)})', edgecolor='black')
        axes[0].hist(y_pred_h1n1[pos_mask], bins=30, alpha=0.6, color='#ff7f0e',
                     label=f'Received Vaccine (n={np.sum(pos_mask)})', edgecolor='black')
    else:
        axes[0].hist(y_pred_h1n1, bins=30, alpha=0.6, color='#1f77b4',
                     edgecolor='black', label=f'All predictions (n={len(y_pred_h1n1)})')
    
    axes[0].axvline(np.mean(y_pred_h1n1), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {np.mean(y_pred_h1n1):.3f}')
    axes[0].set_xlabel('Predicted Probability', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('H1N1 Vaccine', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot Seasonal confidence distribution
    if y_true_seasonal is not None:
        # Separate by class
        neg_mask = y_true_seasonal == 0
        pos_mask = y_true_seasonal == 1
        
        axes[1].hist(y_pred_seasonal[neg_mask], bins=30, alpha=0.6, color='#1f77b4',
                     label=f'No Vaccine (n={np.sum(neg_mask)})', edgecolor='black')
        axes[1].hist(y_pred_seasonal[pos_mask], bins=30, alpha=0.6, color='#ff7f0e',
                     label=f'Received Vaccine (n={np.sum(pos_mask)})', edgecolor='black')
    else:
        axes[1].hist(y_pred_seasonal, bins=30, alpha=0.6, color='#ff7f0e',
                     edgecolor='black', label=f'All predictions (n={len(y_pred_seasonal)})')
    
    axes[1].axvline(np.mean(y_pred_seasonal), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {np.mean(y_pred_seasonal):.3f}')
    axes[1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Seasonal Vaccine', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Set overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confidence_distribution(
    y_pred_h1n1: np.ndarray,
    y_pred_seasonal: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    title: str = "Confidence Distribution - Predicted Probabilities",
) -> plt.Figure:
    """
    Plot histogram of predicted probabilities (confidence distribution).
    
    Displays the distribution of predicted probabilities for both H1N1 and seasonal
    vaccines in a single figure with two subplots. This shows how confident the model
    is in its predictions across all samples.
    
    Parameters:
        y_pred_h1n1 (np.ndarray): Predicted probabilities for H1N1 (n_samples,) in [0.0, 1.0]
        y_pred_seasonal (np.ndarray): Predicted probabilities for seasonal (n_samples,) in [0.0, 1.0]
        output_path (Optional[str]): If provided, save figure to this path
        figsize (Tuple[int, int]): Figure size (width, height). Default: (12, 5)
        title (str): Figure title. Default: "Confidence Distribution - Predicted Probabilities"
    
    Returns:
        plt.Figure: Matplotlib figure object with confidence distribution histograms
    
    Implementation notes:
        - Creates 1x2 subplot figure (H1N1 and seasonal side-by-side)
        - Plots histograms with 20-30 bins (default: 25)
        - X-axis: Predicted probability [0.0, 1.0]
        - Y-axis: Frequency/Count
        - Includes mean predicted probability as vertical reference line
        - Both vaccines on same figure for easy comparison
        - Saves figure to output_path if provided with 300 dpi and tight layout
        - Returns matplotlib Figure object for further manipulation
    """
    # Validate inputs
    if y_pred_h1n1.shape != y_pred_seasonal.shape:
        raise ValueError("y_pred_h1n1 and y_pred_seasonal shapes don't match")
    if not np.all((y_pred_h1n1 >= 0) & (y_pred_h1n1 <= 1)):
        raise ValueError("y_pred_h1n1 must be in [0.0, 1.0]")
    if not np.all((y_pred_seasonal >= 0) & (y_pred_seasonal <= 1)):
        raise ValueError("y_pred_seasonal must be in [0.0, 1.0]")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot H1N1 confidence distribution
    axes[0].hist(y_pred_h1n1, bins=25, alpha=0.7, color='#1f77b4',
                 edgecolor='black', linewidth=0.8)
    axes[0].axvline(np.mean(y_pred_h1n1), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {np.mean(y_pred_h1n1):.3f}')
    axes[0].set_xlabel('Predicted Probability', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('H1N1 Vaccine', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xlim([-0.02, 1.02])
    
    # Plot Seasonal confidence distribution
    axes[1].hist(y_pred_seasonal, bins=25, alpha=0.7, color='#ff7f0e',
                 edgecolor='black', linewidth=0.8)
    axes[1].axvline(np.mean(y_pred_seasonal), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {np.mean(y_pred_seasonal):.3f}')
    axes[1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Seasonal Vaccine', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xlim([-0.02, 1.02])
    
    # Set overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
