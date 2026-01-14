"""
Evaluation metrics and diagnostic tools for the flu vaccine prediction models.

This module provides metric computation and diagnostic functions for assessing
model performance on the multilabel flu vaccine prediction task. Key metrics include:
- ROC AUC (mean of H1N1 and seasonal vaccine AUC scores) - primary evaluation metric
- Calibration error (ECE, MCE, Brier score) - assesses probability calibration
- Confusion matrices and per-vaccine diagnostics

See PROBLEM_DESCRIPTION.md for submission format and evaluation details:
- Submission requires predicted probabilities (0.0-1.0) for both vaccines
- Evaluation metric: mean(AUC_h1n1, AUC_seasonal)
- Each vaccine is a separate binary classification target

Reference: SYSTEM_DESIGN.md - Component 7: Evaluation
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
from abc import ABC
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Compute evaluation metrics for multilabel vaccine prediction models.
    
    The flu vaccine prediction task is multilabel: each respondent can receive
    neither, one, or both vaccines. We evaluate each vaccine independently using
    binary classification metrics, then average the scores.
    
    Key metrics:
    - ROC AUC: Area under ROC curve (0.5 = random, 1.0 = perfect)
      Robust to class imbalance and probability calibration
    - Calibration Error (ECE): Expected Calibration Error
      Measures if predicted probabilities match empirical frequencies
    - Confusion Matrix: TP, FP, TN, FN counts for each vaccine
    
    Attributes:
        y_true_h1n1: True labels for H1N1 vaccine (n_samples,) with values 0/1
        y_true_seasonal: True labels for seasonal vaccine (n_samples,) with values 0/1
        y_pred_h1n1: Predicted probabilities for H1N1 (n_samples,) in [0, 1]
        y_pred_seasonal: Predicted probabilities for seasonal (n_samples,) in [0, 1]
        metrics_: Dictionary of computed metrics (populated after compute_* calls)
        
    Example:
        ```python
        evaluator = Evaluator()
        
        # Compute ROC AUC for each vaccine and their mean
        auroc_h1n1, auroc_seasonal, auroc_mean = evaluator.compute_auroc(
            y_true_h1n1, y_true_seasonal, y_pred_h1n1, y_pred_seasonal
        )
        
        # Compute calibration error
        ece, mce, brier = evaluator.calibration_error(
            y_true_h1n1, y_pred_h1n1, n_bins=10
        )
        
        # Get full diagnostic report
        diagnostics = evaluator.get_diagnostics(
            y_true_h1n1, y_true_seasonal, y_pred_h1n1, y_pred_seasonal
        )
        ```
    """
    
    def __init__(self):
        """Initialize evaluator with empty metric storage."""
        self.y_true_h1n1 = None
        self.y_true_seasonal = None
        self.y_pred_h1n1 = None
        self.y_pred_seasonal = None
        self.metrics_ = {}
    
    @staticmethod
    def compute_auroc(
        y_true_h1n1: np.ndarray,
        y_true_seasonal: np.ndarray,
        y_pred_h1n1: np.ndarray,
        y_pred_seasonal: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute ROC AUC scores for each vaccine and their mean.
        
        ROC AUC (Area Under the Receiver Operating Characteristic Curve) measures
        the probability that the model ranks a random positive example higher than
        a random negative example. Scores range from 0.5 (random) to 1.0 (perfect).
        
        For multilabel vaccine prediction:
        - Compute AUC separately for H1N1 and seasonal vaccines
        - Return individual scores plus their mean (competition evaluation metric)
        
        Parameters:
            y_true_h1n1 (np.ndarray): True labels for H1N1 vaccine.
                Shape: (n_samples,)
                Values: 0 or 1
            y_true_seasonal (np.ndarray): True labels for seasonal vaccine.
                Shape: (n_samples,)
                Values: 0 or 1
            y_pred_h1n1 (np.ndarray): Predicted probabilities for H1N1.
                Shape: (n_samples,)
                Values: in [0, 1]
            y_pred_seasonal (np.ndarray): Predicted probabilities for seasonal.
                Shape: (n_samples,)
                Values: in [0, 1]
        
        Returns:
            Tuple[float, float, float]: (auroc_h1n1, auroc_seasonal, auroc_mean)
                auroc_h1n1: AUC score for H1N1 vaccine (0.0-1.0)
                auroc_seasonal: AUC score for seasonal vaccine (0.0-1.0)
                auroc_mean: Mean AUC (competition evaluation metric)
        
        Implementation notes (COMPLETED):
            - ✅ Validates inputs (shapes match, values in valid ranges)
            - ✅ Uses sklearn.metrics.roc_auc_score()
            - ✅ Handles edge cases (no positive/negative examples via safe division)
            - ✅ Returns tuple of (auc_h1n1, auc_seasonal, mean)
        """
        from sklearn.metrics import roc_auc_score
        
        # Validate inputs
        if y_true_h1n1.shape != y_pred_h1n1.shape:
            raise ValueError(
                f"y_true_h1n1 and y_pred_h1n1 shapes don't match: "
                f"{y_true_h1n1.shape} vs {y_pred_h1n1.shape}"
            )
        if y_true_seasonal.shape != y_pred_seasonal.shape:
            raise ValueError(
                f"y_true_seasonal and y_pred_seasonal shapes don't match: "
                f"{y_true_seasonal.shape} vs {y_pred_seasonal.shape}"
            )
        if y_true_h1n1.shape[0] != y_true_seasonal.shape[0]:
            raise ValueError(
                f"Number of samples mismatch: "
                f"{y_true_h1n1.shape[0]} vs {y_true_seasonal.shape[0]}"
            )
        
        # Check value ranges
        if not np.all((y_true_h1n1 == 0) | (y_true_h1n1 == 1)):
            raise ValueError("y_true_h1n1 must contain only 0 or 1")
        if not np.all((y_true_seasonal == 0) | (y_true_seasonal == 1)):
            raise ValueError("y_true_seasonal must contain only 0 or 1")
        if not np.all((y_pred_h1n1 >= 0) & (y_pred_h1n1 <= 1)):
            raise ValueError("y_pred_h1n1 must be in [0, 1]")
        if not np.all((y_pred_seasonal >= 0) & (y_pred_seasonal <= 1)):
            raise ValueError("y_pred_seasonal must be in [0, 1]")
        
        # Compute AUC for each vaccine
        auroc_h1n1 = roc_auc_score(y_true_h1n1, y_pred_h1n1)
        auroc_seasonal = roc_auc_score(y_true_seasonal, y_pred_seasonal)
        
        # Compute mean AUC (competition evaluation metric)
        auroc_mean = (auroc_h1n1 + auroc_seasonal) / 2.0
        
        return auroc_h1n1, auroc_seasonal, auroc_mean
    
    @staticmethod
    def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute confusion matrix for a single vaccine.
        
        Counts true positives, false positives, true negatives, false negatives.
        The threshold determines the classification boundary for converting
        predicted probabilities to binary predictions.
        
        Parameters:
            y_true (np.ndarray): True labels for one vaccine.
                Shape: (n_samples,)
                Values: 0 or 1
            y_pred (np.ndarray): Predicted probabilities.
                Shape: (n_samples,)
                Values: in [0, 1]
            threshold (float): Classification threshold.
                Default: 0.5
                Predictions >= threshold classified as 1, else 0
        
        Returns:
            Dict[str, int]: Confusion matrix with keys:
                'tp': True Positives
                'fp': False Positives
                'tn': True Negatives
                'fn': False Negatives
                'sensitivity': TP / (TP + FN) - recall, true positive rate
                'specificity': TN / (TN + FP) - true negative rate
                'ppv': TP / (TP + FP) - positive predictive value, precision
                'npv': TN / (TN + FN) - negative predictive value
        
        Implementation notes (COMPLETED):
            - ✅ Validates inputs
            - ✅ Applies threshold: y_pred_binary = (y_pred >= threshold).astype(int)
            - ✅ Computes TP, FP, TN, FN using numpy comparisons
            - ✅ Computes derived metrics (sensitivity, specificity, PPV, NPV)
            - ✅ Returns dictionary with all metrics
        """
        # Validate inputs
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true and y_pred shapes don't match: {y_true.shape} vs {y_pred.shape}"
            )
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y_true must contain only 0 or 1")
        if not np.all((y_pred >= 0) & (y_pred <= 1)):
            raise ValueError("y_pred must be in [0, 1]")
        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        
        # Apply threshold to get binary predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Compute confusion matrix components
        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        tn = np.sum((y_pred_binary == 0) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))
        
        # Compute derived metrics with safe division (avoid division by zero)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        return {
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
        }
    
    @staticmethod
    def calibration_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[float, float, float]:
        """
        Compute calibration error metrics (ECE, MCE, Brier score).
        
        Calibration measures how well predicted probabilities match empirical
        frequencies. A well-calibrated model has P(y=1|pred_prob=p) ≈ p.
        
        Metrics:
        - Expected Calibration Error (ECE): Mean absolute difference between
          predicted probability and empirical frequency across bins
        - Maximum Calibration Error (MCE): Largest absolute difference in any bin
        - Brier Score: Mean squared difference between predicted probability
          and actual outcome (0-1, lower is better)
        
        Parameters:
            y_true (np.ndarray): True labels for one vaccine.
                Shape: (n_samples,)
                Values: 0 or 1
            y_pred (np.ndarray): Predicted probabilities.
                Shape: (n_samples,)
                Values: in [0, 1]
            n_bins (int): Number of bins for binning predictions.
                Default: 10 (divides [0, 1] into 10 equal-width bins)
                Common choices: 5, 10, 15, 20
        
        Returns:
            Tuple[float, float, float]: (ece, mce, brier_score)
                ece: Expected Calibration Error (0.0-1.0, lower is better)
                mce: Maximum Calibration Error (0.0-1.0, lower is better)
                brier_score: Brier Score (0.0-1.0, lower is better)
        
        Implementation notes (COMPLETED):
            - ✅ Validates inputs
            - ✅ Bins predictions into n_bins bins of equal width [0, 1]
            - ✅ For each bin:
                  - Computes bin confidence (mean of predictions in bin)
                  - Computes bin accuracy (mean of y_true in bin)
                  - Counts samples in bin
            - ✅ Weights absolute difference by bin size
            - ✅ Computes ECE as weighted average of differences
            - ✅ Computes MCE as maximum difference
            - ✅ Computes Brier score: mean((y_pred - y_true)^2)
            - ✅ Returns (ece, mce, brier)
        """
        # Validate inputs
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true and y_pred shapes don't match: {y_true.shape} vs {y_pred.shape}"
            )
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y_true must contain only 0 or 1")
        if not np.all((y_pred >= 0) & (y_pred <= 1)):
            raise ValueError("y_pred must be in [0, 1]")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        
        # Compute Brier score first (simplest metric)
        brier_score = np.mean((y_pred - y_true) ** 2)
        
        # Compute ECE and MCE using bin-based method
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        mce = 0.0
        total_samples = len(y_true)
        
        for i in range(n_bins):
            # Get mask for samples in this bin
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Include right boundary in last bin
            if i == n_bins - 1:
                mask = (y_pred >= bin_lower) & (y_pred <= bin_upper)
            else:
                mask = (y_pred >= bin_lower) & (y_pred < bin_upper)
            
            # Skip empty bins
            if not np.any(mask):
                continue
            
            # Compute bin statistics
            bin_confidence = np.mean(y_pred[mask])  # Mean predicted probability
            bin_accuracy = np.mean(y_true[mask])    # Empirical frequency
            bin_size = np.sum(mask)
            
            # Compute difference
            bin_diff = np.abs(bin_confidence - bin_accuracy)
            
            # Weight by bin size for ECE
            ece += (bin_size / total_samples) * bin_diff
            
            # Track maximum error for MCE
            mce = max(mce, bin_diff)
        
        return float(ece), float(mce), float(brier_score)
    
    def get_diagnostics(
        self,
        y_true_h1n1: np.ndarray,
        y_true_seasonal: np.ndarray,
        y_pred_h1n1: np.ndarray,
        y_pred_seasonal: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute comprehensive diagnostic metrics for both vaccines.
        
        Returns a full diagnostic report including:
        - ROC AUC scores for each vaccine and their mean
        - Confusion matrices and derived metrics (sensitivity, specificity, PPV)
        - Calibration metrics (ECE, MCE, Brier score)
        - Class balance information (proportion of positive examples)
        
        Parameters:
            y_true_h1n1 (np.ndarray): True H1N1 labels (n_samples,)
            y_true_seasonal (np.ndarray): True seasonal labels (n_samples,)
            y_pred_h1n1 (np.ndarray): Predicted H1N1 probabilities (n_samples,)
            y_pred_seasonal (np.ndarray): Predicted seasonal probabilities (n_samples,)
            threshold (float): Binary classification threshold. Default: 0.5
        
        Returns:
            Dict[str, float]: Comprehensive metrics including:
                'auroc_h1n1': AUC for H1N1
                'auroc_seasonal': AUC for seasonal
                'auroc_mean': Mean AUC (competition metric)
                'h1n1_sensitivity': TP / (TP + FN) for H1N1
                'h1n1_specificity': TN / (TN + FP) for H1N1
                'h1n1_ppv': TP / (TP + FP) for H1N1
                'h1n1_ece': Expected Calibration Error for H1N1
                'h1n1_brier': Brier Score for H1N1
                'seasonal_sensitivity': TP / (TP + FN) for seasonal
                'seasonal_specificity': TN / (TN + FP) for seasonal
                'seasonal_ppv': TP / (TP + FP) for seasonal
                'seasonal_ece': Expected Calibration Error for seasonal
                'seasonal_brier': Brier Score for seasonal
                'h1n1_prevalence': Proportion of positive examples for H1N1
                'seasonal_prevalence': Proportion of positive examples for seasonal
        
        Implementation notes (COMPLETED):
            - ✅ Stores inputs in self.y_true_h1n1, etc. for reference
            - ✅ Calls compute_auroc() to get AUC scores
            - ✅ Calls confusion_matrix() for both vaccines
            - ✅ Calls calibration_error() for both vaccines
            - ✅ Computes prevalence (proportion of 1s)
            - ✅ Assembles all metrics into dictionary
            - ✅ Stores in self.metrics_
            - ✅ Returns dictionary
        """
        # Store inputs for reference
        self.y_true_h1n1 = y_true_h1n1
        self.y_true_seasonal = y_true_seasonal
        self.y_pred_h1n1 = y_pred_h1n1
        self.y_pred_seasonal = y_pred_seasonal
        
        # Compute AUC scores
        auroc_h1n1, auroc_seasonal, auroc_mean = self.compute_auroc(
            y_true_h1n1, y_true_seasonal, y_pred_h1n1, y_pred_seasonal
        )
        
        # Compute confusion matrices for both vaccines
        cm_h1n1 = self.confusion_matrix(y_true_h1n1, y_pred_h1n1, threshold)
        cm_seasonal = self.confusion_matrix(y_true_seasonal, y_pred_seasonal, threshold)
        
        # Compute calibration metrics for both vaccines
        ece_h1n1, mce_h1n1, brier_h1n1 = self.calibration_error(
            y_true_h1n1, y_pred_h1n1, n_bins=10
        )
        ece_seasonal, mce_seasonal, brier_seasonal = self.calibration_error(
            y_true_seasonal, y_pred_seasonal, n_bins=10
        )
        
        # Compute prevalence
        h1n1_prevalence = np.mean(y_true_h1n1)
        seasonal_prevalence = np.mean(y_true_seasonal)
        
        # Assemble all metrics
        self.metrics_ = {
            'auroc_h1n1': auroc_h1n1,
            'auroc_seasonal': auroc_seasonal,
            'auroc_mean': auroc_mean,
            'h1n1_sensitivity': cm_h1n1['sensitivity'],
            'h1n1_specificity': cm_h1n1['specificity'],
            'h1n1_ppv': cm_h1n1['ppv'],
            'h1n1_ece': ece_h1n1,
            'h1n1_mce': mce_h1n1,
            'h1n1_brier': brier_h1n1,
            'seasonal_sensitivity': cm_seasonal['sensitivity'],
            'seasonal_specificity': cm_seasonal['specificity'],
            'seasonal_ppv': cm_seasonal['ppv'],
            'seasonal_ece': ece_seasonal,
            'seasonal_mce': mce_seasonal,
            'seasonal_brier': brier_seasonal,
            'h1n1_prevalence': h1n1_prevalence,
            'seasonal_prevalence': seasonal_prevalence,
        }
        
        return self.metrics_
    
    def summary(self) -> str:
        """
        Return a formatted string summary of computed metrics.
        
        Example output:
        ```
        ===== Evaluation Summary =====
        ROC AUC:
          H1N1:    0.82
          Seasonal: 0.78
          Mean:     0.80
        
        Calibration (H1N1):
          ECE:   0.04
          Brier: 0.18
        
        Calibration (Seasonal):
          ECE:   0.05
          Brier: 0.21
        
        Prevalence:
          H1N1:     0.47
          Seasonal: 0.64
        ```
        
        Returns:
            str: Formatted summary of self.metrics_
            
        Implementation notes (COMPLETED):
            - ✅ Formats metrics for display
            - ✅ Rounds to 4 decimal places
            - ✅ Groups by vaccine and metric type
            - ✅ Returns human-readable string
        """
        if not self.metrics_:
            return "No metrics computed. Call get_diagnostics() first."
        
        lines = ["===== Evaluation Summary =====", ""]
        
        # ROC AUC section
        lines.append("ROC AUC:")
        lines.append(f"  H1N1:     {self.metrics_['auroc_h1n1']:.4f}")
        lines.append(f"  Seasonal: {self.metrics_['auroc_seasonal']:.4f}")
        lines.append(f"  Mean:     {self.metrics_['auroc_mean']:.4f}")
        lines.append("")
        
        # Confusion Matrix / Classification Metrics
        lines.append("Classification Metrics:")
        lines.append("  H1N1:")
        lines.append(f"    Sensitivity: {self.metrics_['h1n1_sensitivity']:.4f}")
        lines.append(f"    Specificity: {self.metrics_['h1n1_specificity']:.4f}")
        lines.append(f"    PPV:         {self.metrics_['h1n1_ppv']:.4f}")
        lines.append("  Seasonal:")
        lines.append(f"    Sensitivity: {self.metrics_['seasonal_sensitivity']:.4f}")
        lines.append(f"    Specificity: {self.metrics_['seasonal_specificity']:.4f}")
        lines.append(f"    PPV:         {self.metrics_['seasonal_ppv']:.4f}")
        lines.append("")
        
        # Calibration section
        lines.append("Calibration Error:")
        lines.append("  H1N1:")
        lines.append(f"    ECE:   {self.metrics_['h1n1_ece']:.4f}")
        lines.append(f"    MCE:   {self.metrics_['h1n1_mce']:.4f}")
        lines.append(f"    Brier: {self.metrics_['h1n1_brier']:.4f}")
        lines.append("  Seasonal:")
        lines.append(f"    ECE:   {self.metrics_['seasonal_ece']:.4f}")
        lines.append(f"    MCE:   {self.metrics_['seasonal_mce']:.4f}")
        lines.append(f"    Brier: {self.metrics_['seasonal_brier']:.4f}")
        lines.append("")
        
        # Class balance
        lines.append("Class Prevalence:")
        lines.append(f"  H1N1:     {self.metrics_['h1n1_prevalence']:.4f}")
        lines.append(f"  Seasonal: {self.metrics_['seasonal_prevalence']:.4f}")
        
        return "\n".join(lines)
    
    def evaluate(
        self,
        y_true_h1n1: np.ndarray,
        y_true_seasonal: np.ndarray,
        y_pred_h1n1: np.ndarray,
        y_pred_seasonal: np.ndarray,
        threshold: Union[float, str] = 0.5,
        tuned_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model performance with support for adaptive thresholds.
        
        This is a convenience method that wraps get_diagnostics() and handles
        adaptive threshold selection for config-driven pipelines.
        
        Parameters:
            y_true_h1n1 (np.ndarray): True H1N1 labels (n_samples,)
                Values: 0 or 1
            y_true_seasonal (np.ndarray): True seasonal labels (n_samples,)
                Values: 0 or 1
            y_pred_h1n1 (np.ndarray): Predicted H1N1 probabilities (n_samples,)
                Values: in [0, 1]
            y_pred_seasonal (np.ndarray): Predicted seasonal probabilities (n_samples,)
                Values: in [0, 1]
            threshold (Union[float, str]): Classification threshold strategy.
                - If float in [0, 1]: use as fixed threshold for all metrics
                - If string "adaptive": use thresholds from tuned_thresholds dict
                Default: 0.5 (standard probability threshold)
            tuned_thresholds (Optional[Dict[str, float]]): Pre-tuned thresholds
                per vaccine. Required if threshold="adaptive".
                Format: {"h1n1_vaccine": float, "seasonal_vaccine": float}
        
        Returns:
            Dict[str, float]: Comprehensive metrics including:
                - auroc_h1n1, auroc_seasonal, auroc_mean
                - h1n1_sensitivity, h1n1_specificity, h1n1_ppv
                - seasonal_sensitivity, seasonal_specificity, seasonal_ppv
                - h1n1_ece, h1n1_mce, h1n1_brier
                - seasonal_ece, seasonal_mce, seasonal_brier
                - h1n1_prevalence, seasonal_prevalence
        
        Raises:
            ValueError: If threshold="adaptive" but tuned_thresholds is None
            ValueError: If threshold is not a float in [0, 1] or "adaptive"
        
        Example:
            ```python
            # Standard evaluation with fixed threshold
            metrics = evaluator.evaluate(
                y_train_h1n1, y_train_seasonal,
                cv_probs_h1n1, cv_probs_seasonal,
                threshold=0.5
            )
            
            # Adaptive threshold evaluation (from threshold tuning)
            metrics = evaluator.evaluate(
                y_train_h1n1, y_train_seasonal,
                cv_probs_h1n1, cv_probs_seasonal,
                threshold="adaptive",
                tuned_thresholds={"h1n1_vaccine": 0.48, "seasonal_vaccine": 0.52}
            )
            ```
        """
        # Handle adaptive threshold
        if isinstance(threshold, str):
            if threshold.lower() == "adaptive":
                if tuned_thresholds is None:
                    raise ValueError(
                        "threshold='adaptive' requires tuned_thresholds dict "
                        "with keys 'h1n1_vaccine' and 'seasonal_vaccine'"
                    )
                # Use different thresholds for each vaccine
                h1n1_threshold = tuned_thresholds.get('h1n1_vaccine', 0.5)
                seasonal_threshold = tuned_thresholds.get('seasonal_vaccine', 0.5)
                
                logger.info(
                    f"Using adaptive thresholds: "
                    f"H1N1={h1n1_threshold:.3f}, Seasonal={seasonal_threshold:.3f}"
                )
                
                # Compute ROC AUC with raw probabilities (ignores threshold)
                auroc_h1n1, auroc_seasonal, auroc_mean = self.compute_auroc(
                    y_true_h1n1, y_true_seasonal, y_pred_h1n1, y_pred_seasonal
                )
                
                # Compute confusion matrix metrics using tuned thresholds
                cm_h1n1 = self.confusion_matrix(
                    y_true_h1n1, y_pred_h1n1, h1n1_threshold
                )
                cm_seasonal = self.confusion_matrix(
                    y_true_seasonal, y_pred_seasonal, seasonal_threshold
                )
                
                # Calibration metrics (threshold-independent)
                ece_h1n1, mce_h1n1, brier_h1n1 = self.calibration_error(
                    y_true_h1n1, y_pred_h1n1, n_bins=10
                )
                ece_seasonal, mce_seasonal, brier_seasonal = self.calibration_error(
                    y_true_seasonal, y_pred_seasonal, n_bins=10
                )
                
                # Prevalence
                h1n1_prevalence = np.mean(y_true_h1n1)
                seasonal_prevalence = np.mean(y_true_seasonal)
                
                # Assemble metrics
                metrics = {
                    'auroc_h1n1': auroc_h1n1,
                    'auroc_seasonal': auroc_seasonal,
                    'auroc_mean': auroc_mean,
                    'h1n1_sensitivity': cm_h1n1['sensitivity'],
                    'h1n1_specificity': cm_h1n1['specificity'],
                    'h1n1_ppv': cm_h1n1['ppv'],
                    'h1n1_ece': ece_h1n1,
                    'h1n1_mce': mce_h1n1,
                    'h1n1_brier': brier_h1n1,
                    'seasonal_sensitivity': cm_seasonal['sensitivity'],
                    'seasonal_specificity': cm_seasonal['specificity'],
                    'seasonal_ppv': cm_seasonal['ppv'],
                    'seasonal_ece': ece_seasonal,
                    'seasonal_mce': mce_seasonal,
                    'seasonal_brier': brier_seasonal,
                    'h1n1_prevalence': h1n1_prevalence,
                    'seasonal_prevalence': seasonal_prevalence,
                }
                
                self.metrics_ = metrics
                return metrics
            else:
                raise ValueError(
                    f"threshold must be float in [0, 1] or 'adaptive', got: {threshold}"
                )
        
        # Handle numeric threshold
        if not isinstance(threshold, (int, float)):
            raise ValueError(
                f"threshold must be numeric or 'adaptive', got: {type(threshold)}"
            )
        
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got: {threshold}")
        
        # Standard evaluation with fixed threshold
        logger.info(f"Using fixed threshold: {threshold:.3f}")
        return self.get_diagnostics(
            y_true_h1n1, y_true_seasonal,
            y_pred_h1n1, y_pred_seasonal,
            threshold=threshold
        )
