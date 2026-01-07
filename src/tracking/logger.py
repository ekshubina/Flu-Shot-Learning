"""
Experiment tracking and logging for model training runs.

This module provides interfaces and implementations for systematic recording
and querying of model training runs, configurations, hyperparameters, and
evaluation metrics.

Tracking enables:
- Reproducibility: Store all configuration used for each run
- Comparison: Query and rank runs by performance
- History: Access previous experiments for analysis and debugging
- Export: Save experiment results for reporting

Reference: SYSTEM_DESIGN.md - Component 8: Experiment Tracking
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import csv
from pathlib import Path
import pandas as pd


@dataclass
class RunRecord:
    """
    Schema for a single experiment run.
    
    Attributes:
        run_id: Unique identifier for the run (UUID or sequential number)
        timestamp: When the run was started (ISO 8601 format)
        model_type: Type of model used (e.g., 'logistic_regression', 'xgboost')
        config_json: Full pipeline configuration as JSON string
        hyperparameters_json: Model hyperparameters as JSON string
        auroc_h1n1: AUC score for H1N1 vaccine on validation set
        auroc_seasonal: AUC score for seasonal vaccine on validation set
        auroc_mean: Mean AUC (competition metric)
        h1n1_sensitivity: Sensitivity for H1N1 (threshold=0.5)
        h1n1_specificity: Specificity for H1N1 (threshold=0.5)
        h1n1_ppv: Positive Predictive Value for H1N1
        seasonal_sensitivity: Sensitivity for seasonal (threshold=0.5)
        seasonal_specificity: Specificity for seasonal (threshold=0.5)
        seasonal_ppv: Positive Predictive Value for seasonal
        h1n1_ece: Expected Calibration Error for H1N1
        seasonal_ece: Expected Calibration Error for seasonal
        notes: Optional notes/description of the run
        status: Run status ('completed', 'failed', 'in_progress')
    """
    run_id: str
    timestamp: str
    model_type: str
    config_json: str
    hyperparameters_json: str
    auroc_h1n1: Optional[float] = None
    auroc_seasonal: Optional[float] = None
    auroc_mean: Optional[float] = None
    h1n1_sensitivity: Optional[float] = None
    h1n1_specificity: Optional[float] = None
    h1n1_ppv: Optional[float] = None
    seasonal_sensitivity: Optional[float] = None
    seasonal_specificity: Optional[float] = None
    seasonal_ppv: Optional[float] = None
    h1n1_ece: Optional[float] = None
    seasonal_ece: Optional[float] = None
    notes: Optional[str] = None
    status: str = 'in_progress'


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking.
    
    Defines interface for recording and querying model training runs.
    Implementations can use CSV, databases, or cloud services.
    
    Example:
        ```python
        tracker = CSVExperimentLogger(log_path='experiments.csv')
        
        # Log a new run
        tracker.log_run(
            run_id='run_001',
            model_type='xgboost',
            config={'n_trees': 100, 'max_depth': 5},
            hyperparameters={'learning_rate': 0.1},
            metrics={'auroc_mean': 0.82}
        )
        
        # Query runs
        best_runs = tracker.rank_by_auroc(limit=10)
        xgb_runs = tracker.filter_by_model_type('xgboost')
        
        # Export for reporting
        df = tracker.export()
        ```
    """
    
    @abstractmethod
    def log_run(
        self,
        run_id: str,
        model_type: str,
        config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        notes: Optional[str] = None,
    ) -> RunRecord:
        """
        Log a new experiment run.
        
        Parameters:
            run_id: Unique identifier for this run
            model_type: Type of model ('logistic_regression', 'xgboost', 'lightgbm', 'random_forest')
            config: Full pipeline configuration dict
            hyperparameters: Model-specific hyperparameters dict
            metrics: Computed metrics dict (auroc_h1n1, auroc_seasonal, etc.)
            notes: Optional description or notes for the run
        
        Returns:
            RunRecord: The recorded run
        """
        pass
    
    @abstractmethod
    def get_run_by_id(self, run_id: str) -> Optional[RunRecord]:
        """
        Retrieve a specific run by its ID.
        
        Parameters:
            run_id: Unique identifier for the run
        
        Returns:
            RunRecord if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_all_runs(self) -> List[RunRecord]:
        """
        Retrieve all recorded runs.
        
        Returns:
            List of RunRecord objects (empty if no runs)
        """
        pass
    
    @abstractmethod
    def rank_by_auroc(self, limit: Optional[int] = None) -> List[RunRecord]:
        """
        Get runs ranked by mean AUC (highest to lowest).
        
        Parameters:
            limit: Maximum number of runs to return. None = all
        
        Returns:
            List of RunRecord objects sorted by auroc_mean descending
        """
        pass
    
    @abstractmethod
    def filter_by_model_type(self, model_type: str) -> List[RunRecord]:
        """
        Filter runs by model type.
        
        Parameters:
            model_type: Model type to filter ('logistic_regression', 'xgboost', etc.)
        
        Returns:
            List of RunRecord objects matching the model type
        """
        pass
    
    @abstractmethod
    def filter_by_auroc_range(
        self, min_auroc: float, max_auroc: float
    ) -> List[RunRecord]:
        """
        Filter runs by mean AUC range.
        
        Parameters:
            min_auroc: Minimum mean AUC (inclusive)
            max_auroc: Maximum mean AUC (inclusive)
        
        Returns:
            List of RunRecord objects within the AUC range
        """
        pass
    
    @abstractmethod
    def export(self, format: str = 'dataframe') -> Any:
        """
        Export all runs in the requested format.
        
        Parameters:
            format: Output format ('dataframe' for pandas, 'json', 'csv')
        
        Returns:
            DataFrame (pandas) if format='dataframe', dict/str otherwise
        """
        pass


class CSVExperimentLogger(ExperimentTracker):
    """
    Experiment tracker using CSV file for persistence.
    
    Simple, portable tracker that stores runs in a CSV file. Good for
    small-to-medium experiments (< 1000 runs). For larger scales, consider
    database or cloud logging.
    
    Implementation notes:
        - TODO: In __init__(), create CSV file with headers if not exists
        - TODO: In log_run(), append new row to CSV and return RunRecord
        - TODO: In get_run_by_id(), read CSV and find matching run_id
        - TODO: In get_all_runs(), read CSV and return list of RunRecords
        - TODO: In rank_by_auroc(), get all runs and sort by auroc_mean
        - TODO: In filter_by_model_type(), get all runs and filter
        - TODO: In filter_by_auroc_range(), get all runs and filter
        - TODO: In export(), return pandas DataFrame from CSV
    """
    
    def __init__(self, log_path: str = 'experiments.csv'):
        """
        Initialize CSV experiment logger.
        
        Parameters:
            log_path: Path to CSV file for storing runs
        """
        self.log_path = Path(log_path)
        self._initialize_csv()
    
    def _initialize_csv(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        # TODO: Implement
        pass
    
    def log_run(
        self,
        run_id: str,
        model_type: str,
        config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        notes: Optional[str] = None,
    ) -> RunRecord:
        """
        Log a new run to CSV file.
        
        Implementation notes:
            - TODO: Create RunRecord with provided data
            - TODO: Serialize config and hyperparameters to JSON
            - TODO: Append row to CSV file
            - TODO: Return the created RunRecord
        """
        # TODO: Implement
        raise NotImplementedError("Subclass must implement log_run()")
    
    def get_run_by_id(self, run_id: str) -> Optional[RunRecord]:
        """
        Retrieve a run by ID from CSV.
        
        Implementation notes:
            - TODO: Read CSV file
            - TODO: Find row with matching run_id
            - TODO: Parse RunRecord from row
            - TODO: Return RunRecord or None if not found
        """
        # TODO: Implement
        pass
    
    def get_all_runs(self) -> List[RunRecord]:
        """
        Get all runs from CSV.
        
        Implementation notes:
            - TODO: Read CSV file
            - TODO: Parse each row as RunRecord
            - TODO: Return list of RunRecords
        """
        # TODO: Implement
        raise NotImplementedError("Subclass must implement get_all_runs()")
    
    def rank_by_auroc(self, limit: Optional[int] = None) -> List[RunRecord]:
        """
        Get runs sorted by mean AUC (descending).
        
        Implementation notes:
            - TODO: Get all runs
            - TODO: Filter to completed runs (status='completed')
            - TODO: Sort by auroc_mean descending
            - TODO: Apply limit if provided
            - TODO: Return sorted list
        """
        # TODO: Implement
        raise NotImplementedError("Subclass must implement rank_by_auroc()")
    
    def filter_by_model_type(self, model_type: str) -> List[RunRecord]:
        """
        Filter runs by model type.
        
        Implementation notes:
            - TODO: Get all runs
            - TODO: Filter to matching model_type
            - TODO: Return filtered list
        """
        # TODO: Implement
        raise NotImplementedError("Subclass must implement filter_by_model_type()")
    
    def filter_by_auroc_range(
        self, min_auroc: float, max_auroc: float
    ) -> List[RunRecord]:
        """
        Filter runs by mean AUC range.
        
        Implementation notes:
            - TODO: Get all runs
            - TODO: Filter to runs with auroc_mean in [min_auroc, max_auroc]
            - TODO: Return filtered list
        """
        # TODO: Implement
        raise NotImplementedError("Subclass must implement filter_by_auroc_range()")
    
    def export(self, format: str = 'dataframe') -> Any:
        """
        Export all runs in requested format.
        
        Implementation notes:
            - TODO: Read CSV file
            - TODO: If format='dataframe', return pandas DataFrame
            - TODO: If format='json', return list of dicts as JSON string
            - TODO: If format='csv', return raw CSV string
            - TODO: Raise ValueError for unknown format
        """
        # TODO: Implement
        pass
    
    def update_run_status(self, run_id: str, status: str) -> None:
        """
        Update the status of a run (e.g., 'completed' or 'failed').
        
        Parameters:
            run_id: ID of run to update
            status: New status ('completed', 'failed', 'in_progress')
        
        Implementation notes:
            - TODO: Read CSV file
            - TODO: Find row with matching run_id
            - TODO: Update status column
            - TODO: Write back to CSV
        """
        # TODO: Implement
        pass


def create_run_id() -> str:
    """
    Generate a unique run ID using timestamp.
    
    Format: run_YYYYMMDD_HHMMSS_microseconds
    
    Returns:
        str: Unique run ID
        
    Implementation notes:
        - TODO: Get current datetime
        - TODO: Format as YYYYMMDD_HHMMSS_MICROSECONDS
        - TODO: Return as string
    """
    # TODO: Implement
    raise NotImplementedError("create_run_id() not yet implemented")
