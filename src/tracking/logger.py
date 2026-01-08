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
        if self.log_path.exists():
            return
        
        # Create parent directories if they don't exist
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create CSV with headers
        fieldnames = [
            'run_id', 'timestamp', 'model_type', 'config_json',
            'hyperparameters_json', 'auroc_h1n1', 'auroc_seasonal',
            'auroc_mean', 'h1n1_sensitivity', 'h1n1_specificity',
            'h1n1_ppv', 'seasonal_sensitivity', 'seasonal_specificity',
            'seasonal_ppv', 'h1n1_ece', 'seasonal_ece', 'notes', 'status'
        ]
        
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
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
        # Create timestamp
        timestamp = datetime.now().isoformat()
        
        # Serialize config and hyperparameters to JSON
        config_json = json.dumps(config)
        hyperparameters_json = json.dumps(hyperparameters)
        
        # Create RunRecord with metrics
        record = RunRecord(
            run_id=run_id,
            timestamp=timestamp,
            model_type=model_type,
            config_json=config_json,
            hyperparameters_json=hyperparameters_json,
            auroc_h1n1=metrics.get('auroc_h1n1'),
            auroc_seasonal=metrics.get('auroc_seasonal'),
            auroc_mean=metrics.get('auroc_mean'),
            h1n1_sensitivity=metrics.get('h1n1_sensitivity'),
            h1n1_specificity=metrics.get('h1n1_specificity'),
            h1n1_ppv=metrics.get('h1n1_ppv'),
            seasonal_sensitivity=metrics.get('seasonal_sensitivity'),
            seasonal_specificity=metrics.get('seasonal_specificity'),
            seasonal_ppv=metrics.get('seasonal_ppv'),
            h1n1_ece=metrics.get('h1n1_ece'),
            seasonal_ece=metrics.get('seasonal_ece'),
            notes=notes,
            status='completed'
        )
        
        # Append to CSV file
        fieldnames = [
            'run_id', 'timestamp', 'model_type', 'config_json',
            'hyperparameters_json', 'auroc_h1n1', 'auroc_seasonal',
            'auroc_mean', 'h1n1_sensitivity', 'h1n1_specificity',
            'h1n1_ppv', 'seasonal_sensitivity', 'seasonal_specificity',
            'seasonal_ppv', 'h1n1_ece', 'seasonal_ece', 'notes', 'status'
        ]
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(asdict(record))
        
        return record
    
    def get_run_by_id(self, run_id: str) -> Optional[RunRecord]:
        """
        Retrieve a run by ID from CSV.
        
        Implementation notes:
            - TODO: Read CSV file
            - TODO: Find row with matching run_id
            - TODO: Parse RunRecord from row
            - TODO: Return RunRecord or None if not found
        """
        if not self.log_path.exists():
            return None
        
        df = pd.read_csv(self.log_path)
        matching_rows = df[df['run_id'] == run_id]
        
        if matching_rows.empty:
            return None
        
        row = matching_rows.iloc[0]
        return self._row_to_record(row)
    
    def get_all_runs(self) -> List[RunRecord]:
        """
        Get all runs from CSV.
        
        Implementation notes:
            - TODO: Read CSV file
            - TODO: Parse each row as RunRecord
            - TODO: Return list of RunRecords
        """
        if not self.log_path.exists():
            return []
        
        df = pd.read_csv(self.log_path)
        return [self._row_to_record(row) for _, row in df.iterrows()]
    
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
        runs = self.get_all_runs()
        
        # Filter to completed runs with auroc_mean
        completed_runs = [
            r for r in runs 
            if r.status == 'completed' and r.auroc_mean is not None
        ]
        
        # Sort by auroc_mean descending
        sorted_runs = sorted(completed_runs, key=lambda r: r.auroc_mean, reverse=True)
        
        if limit is not None:
            sorted_runs = sorted_runs[:limit]
        
        return sorted_runs
    
    def filter_by_model_type(self, model_type: str) -> List[RunRecord]:
        """
        Filter runs by model type.
        
        Implementation notes:
            - TODO: Get all runs
            - TODO: Filter to matching model_type
            - TODO: Return filtered list
        """
        runs = self.get_all_runs()
        return [r for r in runs if r.model_type == model_type]
    
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
        runs = self.get_all_runs()
        return [
            r for r in runs 
            if r.auroc_mean is not None 
            and min_auroc <= r.auroc_mean <= max_auroc
        ]
    
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
        if not self.log_path.exists():
            if format == 'dataframe':
                return pd.DataFrame()
            elif format == 'json':
                return json.dumps([])
            elif format == 'csv':
                return ""
            else:
                raise ValueError(f"Unknown export format: {format}")
        
        if format == 'dataframe':
            return pd.read_csv(self.log_path)
        elif format == 'json':
            df = pd.read_csv(self.log_path)
            return df.to_json(orient='records')
        elif format == 'csv':
            with open(self.log_path, 'r') as f:
                return f.read()
        else:
            raise ValueError(f"Unknown export format: {format}")
    
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
        if not self.log_path.exists():
            return
        
        df = pd.read_csv(self.log_path)
        df.loc[df['run_id'] == run_id, 'status'] = status
        df.to_csv(self.log_path, index=False)
    
    @staticmethod
    def _row_to_record(row: Any) -> RunRecord:
        """
        Convert a DataFrame row to a RunRecord.
        
        Parameters:
            row: pandas Series representing one row
        
        Returns:
            RunRecord object
        """
        # Handle NaN values by converting to None
        def to_none(val):
            if pd.isna(val):
                return None
            return val
        
        return RunRecord(
            run_id=row['run_id'],
            timestamp=row['timestamp'],
            model_type=row['model_type'],
            config_json=row['config_json'],
            hyperparameters_json=row['hyperparameters_json'],
            auroc_h1n1=to_none(row['auroc_h1n1']),
            auroc_seasonal=to_none(row['auroc_seasonal']),
            auroc_mean=to_none(row['auroc_mean']),
            h1n1_sensitivity=to_none(row['h1n1_sensitivity']),
            h1n1_specificity=to_none(row['h1n1_specificity']),
            h1n1_ppv=to_none(row['h1n1_ppv']),
            seasonal_sensitivity=to_none(row['seasonal_sensitivity']),
            seasonal_specificity=to_none(row['seasonal_specificity']),
            seasonal_ppv=to_none(row['seasonal_ppv']),
            h1n1_ece=to_none(row['h1n1_ece']),
            seasonal_ece=to_none(row['seasonal_ece']),
            notes=to_none(row['notes']),
            status=row['status']
        )


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
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    microseconds = now.microsecond
    return f"run_{timestamp_str}_{microseconds}"
