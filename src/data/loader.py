"""
Data loading interface and implementations.

This module defines the abstract interface for data loaders and provides concrete
implementations for loading the H1N1 flu shot prediction dataset.

The multilabel nature of this problem requires careful handling of two independent
binary targets (h1n1_vaccine and seasonal_vaccine), and the data loader must ensure
proper stratification and validation across both labels.

See PROBLEM_DESCRIPTION.md for feature definitions and data specifications.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class DataSplit:
    """
    Container for train/val/test split data.
    
    Attributes:
        X_train: Training features (n_train_samples, n_features)
        y_train: Training labels (n_train_samples, 2) - h1n1_vaccine and seasonal_vaccine
        X_val: Validation features (n_val_samples, n_features)
        y_val: Validation labels (n_val_samples, 2)
        X_test: Test features (n_test_samples, n_features)
        respondent_ids_test: Test respondent IDs for submission formatting
    """
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_val: pd.DataFrame
    y_val: pd.DataFrame
    X_test: pd.DataFrame
    respondent_ids_test: pd.Series


@dataclass
class DataValidationResult:
    """
    Results of data validation checks.
    
    Attributes:
        is_valid: Whether all checks passed
        n_samples: Number of samples loaded
        n_features: Number of features
        missing_by_feature: Dict of feature -> % missing values
        class_distribution: Dict of target -> class distribution percentages
        feature_types: Dict of feature -> inferred data type
        issues: List of validation issue descriptions
        warnings: List of validation warning descriptions
    """
    is_valid: bool
    n_samples: int
    n_features: int
    missing_by_feature: Dict[str, float]
    class_distribution: Dict[str, Dict[str, float]]
    feature_types: Dict[str, str]
    issues: List[str]
    warnings: List[str]


class DataLoader(ABC):
    """
    Abstract base class for loading and managing H1N1 flu shot prediction data.
    
    This loader handles the multilabel classification problem where each respondent
    has two independent binary targets: h1n1_vaccine and seasonal_vaccine (both 0 or 1).
    
    Key responsibilities:
    - Load training features, training labels, and test features from files
    - Create stratified cross-validation splits respecting both vaccine labels
    - Validate data integrity and report missing values, class distribution
    - Provide convenient access to respondent IDs for submission formatting
    
    The loader assumes:
    - Features and labels are in separate CSV files with 'respondent_id' as common key
    - All respondents in features have corresponding labels
    - Target columns are 'h1n1_vaccine' and 'seasonal_vaccine' (both 0/1)
    - Features contain respondent_id + 35 feature columns (see PROBLEM_DESCRIPTION.md)
    
    Example:
        >>> from src.config import DataConfig
        >>> from src.data import CSVDataLoader
        >>> config = DataConfig(
        ...     train_features_path="data/training_set_features.csv",
        ...     train_labels_path="data/training_set_labels.csv",
        ...     test_features_path="data/test_set_features.csv",
        ...     n_folds=5,
        ...     stratify=True,
        ... )
        >>> loader = CSVDataLoader(config)
        >>> X_train, y_train, X_test = loader.load_train(), loader.load_test()
        >>> splits = loader.create_splits()  # List[DataSplit] with 5 folds
        >>> validation_report = loader.validate()
    """

    @abstractmethod
    def load_train(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training features and labels.
        
        Returns:
            Tuple of (X_train, y_train) where:
            - X_train: DataFrame with shape (n_train, 35 features + respondent_id)
            - y_train: DataFrame with columns ['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine']
            
        Raises:
            FileNotFoundError: If data files not found
            ValueError: If data format is invalid
        """
        pass

    @abstractmethod
    def load_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load test features and respondent IDs.
        
        Returns:
            Tuple of (X_test, respondent_ids) where:
            - X_test: DataFrame with shape (n_test, 35 features + respondent_id)
            - respondent_ids: Series of test respondent IDs for submission formatting
            
        Raises:
            FileNotFoundError: If data files not found
            ValueError: If data format is invalid
        """
        pass

    @abstractmethod
    def create_splits(
        self,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
    ) -> List[DataSplit]:
        """
        Create cross-validation splits stratified by both target variables.
        
        For multilabel problems, stratification should respect both h1n1_vaccine and
        seasonal_vaccine distributions. Uses StratifiedKFold on combined label patterns
        to ensure each fold has representative examples of all 4 vaccine combinations:
        (0,0), (0,1), (1,0), (1,1).
        
        Args:
            X_train: Training features (if None, loads internally)
            y_train: Training labels (if None, loads internally)
        
        Returns:
            List of DataSplit objects, one per fold, containing:
            - X_train/y_train: Fold training set
            - X_val/y_val: Fold validation set
            - X_test/respondent_ids_test: Same test set for all folds
            
        Raises:
            ValueError: If n_folds < 2 or data is invalid
        """
        pass

    @abstractmethod
    def validate(self) -> DataValidationResult:
        """
        Validate data integrity and report summary statistics.
        
        Checks include:
        - File accessibility and format
        - Expected number of rows and columns
        - Missing value patterns
        - Class distribution for both targets
        - Respondent ID uniqueness
        - Target value ranges (should be 0 or 1)
        - Feature data types and ranges
        
        Returns:
            DataValidationResult with:
            - is_valid: True if all critical checks pass
            - Detailed breakdown of issues and warnings
            
        Example output:
            >>> result = loader.validate()
            >>> print(f"Valid: {result.is_valid}, Samples: {result.n_samples}")
            >>> for feature, missing_pct in result.missing_by_feature.items():
            ...     if missing_pct > 0:
            ...         print(f"{feature}: {missing_pct:.1%} missing")
        """
        pass

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names (excluding respondent_id).
        
        Returns:
            List of 35 feature names in order:
            - 5 opinion features
            - 6 behavioral features
            - 5 medical features
            - 8 demographic features
            - 2 household features
            - 2 geographic features
            - 2 concern/knowledge features
            - 3 employment features (may contain NaN)
            
        See PROBLEM_DESCRIPTION.md for detailed feature definitions.
        """
        return [
            # Opinions (5)
            "opinion_h1n1_vacc_effective",
            "opinion_h1n1_risk",
            "opinion_h1n1_sick_from_vacc",
            "opinion_seas_vacc_effective",
            "opinion_seas_risk",
            "opinion_seas_sick_from_vacc",
            # Behavioral (6)
            "behavioral_antiviral_meds",
            "behavioral_avoidance",
            "behavioral_face_mask",
            "behavioral_large_gatherings",
            "behavioral_outside_home",
            "behavioral_touch_face",
            # Medical (5)
            "doctor_recc_h1n1",
            "doctor_recc_seasonal",
            "chronic_med_condition",
            "health_worker",
            "health_insurance",
            # Demographics (8)
            "age_group",
            "education",
            "race",
            "sex",
            "income_poverty",
            "marital_status",
            "rent_or_own",
            "employment_status",
            # Household (2)
            "household_adults",
            "household_children",
            # Geographic (2)
            "hhs_geo_region",
            "census_msa",
            # Concern & Knowledge (2)
            "h1n1_concern",
            "h1n1_knowledge",
            # Employment (3)
            "employment_industry",
            "employment_occupation",
        ]

    def get_target_names(self) -> List[str]:
        """
        Get target column names.
        
        Returns:
            List: ["h1n1_vaccine", "seasonal_vaccine"]
        """
        return ["h1n1_vaccine", "seasonal_vaccine"]

    def get_id_column(self) -> str:
        """
        Get the respondent ID column name.
        
        Returns:
            str: "respondent_id"
        """
        return "respondent_id"


class CSVDataLoader(DataLoader):
    """
    Concrete DataLoader implementation for CSV files.
    
    Loads H1N1 flu shot prediction data from CSV files following the DrivenData
    competition format. Handles file loading, preprocessing, and cross-validation
    splitting for multilabel classification.
    
    Expected file formats:
    - training_set_features.csv: respondent_id + 35 features
    - training_set_labels.csv: respondent_id, h1n1_vaccine (0/1), seasonal_vaccine (0/1)
    - test_set_features.csv: respondent_id + 35 features
    
    Attributes:
        config: DataConfig instance with file paths and CV settings
        _X_train: Cached training features (loaded on first access)
        _y_train: Cached training labels (loaded on first access)
        _X_test: Cached test features (loaded on first access)
    """

    def __init__(self, config):
        """
        Initialize CSVDataLoader with configuration.
        
        Args:
            config: DataConfig instance with file paths and CV settings
                   (train_features_path, train_labels_path, test_features_path,
                    n_folds, random_seed, stratify, etc.)
        """
        self.config = config
        self._X_train = None
        self._y_train = None
        self._X_test = None

    def load_train(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training features and labels from CSV files.
        
        Reads training_set_features.csv and training_set_labels.csv, merges them
        on respondent_id, and returns features and labels as separate DataFrames.
        
        Implementation notes:
        - TODO: Read training_set_features.csv
        - TODO: Read training_set_labels.csv
        - TODO: Merge on respondent_id
        - TODO: Cache internally
        - TODO: Return (X_train, y_train)
        
        Returns:
            Tuple of (X_train, y_train) where:
            - X_train: DataFrame with respondent_id + 35 features
            - y_train: DataFrame with respondent_id, h1n1_vaccine, seasonal_vaccine
            
        Raises:
            FileNotFoundError: If CSV files not found
            ValueError: If files are missing required columns
        """
        # TODO: Implement file reading and merging
        pass

    def load_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load test features and respondent IDs from CSV file.
        
        Reads test_set_features.csv and extracts respondent_id for submission
        formatting.
        
        Implementation notes:
        - TODO: Read test_set_features.csv
        - TODO: Extract respondent_id column
        - TODO: Cache internally
        - TODO: Return (X_test, respondent_ids)
        
        Returns:
            Tuple of (X_test, respondent_ids) where:
            - X_test: DataFrame with respondent_id + 35 features
            - respondent_ids: Series of test respondent IDs
            
        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If file is missing respondent_id column
        """
        # TODO: Implement file reading
        pass

    def create_splits(
        self,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
    ) -> List[DataSplit]:
        """
        Create stratified k-fold splits for multilabel classification.
        
        For multilabel problems with two targets (h1n1_vaccine, seasonal_vaccine),
        this creates stratified splits on combined label patterns to ensure each
        fold has balanced representation of all 4 label combinations:
        (0,0), (0,1), (1,0), (1,1).
        
        Implementation notes:
        - TODO: Load training data if not provided
        - TODO: Load test data
        - TODO: Combine h1n1_vaccine and seasonal_vaccine into label patterns
        - TODO: Use StratifiedKFold with n_folds from config
        - TODO: For each fold, create DataSplit with train/val/test data
        - TODO: Return list of DataSplit objects
        
        Args:
            X_train: Training features (optional, loads internally if None)
            y_train: Training labels (optional, loads internally if None)
            
        Returns:
            List of DataSplit objects, one per fold with:
            - X_train: Fold training features
            - y_train: Fold training labels
            - X_val: Fold validation features
            - y_val: Fold validation labels
            - X_test: Test features (same for all folds)
            - respondent_ids_test: Test respondent IDs (same for all folds)
            
        Raises:
            ValueError: If n_folds < 2 or training data is invalid
        """
        # TODO: Implement stratified k-fold splitting
        pass

    def validate(self) -> DataValidationResult:
        """
        Validate data integrity and generate validation report.
        
        Performs comprehensive validation checks on training and test data,
        including file accessibility, shape consistency, missing values,
        class distribution, and target value ranges.
        
        Implementation notes:
        - TODO: Load training and test data
        - TODO: Check file accessibility
        - TODO: Verify expected column counts (35 features + respondent_id)
        - TODO: Check for missing values by column
        - TODO: Compute class distribution for both targets
        - TODO: Verify respondent_id uniqueness
        - TODO: Verify target values are 0 or 1
        - TODO: Infer and report feature data types
        - TODO: Collect issues (must-fail) and warnings (concerning but not failing)
        - TODO: Return DataValidationResult
        
        Returns:
            DataValidationResult with:
            - is_valid: True if no critical issues found
            - n_samples: Number of training samples
            - n_features: Number of features (should be 35)
            - missing_by_feature: Dict of feature -> % missing
            - class_distribution: Dict of target -> {0: %, 1: %}
            - feature_types: Dict of feature -> inferred type
            - issues: Critical validation failures
            - warnings: Non-critical concerns
            
        Example output structure:
            >>> result = loader.validate()
            >>> print(f"Valid: {result.is_valid}")
            >>> print(f"Samples: {result.n_samples}, Features: {result.n_features}")
            >>> for issue in result.issues:
            ...     print(f"ERROR: {issue}")
            >>> for warning in result.warnings:
            ...     print(f"WARNING: {warning}")
        """
        # TODO: Implement comprehensive data validation
        pass
