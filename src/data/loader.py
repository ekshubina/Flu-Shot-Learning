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
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


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
        # Return cached data if available
        if self._X_train is not None and self._y_train is not None:
            return self._X_train, self._y_train
        
        # Load training features
        feat_path = Path(self.config.train_features_path)
        if not feat_path.exists():
            raise FileNotFoundError(f"Training features file not found: {feat_path}")
        
        X_train = pd.read_csv(feat_path)
        if 'respondent_id' not in X_train.columns:
            raise ValueError(f"respondent_id column not found in {feat_path}")
        
        # Load training labels
        label_path = Path(self.config.train_labels_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Training labels file not found: {label_path}")
        
        y_train = pd.read_csv(label_path)
        if 'respondent_id' not in y_train.columns:
            raise ValueError(f"respondent_id column not found in {label_path}")
        if 'h1n1_vaccine' not in y_train.columns or 'seasonal_vaccine' not in y_train.columns:
            raise ValueError(f"Target columns (h1n1_vaccine, seasonal_vaccine) not found in {label_path}")
        
        # Cache and return
        self._X_train = X_train
        self._y_train = y_train
        
        return self._X_train, self._y_train

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
        # Return cached data if available
        if self._X_test is not None:
            return self._X_test, self._X_test['respondent_id']
        
        # Load test features
        test_path = Path(self.config.test_features_path)
        if not test_path.exists():
            raise FileNotFoundError(f"Test features file not found: {test_path}")
        
        X_test = pd.read_csv(test_path)
        if 'respondent_id' not in X_test.columns:
            raise ValueError(f"respondent_id column not found in {test_path}")
        
        # Cache and return
        self._X_test = X_test
        
        return self._X_test, self._X_test['respondent_id']

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
        # Load data if not provided
        if X_train is None or y_train is None:
            X_train, y_train = self.load_train()
        
        X_test, respondent_ids_test = self.load_test()
        
        # Validate n_folds
        if self.config.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.config.cv_folds}")
        
        # Create combined stratification column (4 classes for all label combinations)
        # combined_label = h1n1_vaccine + 2*seasonal_vaccine
        # (0,0) -> 0, (1,0) -> 1, (0,1) -> 2, (1,1) -> 3
        combined_labels = y_train['h1n1_vaccine'] + 2 * y_train['seasonal_vaccine']
        
        # Perform stratified k-fold split
        skf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_seed
        )
        
        splits = []
        for train_idx, val_idx in skf.split(X_train, combined_labels):
            # Get fold data
            X_train_fold = X_train.iloc[train_idx].reset_index(drop=True)
            y_train_fold = y_train.iloc[train_idx].reset_index(drop=True)
            X_val_fold = X_train.iloc[val_idx].reset_index(drop=True)
            y_val_fold = y_train.iloc[val_idx].reset_index(drop=True)
            
            # Create DataSplit
            split = DataSplit(
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                X_test=X_test,
                respondent_ids_test=respondent_ids_test,
            )
            splits.append(split)
        
        return splits

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
        issues = []
        warnings = []
        
        # Try loading data
        try:
            X_train, y_train = self.load_train()
            X_test, _ = self.load_test()
        except FileNotFoundError as e:
            issues.append(f"File not found: {e}")
            return DataValidationResult(
                is_valid=False,
                n_samples=0,
                n_features=0,
                missing_by_feature={},
                class_distribution={},
                feature_types={},
                issues=issues,
                warnings=warnings,
            )
        
        # Check feature counts (35 features + respondent_id = 36 columns)
        n_features = len(X_train.columns) - 1  # Exclude respondent_id
        if n_features != 35:
            issues.append(f"Expected 35 features + respondent_id, got {len(X_train.columns)} columns")
        
        # Check target columns
        for target in self.get_target_names():
            if target not in y_train.columns:
                issues.append(f"Target column '{target}' not found in labels")
        
        # Check respondent_id uniqueness in features
        if X_train['respondent_id'].nunique() != len(X_train):
            issues.append("Non-unique respondent_id values in training features")
        if y_train['respondent_id'].nunique() != len(y_train):
            issues.append("Non-unique respondent_id values in training labels")
        if X_test['respondent_id'].nunique() != len(X_test):
            issues.append("Non-unique respondent_id values in test features")
        
        # Check that train features and labels align
        if len(X_train) != len(y_train):
            issues.append(f"Train features ({len(X_train)}) and labels ({len(y_train)}) have different lengths")
        
        # Check target value ranges (should be 0 or 1)
        for target in self.get_target_names():
            if target in y_train.columns:
                invalid_values = y_train[~y_train[target].isin([0, 1, np.nan])]
                if len(invalid_values) > 0:
                    issues.append(f"Non-binary values found in {target}")
        
        # Compute missing values
        missing_by_feature = {}
        for col in X_train.columns:
            if col != 'respondent_id':
                pct_missing = X_train[col].isna().sum() / len(X_train) * 100
                missing_by_feature[col] = pct_missing
                if pct_missing > 50:
                    warnings.append(f"{col}: {pct_missing:.1f}% missing")
        
        # Compute class distribution
        class_distribution = {}
        for target in self.get_target_names():
            if target in y_train.columns:
                dist = y_train[target].value_counts(normalize=True) * 100
                class_distribution[target] = {int(k): float(v) for k, v in dist.items()}
        
        # Infer feature types
        feature_types = {}
        for col in X_train.columns:
            if col != 'respondent_id':
                if X_train[col].dtype == 'object':
                    feature_types[col] = 'categorical'
                elif X_train[col].dtype in ['int64', 'int32']:
                    # Check if it's actually categorical (low cardinality)
                    if X_train[col].nunique() <= 10:
                        feature_types[col] = 'ordinal'
                    else:
                        feature_types[col] = 'numeric'
                else:
                    feature_types[col] = 'numeric'
        
        # Determine overall validity
        is_valid = len(issues) == 0
        
        return DataValidationResult(
            is_valid=is_valid,
            n_samples=len(X_train),
            n_features=n_features,
            missing_by_feature=missing_by_feature,
            class_distribution=class_distribution,
            feature_types=feature_types,
            issues=issues,
            warnings=warnings,
        )
