"""
Missing value imputation strategies.

This module defines the abstract interface for imputation strategies and provides
concrete implementations for various approaches to handling missing values in the
H1N1 flu shot prediction dataset.

Imputation strategies are applied after data loading but before feature encoding.
Different strategies are suitable for different feature types:
- Binary/categorical: Mode or flag-as-missing imputation
- Ordinal: Mean or KNN imputation
- Continuous: Mean, KNN, or MICE imputation

See CONTEXT_REPORT.md for analysis of missing data patterns in the dataset.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer as SklearnKNNImputer


class ImputationStrategy(ABC):
    """
    Abstract base class for missing value imputation strategies.
    
    Imputation strategies transform datasets by filling missing values using
    different approaches. Each strategy must implement fit() and transform()
    methods following the scikit-learn pattern.
    
    The strategy should handle:
    - Different feature types (binary, categorical, ordinal, continuous)
    - Column-specific handling (some columns may need different treatment)
    - Preservation of data types and valid value ranges
    - Caching of fitted parameters for consistent test set transformation
    
    Attributes:
        feature_names: List of feature column names
        fitted: Whether the strategy has been fit to training data
        fit_params: Dictionary of parameters learned from training data
        
    Example:
        >>> from src.preprocessing.imputation import MeanImputation
        >>> strategy = MeanImputation()
        >>> X_train_imputed = strategy.fit_transform(X_train)
        >>> X_test_imputed = strategy.transform(X_test)
        >>> X_val_imputed = strategy.transform(X_val)
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize imputation strategy.
        
        Args:
            feature_names: List of feature column names (optional, can be set later)
        """
        self.feature_names = feature_names
        self.fitted = False
        self.fit_params = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "ImputationStrategy":
        """
        Fit imputation strategy on training data.
        
        Learns any parameters needed for transformation (e.g., mean values,
        KNN neighbors, MICE regression coefficients). Must be called before
        transform() on different data.
        
        Args:
            X: Training features DataFrame with potential missing values
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If X is invalid or empty
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset by filling missing values.
        
        Uses parameters learned in fit() to impute missing values in new data.
        Must call fit() first on training data.
        
        Args:
            X: Features DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values filled, same shape as input
            
        Raises:
            ValueError: If strategy not fitted yet
            ValueError: If X has different columns than training data
        """
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit imputation strategy and transform data in one step.
        
        Convenience method equivalent to fit(X).transform(X).
        
        Args:
            X: Training features DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values filled
        """
        return self.fit(X).transform(X)

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names.
        
        Returns:
            List of feature names
            
        Raises:
            ValueError: If feature names not set
        """
        if self.feature_names is None:
            raise ValueError("Feature names not set. Pass feature_names to constructor or set before transform.")
        return self.feature_names


class DropRowsImputation(ImputationStrategy):
    """
    Strategy that removes rows with any missing values.
    
    Simple but aggressive approach: drops entire rows containing any NaN values.
    Suitable for datasets with small amounts of missing data distributed across
    many rows, or when missing data is assumed to be MCAR (missing completely
    at random).
    
    Trade-offs:
    - Pro: Simple, no assumptions, no information loss
    - Con: Loses all data from affected rows, may lose significant data
    
    Implementation notes:
        - TODO: In fit(), store which columns have any missing values
        - TODO: In transform(), drop rows with NaN values
        - TODO: Log number of rows dropped for transparency
    """

    def fit(self, X: pd.DataFrame) -> "DropRowsImputation":
        """
        Fit by analyzing missing data pattern.
        
        Implementation notes:
            - TODO: Count missing values per column
            - TODO: Store as fit_params for reporting
        """
        # TODO: Implement
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with any missing values.
        
        Implementation notes:
            - TODO: Drop rows where any column is NaN
            - TODO: Return cleaned DataFrame
        """
        # TODO: Implement
        pass


class DropColumnsImputation(ImputationStrategy):
    """
    Strategy that removes columns with excessive missing values.
    
    Removes entire columns where missing percentage exceeds a threshold.
    Suitable for columns with systematic missingness (e.g., employment columns
    that are NA for unemployed respondents).
    
    Configuration:
        - drop_threshold: Remove columns with > this % missing (default: 0.5)
    
    Trade-offs:
    - Pro: Removes uninformative columns, preserves rows
    - Con: Loses feature information, threshold choice is arbitrary
    
    Implementation notes:
        - TODO: In fit(), compute % missing per column, mark columns to drop
        - TODO: In transform(), drop marked columns
        - TODO: Store dropped column names in fit_params
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        drop_threshold: float = 0.5,
    ):
        """
        Initialize with drop threshold.
        
        Args:
            feature_names: List of feature column names
            drop_threshold: Drop columns with > this fraction missing (0.0-1.0)
        """
        super().__init__(feature_names)
        self.drop_threshold = drop_threshold

    def fit(self, X: pd.DataFrame) -> "DropColumnsImputation":
        """
        Identify columns to drop based on missing data threshold.
        
        Implementation notes:
            - TODO: Compute % missing for each column
            - TODO: Mark columns exceeding drop_threshold
            - TODO: Store list of columns to drop in fit_params
        """
        # TODO: Implement
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns marked during fit.
        
        Implementation notes:
            - TODO: Drop columns stored in fit_params['columns_to_drop']
            - TODO: Return DataFrame with dropped columns removed
        """
        # TODO: Implement
        pass


class MeanImputation(ImputationStrategy):
    """
    Strategy that fills missing values with column means.
    
    For numerical features, imputes missing values with the mean value computed
    from training data. Simple and fast, assumes MCAR and that mean is reasonable.
    
    Suitable for:
    - Ordinal/numerical features with sparse missing data
    - Features where mean is interpretable
    - Quick baseline imputation
    
    Trade-offs:
    - Pro: Simple, fast, preserves sample size
    - Con: Reduces variance, may not be suitable for binary/categorical
    
    Implementation notes:
        - TODO: In fit(), compute mean for each numerical column
        - TODO: In transform(), fill NaN with stored means
        - TODO: Handle categorical/binary columns separately if needed
    """

    def fit(self, X: pd.DataFrame) -> "MeanImputation":
        """
        Compute column means on training data.
        
        Implementation notes:
            - TODO: For each column, compute mean (skip non-numeric)
            - TODO: Store means in fit_params['column_means']
        """
        if X.empty:
            raise ValueError("Training data X cannot be empty")
        
        # Compute means for all numeric columns
        self.fit_params['column_means'] = X.select_dtypes(include=[np.number]).mean()
        self.feature_names = list(X.columns)
        self.fitted = True
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with learned means.
        
        Implementation notes:
            - TODO: For each column with stored mean, fill NaN values
            - TODO: Return DataFrame with NaN filled
        """
        if not self.fitted:
            raise ValueError("MeanImputation strategy must be fit before transform")
        
        # Create a copy to avoid modifying input
        X_imputed = X.copy()
        
        # Fill missing values in numeric columns with learned means
        for column, mean_value in self.fit_params['column_means'].items():
            if column in X_imputed.columns:
                X_imputed[column].fillna(mean_value, inplace=True)
        
        return X_imputed


class ModeImputation(ImputationStrategy):
    """
    Strategy that fills missing values with column modes.
    
    For categorical/binary features, imputes missing values with the most common
    value computed from training data. Suitable for binary and categorical features.
    
    Suitable for:
    - Binary features (0/1)
    - Categorical features with small cardinality
    - Features where mode (most common value) is reasonable default
    
    Trade-offs:
    - Pro: Preserves categorical structure, simple and fast
    - Con: Reduces variance, mode may not be representative
    
    Implementation notes:
        - TODO: In fit(), compute mode for each column
        - TODO: In transform(), fill NaN with stored modes
        - TODO: Handle ties in mode selection
    """

    def fit(self, X: pd.DataFrame) -> "ModeImputation":
        """
        Compute column modes on training data.
        
        Implementation notes:
            - TODO: For each column, compute mode (most common value)
            - TODO: Handle ties (multiple modes) by choosing first/most common
            - TODO: Store modes in fit_params['column_modes']
        """
        if X.empty:
            raise ValueError("Training data X cannot be empty")
        
        # Compute mode for all columns (numeric, categorical, mixed)
        # mode() returns a DataFrame with modes; we extract the first mode for each column
        self.fit_params['column_modes'] = X.mode(dropna=True).iloc[0] if len(X.mode(dropna=True)) > 0 else X.mode(dropna=False).iloc[0]
        self.feature_names = list(X.columns)
        self.fitted = True
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with learned modes.
        
        Implementation notes:
            - TODO: For each column with stored mode, fill NaN values
            - TODO: Return DataFrame with NaN filled
        """
        if not self.fitted:
            raise ValueError("ModeImputation strategy must be fit before transform")
        
        # Create a copy to avoid modifying input
        X_imputed = X.copy()
        
        # Fill missing values in all columns with learned modes
        for column, mode_value in self.fit_params['column_modes'].items():
            if column in X_imputed.columns:
                X_imputed[column].fillna(mode_value, inplace=True)
        
        return X_imputed


class KNNImputation(ImputationStrategy):
    """
    Strategy that imputes missing values using k-nearest neighbors.
    
    For each sample with missing values, finds k nearest neighbors (based on
    non-missing features) and imputes using mean/median of neighbors' values.
    More sophisticated than mean/mode, captures local structure in data.
    
    Configuration:
        - n_neighbors: Number of neighbors to use (default: 5)
        - weights: 'uniform' (equal weight) or 'distance' (weighted by distance)
    
    Suitable for:
    - Mixed feature types
    - When local structure matters (e.g., respondents similar in other features
      likely have similar missing values)
    - Datasets with sparse missing data
    
    Trade-offs:
    - Pro: More sophisticated than mean/mode, captures local patterns
    - Con: Computationally expensive (scales with data size), requires distance metric
    
    Implementation notes:
        - TODO: In fit(), prepare for KNN (fit KDTree or similar on non-missing features)
        - TODO: In transform(), for each row with missing values:
        -   TODO: Find k nearest neighbors from training data
        -   TODO: Impute using neighbors' values
        - TODO: Handle all-missing columns
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        n_neighbors: int = 5,
        weights: str = "uniform",
    ):
        """
        Initialize with KNN parameters.
        
        Args:
            feature_names: List of feature column names
            n_neighbors: Number of neighbors to use (default: 5)
            weights: 'uniform' or 'distance' weighting (default: 'uniform')
        """
        super().__init__(feature_names)
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X: pd.DataFrame) -> "KNNImputation":
        """
        Prepare KNN structure on training data.
        
        Implementation notes:
            - Store training data for neighbor lookup
            - Build KDTree or similar structure for efficient neighbors
            - Handle features with all missing values
        """
        if X.empty:
            raise ValueError("Training data X cannot be empty")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Create and fit sklearn's KNNImputer
        self.fit_params['knn_imputer'] = SklearnKNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
        
        # Fit the KNN imputer on training data (numeric columns)
        # KNNImputer handles numeric data; it will skip non-numeric columns
        self.fit_params['knn_imputer'].fit(X)
        self.fitted = True
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using k-nearest neighbors.
        
        Implementation notes:
            - For each row with missing values:
            -   Find k nearest neighbors from training data
            -   Compute mean/median of neighbors' values
            -   Fill NaN with computed values
            - Return imputed DataFrame
        """
        if not self.fitted:
            raise ValueError("KNNImputation strategy must be fit before transform")
        
        # Apply the fitted KNN imputer
        X_imputed_array = self.fit_params['knn_imputer'].transform(X)
        
        # Convert back to DataFrame with original column names
        X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)
        
        return X_imputed


class MICEImputation(ImputationStrategy):
    """
    Strategy using Multivariate Imputation by Chained Equations (MICE).
    
    Iteratively imputes missing values by fitting regression models for each
    feature with missing values, using other features as predictors. Creates
    multiple imputations and averages (though this implementation does single imputation).
    
    Configuration:
        - max_iter: Number of MICE iterations (default: 10)
        - random_state: Random seed for reproducibility
    
    Suitable for:
    - Complex multivariate missing data patterns
    - When relationship between features should be preserved
    - When MAR (missing at random) assumption is reasonable
    
    Trade-offs:
    - Pro: Sophisticated, preserves relationships between features
    - Con: Computationally expensive, complex to understand, requires assumptions
    
    Implementation notes:
        - TODO: In fit(), initialize by simple mean/mode imputation
        - TODO: In transform(), iteratively fit regression for each feature
        -   TODO: For each feature with missing, fit regression on other features
        -   TODO: Predict missing values
        -   TODO: Repeat max_iter times
        - TODO: Use final predictions as imputations
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        max_iter: int = 10,
        random_state: Optional[int] = None,
    ):
        """
        Initialize with MICE parameters.
        
        Args:
            feature_names: List of feature column names
            max_iter: Number of iteration cycles (default: 10)
            random_state: Random seed for reproducibility
        """
        super().__init__(feature_names)
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: pd.DataFrame) -> "MICEImputation":
        """
        Initialize MICE on training data.
        
        Implementation notes:
            - TODO: Store training data stats (means/modes for initial imputation)
            - TODO: Identify features with missing values
            - TODO: Identify predictor features for each target
        """
        # TODO: Implement
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using MICE algorithm.
        
        Implementation notes:
            - TODO: Start with mean/mode imputation
            - TODO: For each iteration (max_iter times):
            -   TODO: For each feature with missing values:
            -     TODO: Fit regression using other features as predictors
            -     TODO: Predict missing values
            - TODO: Return final imputed DataFrame
        """
        # TODO: Implement
        pass


class FlagAsMissingImputation(ImputationStrategy):
    """
    Strategy that fills missing values and creates indicator columns.
    
    Imputes missing values (using mean/mode) AND creates binary "was_missing"
    indicator columns to preserve information that value was originally missing.
    Useful when missingness itself is informative.
    
    Configuration:
        - base_strategy: Strategy to use for actual imputation ('mean' or 'mode')
        - prefix: Prefix for indicator columns (default: 'missing_')
    
    Trade-offs:
    - Pro: Preserves missingness information, can improve models
    - Con: Doubles feature count, may overfit if missingness is sparse
    
    Output shape: 2 * n_features (original features + indicator columns)
    
    Implementation notes:
        - TODO: In fit():
        -   TODO: Create a base imputation strategy (MeanImputation or ModeImputation)
        -   TODO: Fit base strategy
        -   TODO: Identify columns with missing values
        - TODO: In transform():
        -   TODO: Identify which columns are missing in this data
        -   TODO: Apply base strategy imputation
        -   TODO: Create indicator columns for originally-missing values
        -   TODO: Return DataFrame with original features + indicators
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        base_strategy: str = "mean",
    ):
        """
        Initialize with flag-as-missing strategy.
        
        Args:
            feature_names: List of feature column names
            base_strategy: Base imputation strategy - 'mean' or 'mode'
        """
        super().__init__(feature_names)
        self.base_strategy = base_strategy
        self.base_imputer = None

    def fit(self, X: pd.DataFrame) -> "FlagAsMissingImputation":
        """
        Initialize base imputation strategy.
        
        Implementation notes:
            - TODO: Create MeanImputation or ModeImputation based on base_strategy
            - TODO: Fit base strategy on X
            - TODO: Store which columns have any missing values
        """
        # TODO: Implement
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values and create indicator columns.
        
        Implementation notes:
            - TODO: Apply base imputation
            - TODO: Create binary indicator columns for originally-missing values
            - TODO: Column names: 'missing_' + original_column_name
            - TODO: Return DataFrame with original features + indicators
        """
        # TODO: Implement
        pass
