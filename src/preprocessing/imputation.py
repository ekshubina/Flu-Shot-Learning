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
from typing import Optional, List, Dict, Tuple, Any, Union
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer as SklearnKNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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
        
        Analyzes which columns contain any missing values and stores the pattern
        for reference. Rows will be dropped during transform().
        """
        if X.empty:
            raise ValueError("Training data X cannot be empty")
        
        # Count missing values per column
        self.fit_params['columns_with_missing'] = [
            col for col in X.columns if X[col].isna().any()
        ]
        self.feature_names = list(X.columns)
        self.fitted = True
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with any missing values.
        
        Drops all rows containing any NaN values and returns the cleaned DataFrame.
        """
        if not self.fitted:
            raise ValueError("DropRowsImputation strategy must be fit before transform")
        
        # Drop rows with any NaN values
        X_cleaned = X.dropna()
        
        return X_cleaned


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
        
        Computes the percentage of missing values for each column and identifies
        columns exceeding the drop_threshold. Stores these column names for removal
        during transform().
        """
        if X.empty:
            raise ValueError("Training data X cannot be empty")
        
        if not (0.0 <= self.drop_threshold <= 1.0):
            raise ValueError(f"drop_threshold must be between 0.0 and 1.0, got {self.drop_threshold}")
        
        # Compute % missing per column
        columns_to_drop = []
        for col in X.columns:
            pct_missing = X[col].isna().sum() / len(X)
            if pct_missing > self.drop_threshold:
                columns_to_drop.append(col)
        
        self.fit_params['columns_to_drop'] = columns_to_drop
        self.feature_names = list(X.columns)
        self.fitted = True
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns marked during fit.
        
        Removes columns that were identified as exceeding the missing data threshold
        during fitting.
        """
        if not self.fitted:
            raise ValueError("DropColumnsImputation strategy must be fit before transform")
        
        # Drop marked columns (only if they exist in X)
        columns_to_drop = [
            col for col in self.fit_params['columns_to_drop'] if col in X.columns
        ]
        X_reduced = X.drop(columns=columns_to_drop)
        
        return X_reduced


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
        
        Fills missing values in each column with the mode (most common value)
        learned from the training data.
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
        # Type cast: ensure weights is 'uniform' or 'distance'
        weights_param: Union[str, Any] = 'uniform' if isinstance(self.weights, str) and self.weights == 'uniform' else 'distance'
        self.fit_params['knn_imputer'] = SklearnKNNImputer(
            n_neighbors=self.n_neighbors,
            weights=weights_param  # type: ignore
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


class OrdinalStringKNNImputation(ImputationStrategy):
    """
    Strategy that imputes missing ordinal values using KNN on encoded ordinal ranks.
    
    Handles ordinal features stored as strings (e.g., '18-34', '35-44', '45-54', ...)
    by encoding them to their ordinal rank (0, 1, 2, ...), applying KNN imputation,
    then decoding back to original ordinal values.
    
    Process:
    1. Determine ordinal order for each column (sorted list of unique values)
    2. Encode ordinal values: first value → 0, second → 1, etc. (NaN → NaN)
    3. Apply KNN imputation on encoded numeric ranks
    4. Decode imputed ranks back to original ordinal values
    5. Return ordinal column with missing values filled
    
    Example:
    - age_group: '18-34' → 0, '35-44' → 1, '45-54' → 2, '55-64' → 3, '65+' → 4
    - education: '<12 Years' → 0, '12 Years' → 1, 'Some College' → 2, 'College Grad' → 3
    
    Configuration:
        - n_neighbors: Number of neighbors to use (default: 5)
        - weights: 'uniform' (equal weight) or 'distance' (weighted by distance)
    
    Suitable for:
    - Ordinal features with natural order but string representation
    - Age groups, education levels, income brackets
    - Preserving ordinal structure in imputations
    
    Trade-offs:
    - Pro: Handles ordinal strings, preserves order, KNN effectiveness on numeric
    - Con: Assumes ordinal values are in sorted order; precision may degrade in decoding
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        n_neighbors: int = 5,
        weights: str = "uniform",
    ):
        """
        Initialize ordinal string KNN imputation.
        
        Args:
            feature_names: List of feature column names (all should be ordinal)
            n_neighbors: Number of neighbors to use (default: 5)
            weights: 'uniform' or 'distance' weighting (default: 'uniform')
        """
        super().__init__(feature_names)
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X: pd.DataFrame) -> "OrdinalStringKNNImputation":
        """
        Prepare KNN structure for ordinal string data.
        
        For each ordinal column:
        1. Extract unique values (excluding NaN) and sort them
        2. Create mapping: ordinal_value → rank (0, 1, 2, ...)
        3. Store mapping for later decoding
        4. Encode data to numeric ranks
        5. Fit KNN imputer on encoded numeric data
        
        Args:
            X: Training features DataFrame with ordinal string values (may have NaN)
            
        Returns:
            self (for method chaining)
        """
        if X.empty:
            raise ValueError("Training data X cannot be empty")
        
        self.feature_names = list(X.columns)
        
        # Step 1: Create ordinal mappings for each column
        ordinal_maps = {}   # column_name → {ordinal_value: rank}
        reverse_maps = {}   # column_name → {rank: ordinal_value}
        X_encoded = X.copy()
        
        for col in X.columns:
            # Get unique values (excluding NaN) and sort them
            # Sorted order defines the ordinal rank
            unique_vals = sorted(X[col].dropna().unique())
            
            # Create mapping: value → rank (0, 1, 2, ...)
            col_map = {val: idx for idx, val in enumerate(unique_vals)}
            # Reverse map: rank → value
            rev_map = {idx: val for val, idx in col_map.items()}
            
            ordinal_maps[col] = col_map
            reverse_maps[col] = rev_map
            
            # Encode column: ordinal_value → rank, NaN → NaN
            X_encoded[col] = X[col].map(col_map)
        
        self.fit_params['ordinal_maps'] = ordinal_maps
        self.fit_params['reverse_maps'] = reverse_maps
        
        # Step 2: Fit KNN imputer on encoded numeric data
        weights_param: Union[str, Any] = 'uniform' if isinstance(self.weights, str) and self.weights == 'uniform' else 'distance'
        self.fit_params['knn_imputer'] = SklearnKNNImputer(
            n_neighbors=self.n_neighbors,
            weights=weights_param  # type: ignore
        )
        self.fit_params['knn_imputer'].fit(X_encoded)
        
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing ordinal string values using KNN on encoded ranks.
        
        Process:
        1. Encode ordinal columns using fitted mappings
        2. Apply KNN imputation on encoded numeric ranks
        3. For imputed ranks, round to nearest integer and decode to ordinal values
        4. Return DataFrame with original ordinal string values
        
        Args:
            X: Features DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values filled using original ordinal values
            
        Raises:
            ValueError: If strategy not fitted yet
        """
        if not self.fitted:
            raise ValueError("OrdinalStringKNNImputation must be fit before transform")
        
        ordinal_maps = self.fit_params['ordinal_maps']
        reverse_maps = self.fit_params['reverse_maps']
        
        # Step 1: Encode ordinal columns
        X_encoded = X.copy()
        missing_mask = X.isna()  # Track where values were missing
        
        for col in X.columns:
            if col in ordinal_maps:
                col_map = ordinal_maps[col]
                # Map ordinal values to ranks; unmapped values (shouldn't happen) → NaN
                X_encoded[col] = X[col].map(col_map)
        
        # Step 2: Apply KNN imputation on encoded data
        X_imputed_array = self.fit_params['knn_imputer'].transform(X_encoded)
        X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)
        
        # Step 3: Decode imputed ranks back to ordinal values
        X_decoded = X.copy()  # Start with original (to preserve non-missing values)
        
        for col in X.columns:
            if col in reverse_maps:
                rev_map = reverse_maps[col]
                
                # For each value that was originally missing, decode the imputed rank
                col_missing_mask = missing_mask[col]
                if col_missing_mask.any():
                    # Get imputed ranks for this column
                    imputed_ranks = X_imputed[col][col_missing_mask]
                    
                    # Round to nearest integer (since KNN returns floats)
                    imputed_ranks_int = np.round(imputed_ranks).astype(int)
                    
                    # Clamp to valid range [0, max_rank]
                    max_rank = max(rev_map.keys())
                    imputed_ranks_int = np.clip(imputed_ranks_int, 0, max_rank)
                    
                    # Map ranks back to ordinal values
                    decoded_values = [rev_map[rank] for rank in imputed_ranks_int]
                    
                    # Update decoded column with imputed ordinal values
                    X_decoded.loc[col_missing_mask, col] = decoded_values
        
        return X_decoded


class CategoricalKNNImputation(ImputationStrategy):
    """
    Strategy that imputes missing categorical values using KNN on encoded categories.
    
    Handles categorical features (strings) by encoding them to ordinal numbers,
    applying KNN imputation, then decoding back to original categories using the
    mode (most common value) of the k nearest neighbors.
    
    Process:
    1. Encode each categorical column: category → integer (0, 1, 2, ...)
    2. Apply KNN imputation on encoded numeric values
    3. For each imputed value, find the original category from nearest neighbors
    4. Return categorical column with missing values filled
    
    Configuration:
        - n_neighbors: Number of neighbors to use (default: 5)
        - weights: 'uniform' (equal weight) or 'distance' (weighted by distance)
    
    Suitable for:
    - Categorical features (strings or objects)
    - Mixed categorical/numeric feature sets
    - Preserving categorical structure in imputations
    
    Trade-offs:
    - Pro: Handles categorical data properly, captures local patterns
    - Con: Computationally expensive, requires encoding/decoding, may lose some precision
    
    Implementation notes:
    - Stores category mappings for each column (fit_params['category_maps'])
    - Uses KNNImputer internally on encoded values
    - Decodes predictions back using nearest neighbors' modes
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        n_neighbors: int = 5,
        weights: str = "uniform",
    ):
        """
        Initialize categorical KNN imputation.
        
        Args:
            feature_names: List of feature column names (all should be categorical)
            n_neighbors: Number of neighbors to use (default: 5)
            weights: 'uniform' or 'distance' weighting (default: 'uniform')
        """
        super().__init__(feature_names)
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X: pd.DataFrame) -> "CategoricalKNNImputation":
        """
        Prepare KNN structure for categorical data.
        
        For each categorical column:
        1. Extract all unique categories (including NaN)
        2. Create mapping: category → integer code
        3. Store mapping for later decoding
        4. Encode data to numeric values
        5. Fit KNN imputer on encoded numeric data
        
        Args:
            X: Training features DataFrame with categorical values (may have NaN)
            
        Returns:
            self (for method chaining)
        """
        if X.empty:
            raise ValueError("Training data X cannot be empty")
        
        self.feature_names = list(X.columns)
        
        # Step 1: Create category mappings for each column
        category_maps = {}  # column_name → {category: code}
        reverse_maps = {}   # column_name → {code: category}
        X_encoded = X.copy()
        
        for col in X.columns:
            # Get unique categories (excluding NaN)
            categories = X[col].dropna().unique()
            
            # Create mapping: category → integer code (0, 1, 2, ...)
            col_map = {cat: idx for idx, cat in enumerate(sorted(categories))}
            # Reverse map: code → category
            rev_map = {idx: cat for cat, idx in col_map.items()}
            
            category_maps[col] = col_map
            reverse_maps[col] = rev_map
            
            # Encode column: category → code, NaN → NaN
            X_encoded[col] = X[col].map(col_map)
        
        self.fit_params['category_maps'] = category_maps
        self.fit_params['reverse_maps'] = reverse_maps
        
        # Step 2: Fit KNN imputer on encoded numeric data
        weights_param: Union[str, Any] = 'uniform' if isinstance(self.weights, str) and self.weights == 'uniform' else 'distance'
        self.fit_params['knn_imputer'] = SklearnKNNImputer(
            n_neighbors=self.n_neighbors,
            weights=weights_param  # type: ignore
        )
        self.fit_params['knn_imputer'].fit(X_encoded)
        
        # Step 3: Store training data (encoded) for neighbor lookups
        self.fit_params['X_train_encoded'] = X_encoded.fillna(-999)  # NaN → -999 as placeholder
        
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing categorical values using KNN on encoded data.
        
        Process:
        1. Encode categorical columns using fitted mappings
        2. Apply KNN imputation on encoded values
        3. For imputed values, find k nearest neighbors' modes
        4. Decode back to original categories
        5. Return DataFrame with original categorical values
        
        Args:
            X: Features DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values filled using original categories
            
        Raises:
            ValueError: If strategy not fitted yet
        """
        if not self.fitted:
            raise ValueError("CategoricalKNNImputation must be fit before transform")
        
        category_maps = self.fit_params['category_maps']
        reverse_maps = self.fit_params['reverse_maps']
        
        # Step 1: Encode categorical columns
        X_encoded = X.copy()
        missing_mask = X.isna()  # Track where values were missing
        
        for col in X.columns:
            if col in category_maps:
                col_map = category_maps[col]
                # Map categories to codes; unmapped values → NaN
                X_encoded[col] = X[col].map(col_map)
        
        # Step 2: Apply KNN imputation on encoded data
        X_imputed_array = self.fit_params['knn_imputer'].transform(X_encoded)
        X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)
        
        # Step 3: Decode imputed values back to categories
        X_decoded = X.copy()  # Start with original (to preserve non-missing values)
        
        for col in X.columns:
            if col in reverse_maps:
                rev_map = reverse_maps[col]
                
                # For each value that was originally missing, decode the imputed code
                col_missing_mask = missing_mask[col]
                if col_missing_mask.any():
                    # Get imputed codes for this column
                    imputed_codes = X_imputed[col][col_missing_mask]
                    
                    # Round to nearest integer and decode
                    imputed_codes_int = np.round(imputed_codes).astype(int)
                    
                    # Map codes back to categories (with safety for out-of-range)
                    decoded_values = []
                    for code in imputed_codes_int:
                        # Get the category for this code, or use mode if code not in mapping
                        if code in rev_map:
                            decoded_values.append(rev_map[code])
                        else:
                            # Fallback to mode (most common category)
                            mode_cat = X[col].mode()[0] if len(X[col].mode()) > 0 else list(rev_map.values())[0]
                            decoded_values.append(mode_cat)
                    
                    # Update decoded column with imputed categories
                    X_decoded.loc[col_missing_mask, col] = decoded_values
        
        return X_decoded


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
        
        Fits IterativeImputer (sklearn's implementation of MICE) on training data.
        IterativeImputer uses estimators to model relationships between features
        and imputes missing values iteratively.
        """
        if X.empty:
            raise ValueError("Training data X cannot be empty")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Create and fit sklearn's IterativeImputer (MICE implementation)
        # BayesianRidge is the default estimator and works well for mixed types
        self.fit_params['mice_imputer'] = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0
        )
        
        # Fit the imputer on training data
        self.fit_params['mice_imputer'].fit(X)
        self.fitted = True
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using MICE algorithm.
        
        Uses the fitted IterativeImputer to impute missing values in new data
        by modeling feature relationships learned from the training data.
        """
        if not self.fitted:
            raise ValueError("MICEImputation strategy must be fit before transform")
        
        # Apply the fitted MICE imputer
        X_imputed_array = self.fit_params['mice_imputer'].transform(X)
        
        # Convert back to DataFrame with original column names
        X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)
        
        return X_imputed


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
        # Track which columns have missing values
        self.missing_columns_ = X.columns[X.isnull().any()].tolist()
        
        # Create and fit base imputer
        if self.base_strategy == 'mean':
            from sklearn.impute import SimpleImputer
            self.base_imputer = SimpleImputer(strategy='mean')
            self.base_imputer.fit(X.select_dtypes(include=[np.number]))
        else:  # mode
            from sklearn.impute import SimpleImputer
            self.base_imputer = SimpleImputer(strategy='most_frequent')
            self.base_imputer.fit(X)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values and create indicator columns.
        
        Implementation notes:
            - TODO: Apply base imputation
            - TODO: Create binary indicator columns for originally-missing values
            - TODO: Column names: 'missing_' + original_column_name
            - TODO: Return DataFrame with original features + indicators
        """
        X = X.copy()
        
        # Create indicator columns for originally missing values
        for col in self.missing_columns_:
            X[f'missing_{col}'] = X[col].isnull().astype(int)
        
        # Apply base imputation
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols and self.base_imputer is not None:
            X[numeric_cols] = self.base_imputer.transform(X[numeric_cols])
        
        # Fill remaining missing categorical values
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown', inplace=True)
        
        return X


class TypeBasedImputation(ImputationStrategy):
    """
    Strategy that applies different imputation methods to different feature types.
    
    Enables flexible, per-type strategy specification:
    - Ordinal features (and binary numeric): mean, median, knn, or mice
    - Nominal features: mode or mice
    
    Composes existing strategy classes (MeanImputation, ModeImputation, KNNImputation)
    rather than reimplementing logic. Applies each strategy to its designated columns.
    
    Configuration example (YAML):
        imputation:
            type: 'type_based'
            ordinal_strategy: 'mean'  # or knn, median
            nominal_strategy: 'mode'  # only option for nominal
            ordinal_params: {n_neighbors: 5}  # for knn strategy
            nominal_params: {}
    
    Attributes:
        ordinal_columns: List of ordinal and binary numeric feature names
        nominal_columns: List of nominal feature names
        ordinal_strategy: Strategy name for ordinal columns ('mean', 'median', 'knn', 'mice')
        nominal_strategy: Strategy name for nominal columns ('mode', 'mice')
        ordinal_params: Parameters dict for ordinal strategy
        nominal_params: Parameters dict for nominal strategy
        ordinal_imputer: Fitted imputation strategy for ordinal columns
        nominal_imputer: Fitted imputation strategy for nominal columns
    """
    
    def __init__(
        self,
        ordinal_columns: List[str],
        nominal_columns: List[str],
        ordinal_strategy: str = "mean",
        nominal_strategy: str = "mode",
        ordinal_params: Optional[Dict[str, Any]] = None,
        nominal_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize type-based imputation strategy.
        
        Args:
            ordinal_columns: List of ordinal/binary numeric feature names
            nominal_columns: List of nominal feature names
            ordinal_strategy: Strategy for ordinal columns (mean, median, knn, mice)
            nominal_strategy: Strategy for nominal columns (mode, mice)
            ordinal_params: Parameters dict for ordinal strategy (default: {})
            nominal_params: Parameters dict for nominal strategy (default: {})
        """
        super().__init__()
        self.ordinal_columns = ordinal_columns
        self.nominal_columns = nominal_columns
        self.ordinal_strategy = ordinal_strategy
        self.nominal_strategy = nominal_strategy
        self.ordinal_params = ordinal_params or {}
        self.nominal_params = nominal_params or {}
        
        self.ordinal_imputer = None
        self.nominal_imputer = None
    
    def fit(self, X: pd.DataFrame) -> "TypeBasedImputation":
        """
        Fit imputation strategies on training data.
        
        Creates strategy instances for each type and fits them on respective columns.
        Special handling for KNN:
        - Numeric ordinal columns: standard KNNImputation
        - Categorical nominal columns: CategoricalKNNImputation (encode→KNN→decode)
        - Non-numeric ordinal columns: fall back to mode
        
        Args:
            X: Training features DataFrame with potential missing values
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If X is empty or doesn't contain required columns
        """
        if X.empty:
            raise ValueError("Training data X cannot be empty")
        
        self.feature_names = list(X.columns)
        
        # Filter to columns that actually exist in X
        ordinal_cols = [c for c in self.ordinal_columns if c in X.columns]
        nominal_cols = [c for c in self.nominal_columns if c in X.columns]
        
        if not ordinal_cols and not nominal_cols:
            raise ValueError(
                f"No valid columns found. ordinal_columns: {self.ordinal_columns}, "
                f"nominal_columns: {self.nominal_columns}, X.columns: {list(X.columns)}"
            )
        
        # ===== Handle Ordinal Columns =====
        if ordinal_cols:
            self.ordinal_imputer = self._create_strategy(
                self.ordinal_strategy,
                ordinal_cols,
                self.ordinal_params,
                feature_type='ordinal',
                X_sample=X[ordinal_cols]
            )
            self.ordinal_imputer.fit(X[ordinal_cols])
        
        # ===== Handle Nominal Columns =====
        if nominal_cols:
            self.nominal_imputer = self._create_strategy(
                self.nominal_strategy,
                nominal_cols,
                self.nominal_params,
                feature_type='nominal',
                X_sample=X[nominal_cols]
            )
            self.nominal_imputer.fit(X[nominal_cols])
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted imputation strategies to dataset.
        
        Applies each strategy to its designated columns, then combines results.
        
        Args:
            X: Features DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values filled by appropriate strategies
            
        Raises:
            ValueError: If strategy not fitted yet
        """
        if not self.fitted:
            raise ValueError("TypeBasedImputation must be fit before transform")
        
        if not self.feature_names:
            raise ValueError("Feature names not set. Call fit() first.")
        
        # Filter to columns that actually exist in X
        ordinal_cols = [c for c in self.ordinal_columns if c in X.columns]
        nominal_cols = [c for c in self.nominal_columns if c in X.columns]
        
        # Start with a copy to preserve all columns including those not imputed
        result = X.copy()
        imputed_parts = {}
        
        # Apply ordinal imputation if imputer exists and columns are present
        if self.ordinal_imputer is not None and ordinal_cols:
            X_ordinal = self.ordinal_imputer.transform(X[ordinal_cols])
            imputed_parts.update({col: X_ordinal[col] for col in ordinal_cols})
        
        # Apply nominal imputation if imputer exists and columns are present
        if self.nominal_imputer is not None and nominal_cols:
            X_nominal = self.nominal_imputer.transform(X[nominal_cols])
            imputed_parts.update({col: X_nominal[col] for col in nominal_cols})
        
        # Update the result dataframe with imputed columns
        for col, values in imputed_parts.items():
            result[col] = values
        
        # Ensure column order matches original X
        return result[list(X.columns)]
    
    def _create_strategy(
        self,
        strategy_name: str,
        columns: List[str],
        params: Dict[str, Any],
        feature_type: str = 'unknown',
        X_sample: Optional[pd.DataFrame] = None,
    ) -> ImputationStrategy:
        """
        Create appropriate imputation strategy instance.
        
        Selects the best imputation strategy based on:
        - strategy_name: requested strategy (mean, knn, mode, mice)
        - feature_type: ordinal, nominal, or unknown
        - X_sample: sample data to check if columns are numeric or categorical
        
        Special handling:
        - KNN + ordinal string features → OrdinalStringKNNImputation
        - KNN + nominal categorical features → CategoricalKNNImputation
        - KNN + numeric features → standard KNNImputation
        - mode + any features → ModeImputation
        
        Args:
            strategy_name: Name of strategy (mean, median, knn, mode, mice)
            columns: List of columns for this strategy
            params: Parameters dict for the strategy
            feature_type: Type hint (ordinal, nominal, unknown)
            X_sample: Sample DataFrame to check feature types
            
        Returns:
            Initialized ImputationStrategy instance
            
        Raises:
            ValueError: If strategy name not recognized
        """
        strategy_lower = strategy_name.lower()
        
        if strategy_lower == "mean":
            return MeanImputation(feature_names=columns)
        
        elif strategy_lower == "median":
            # Median imputation - use mean for now (TODO: implement)
            return MeanImputation(feature_names=columns)
        
        elif strategy_lower == "knn":
            n_neighbors = params.get("n_neighbors", 5)
            
            # Check feature types to decide which KNN variant to use
            if X_sample is not None and len(columns) > 0:
                # Check if any columns are non-numeric (object dtype)
                has_non_numeric = any(X_sample[col].dtype == 'object' for col in columns if col in X_sample.columns)
                
                if has_non_numeric:
                    # For ordinal features: use OrdinalStringKNNImputation
                    if feature_type == 'ordinal':
                        return OrdinalStringKNNImputation(
                            feature_names=columns,
                            n_neighbors=n_neighbors
                        )
                    # For nominal features: use CategoricalKNNImputation
                    elif feature_type == 'nominal':
                        return CategoricalKNNImputation(
                            feature_names=columns,
                            n_neighbors=n_neighbors
                        )
            
            # Otherwise use standard numeric KNN
            return KNNImputation(feature_names=columns, n_neighbors=n_neighbors)
        
        elif strategy_lower == "mode":
            return ModeImputation(feature_names=columns)
        
        elif strategy_lower == "mice":
            max_iter = params.get("max_iter", 10)
            
            # Check if columns are numeric or string - MICE only works on numeric
            if X_sample is not None and len(columns) > 0:
                has_non_numeric = any(X_sample[col].dtype == 'object' for col in columns if col in X_sample.columns)
                
                if has_non_numeric:
                    # For ordinal string features, fall back to KNN-based approach
                    if feature_type == 'ordinal':
                        n_neighbors = params.get("n_neighbors", 5)
                        return OrdinalStringKNNImputation(
                            feature_names=columns,
                            n_neighbors=n_neighbors
                        )
                    # For nominal string features, use mode
                    elif feature_type == 'nominal':
                        return ModeImputation(feature_names=columns)
            
            # For numeric columns, use standard MICE
            return MICEImputation(feature_names=columns, max_iter=max_iter)
        
        else:
            raise ValueError(
                f"Unknown imputation strategy: {strategy_name}. "
                f"Must be one of: mean, median, knn, mode, mice"
            )
