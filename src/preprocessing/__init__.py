"""
Preprocessing module for data transformation.

This module provides interfaces and implementations for missing value imputation
and feature encoding strategies, as well as the PreprocessingPipeline orchestrator
that combines them.

Classes:
    ImputationStrategy: Abstract base class for imputation implementations
    DropRowsImputation: Remove rows with any missing values
    DropColumnsImputation: Remove columns with excessive missing values
    MeanImputation: Fill with column means
    ModeImputation: Fill with column modes
    KNNImputation: k-nearest neighbors imputation
    MICEImputation: Multivariate imputation by chained equations
    FlagAsMissingImputation: Impute and create missing indicators
    FeatureEncoder: Abstract base class for encoding implementations
    OrdinalEncoder: Preserve order for ordinal features
    OneHotEncoder: One-hot encoding for categorical features
    TargetEncoder: Target encoding using target variable
    InteractionEncoder: Create interaction terms
    PolynomialEncoder: Create polynomial features
    PreprocessingPipeline: Orchestrator combining imputation and encoding

See docs/SYSTEM_DESIGN.md for detailed component architecture.
"""

from typing import Optional, List, Dict, Any
import pandas as pd

from src.preprocessing.imputation import (
    ImputationStrategy,
    DropRowsImputation,
    DropColumnsImputation,
    MeanImputation,
    ModeImputation,
    KNNImputation,
    MICEImputation,
    FlagAsMissingImputation,
)
from src.preprocessing.encoding import (
    FeatureEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    TargetEncoder,
    InteractionEncoder,
    PolynomialEncoder,
    FEATURE_GROUPS,
)


class PreprocessingPipeline:
    """
    Orchestrator that combines imputation and encoding into a single pipeline.
    
    The PreprocessingPipeline manages the sequential application of imputation
    and encoding strategies to training and test data, ensuring:
    - Imputation statistics are learned ONLY on the training fold (no data leakage)
    - Encoding parameters are learned ONLY on the training fold (after imputation)
    - All datasets (validation, test) use the same fitted transformations
    - Missing values are handled before encoding
    - Output is a fully encoded DataFrame ready for model training
    
    Workflow:
    1. fit(X_train): Fit imputation strategy on training data, then fit encoding
                     on imputed training data
    2. transform(X): Apply fitted imputation, then fitted encoding to any dataset
    3. fit_transform(X_train): Convenience method combining fit + transform
    
    Attributes:
        imputation_config: ImputationConfig specifying imputation strategy
        encoding_config: EncodingConfig specifying encoding strategies
        imputer: Fitted ImputationStrategy instance
        ordinal_encoder: Fitted OrdinalEncoder instance
        onehot_encoder: Fitted OneHotEncoder instance
        fitted: Boolean indicating whether pipeline has been fit
        
    Example:
        >>> from src.config import ImputationConfig, EncodingConfig
        >>> from src.preprocessing import PreprocessingPipeline
        >>> 
        >>> impute_cfg = ImputationConfig(strategy='mean')
        >>> encode_cfg = EncodingConfig(
        ...     strategies={'opinion': {'type': 'ordinal'}, 'demographic': {'type': 'onehot'}}
        ... )
        >>> pipeline = PreprocessingPipeline(impute_cfg, encode_cfg)
        >>> 
        >>> # Fit on training fold only
        >>> X_train_processed = pipeline.fit_transform(X_train)
        >>> 
        >>> # Apply to validation and test (using training statistics)
        >>> X_val_processed = pipeline.transform(X_val)
        >>> X_test_processed = pipeline.transform(X_test)
        >>> 
        >>> # Get output feature names
        >>> feature_names = pipeline.get_feature_names()
    """
    
    def __init__(self, imputation_config: Any, encoding_config: Any):
        """
        Initialize preprocessing pipeline with configuration objects.
        
        Args:
            imputation_config: ImputationConfig object specifying imputation strategy
            encoding_config: EncodingConfig object specifying encoding strategies
            
        Raises:
            TypeError: If config objects are not valid configuration objects
        """
        self.imputation_config = imputation_config
        self.encoding_config = encoding_config
        
        self.imputer = None
        self.ordinal_encoder = None
        self.onehot_encoder = None
        self.fitted = False
    
    def fit(self, X: pd.DataFrame) -> "PreprocessingPipeline":
        """
        Fit imputation and encoding strategies on training data.
        
        Fitting is performed in two stages:
        1. Fit imputation strategy on raw training data
        2. Fit encoding strategies on imputed training data
        
        This ensures:
        - Imputation statistics (mean, mode, KNN neighbors) are learned from training
        - Encoding statistics (category lists, ordinal mappings) are learned from
          imputed training data, not raw data with missing values
        - No data from validation or test sets influences the learned statistics
        
        Args:
            X: Training features DataFrame with potential missing values
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If X is empty or invalid
        """
        if X.empty:
            raise ValueError("Cannot fit PreprocessingPipeline on empty DataFrame")
        
        # Step 1: Fit imputation strategy on raw training data
        self.imputer = self._create_imputer()
        self.imputer.fit(X)
        
        # Step 2: Apply imputation to training data
        X_imputed = self.imputer.transform(X)
        
        # Step 3: Fit encoding strategies on imputed training data
        self._fit_encoders(X_imputed)
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted imputation and encoding to dataset.
        
        Applies the transformations learned during fit() to new data:
        1. Apply fitted imputation (fills missing values using training statistics)
        2. Apply fitted ordinal encoding (preserves order for ordinal features)
        3. Apply fitted one-hot encoding (creates binary indicators for categorical)
        
        This method uses ONLY the statistics learned during fit() and does NOT
        learn any new parameters from the input data.
        
        Args:
            X: Features DataFrame with potential missing values
            
        Returns:
            Fully encoded DataFrame ready for model training
            - All missing values filled
            - Ordinal features normalized
            - Categorical features one-hot encoded
            - Shape: (n_samples, n_encoded_features)
            
        Raises:
            ValueError: If pipeline has not been fitted yet
            ValueError: If X has different columns than training data used for fit
        """
        if not self.fitted:
            raise ValueError("PreprocessingPipeline has not been fitted. Call fit() first.")
        
        if self.imputer is None or self.ordinal_encoder is None or self.onehot_encoder is None:
            raise ValueError("Pipeline components not properly initialized. Call fit() first.")
        
        # Step 1: Apply imputation
        X_imputed = self.imputer.transform(X)
        
        # Step 2: Apply ordinal encoding
        X_ordinal = self.ordinal_encoder.transform(X_imputed)
        
        # Step 3: Apply one-hot encoding
        X_encoded = self.onehot_encoder.transform(X_ordinal)
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessing pipeline and transform data in one step.
        
        Convenience method equivalent to fit(X).transform(X). Useful when
        fitting and transforming the same dataset (though in practice,
        preprocessing is fit on training folds and applied to separate folds).
        
        Args:
            X: Training features DataFrame with potential missing values
            
        Returns:
            Fully encoded DataFrame
        """
        return self.fit(X).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of output feature names after preprocessing.
        
        Returns the column names of the encoded DataFrame that would result
        from transform(). Includes:
        - Ordinal features (unchanged names from ordinal encoding)
        - One-hot encoded features (feature_name_category format)
        
        Returns:
            List of output feature names
            
        Raises:
            ValueError: If pipeline has not been fitted yet
        """
        if not self.fitted:
            raise ValueError("PreprocessingPipeline has not been fitted. Call fit() first.")
        
        # Ordinal encoder returns same feature names for ordinal features
        ordinal_names = self.ordinal_encoder.get_feature_names()
        
        # One-hot encoder returns expanded feature names (includes ordinal features unchanged)
        output_names = self.onehot_encoder.get_feature_names()
        
        return output_names
    
    def _create_imputer(self) -> ImputationStrategy:
        """
        Create imputation strategy instance based on config.
        
        Creates the appropriate imputer class based on imputation_config.strategy.
        Passes strategy-specific parameters from config.
        
        Args:
            None (uses self.imputation_config)
            
        Returns:
            Initialized ImputationStrategy instance
            
        Raises:
            ValueError: If strategy name not recognized
        """
        strategy = self.imputation_config.strategy.lower()
        
        if strategy == "drop_rows":
            return DropRowsImputation()
        
        elif strategy == "drop_columns":
            drop_threshold = getattr(
                self.imputation_config, 'drop_threshold', 0.5
            )
            return DropColumnsImputation(drop_threshold=drop_threshold)
        
        elif strategy == "mean":
            return MeanImputation()
        
        elif strategy == "mode":
            return ModeImputation()
        
        elif strategy == "knn":
            n_neighbors = getattr(
                self.imputation_config, 'n_neighbors', 5
            )
            return KNNImputation(n_neighbors=n_neighbors)
        
        elif strategy == "mice":
            mice_iterations = getattr(
                self.imputation_config, 'mice_iterations', 10
            )
            return MICEImputation(n_imputations=mice_iterations)
        
        elif strategy == "flag_as_missing":
            return FlagAsMissingImputation()
        
        else:
            raise ValueError(
                f"Unknown imputation strategy: {strategy}. "
                f"Must be one of: drop_rows, drop_columns, mean, mode, knn, mice, flag_as_missing"
            )
    
    def _fit_encoders(self, X: pd.DataFrame) -> None:
        """
        Fit ordinal and one-hot encoders on imputed training data.
        
        Creates encoder instances based on encoding_config and fits them on
        the imputed training data. Handles mapping of feature groups to
        encoding strategies specified in config.
        
        The encoding strategy depends on feature types:
        - Ordinal features (opinions, concern, knowledge): OrdinalEncoder
        - Categorical features (demographics, employment): OneHotEncoder
        - Binary features: Passed through unchanged (by OneHotEncoder)
        
        Args:
            X: Imputed training features DataFrame
            
        Raises:
            ValueError: If encoding_config is invalid
        """
        all_features = list(X.columns)
        
        # Determine which features should be ordinal-encoded
        ordinal_features = getattr(
            self.encoding_config, 'ordinal_features', []
        )
        if not ordinal_features:
            # Fallback: use feature groups with 'ordinal' strategy
            ordinal_features = self._get_features_by_strategy('ordinal')
        
        # Determine which features should be one-hot encoded
        categorical_features = getattr(
            self.encoding_config, 'categorical_features', []
        )
        if not categorical_features:
            # Fallback: use feature groups with 'onehot' strategy
            categorical_features = self._get_features_by_strategy('onehot')
        
        # Create and fit ordinal encoder
        self.ordinal_encoder = OrdinalEncoder(
            input_features=all_features,
            ordinal_features=ordinal_features,
            keep_as_is=True
        )
        self.ordinal_encoder.fit(X)
        
        # Apply ordinal encoding to get intermediate dataset
        X_ordinal = self.ordinal_encoder.transform(X)
        
        # Create and fit one-hot encoder
        drop_first = getattr(
            self.encoding_config, 'drop_first_onehot', True
        )
        self.onehot_encoder = OneHotEncoder(
            input_features=list(X_ordinal.columns),
            categorical_features=categorical_features,
            drop_first=drop_first,
            handle_unknown="ignore"
        )
        self.onehot_encoder.fit(X_ordinal)
    
    def _get_features_by_strategy(self, strategy_type: str) -> List[str]:
        """
        Extract features for a given encoding strategy from feature groups.
        
        Helper method to map feature group configurations to actual feature lists.
        Looks up which features belong to groups that should use the specified
        encoding strategy (e.g., 'ordinal' or 'onehot').
        
        Args:
            strategy_type: Type of strategy ('ordinal' or 'onehot')
            
        Returns:
            List of feature names matching the strategy type
        """
        features = []
        strategies = getattr(
            self.encoding_config, 'strategies', {}
        )
        
        for group_name, group_strategy in strategies.items():
            strategy = group_strategy.get('type', 'ordinal') if isinstance(group_strategy, dict) else group_strategy
            if strategy == strategy_type:
                if group_name in FEATURE_GROUPS:
                    features.extend(FEATURE_GROUPS[group_name])
        
        return features


__all__ = [
    "ImputationStrategy",
    "DropRowsImputation",
    "DropColumnsImputation",
    "MeanImputation",
    "ModeImputation",
    "KNNImputation",
    "MICEImputation",
    "FlagAsMissingImputation",
    "FeatureEncoder",
    "OrdinalEncoder",
    "OneHotEncoder",
    "TargetEncoder",
    "InteractionEncoder",
    "PolynomialEncoder",
    "FEATURE_GROUPS",
    "PreprocessingPipeline",
]
