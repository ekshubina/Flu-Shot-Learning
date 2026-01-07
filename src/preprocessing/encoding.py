"""
Feature encoding strategies for categorical, ordinal, and numerical features.

This module defines the abstract interface for feature encoders and provides
concrete implementations for various encoding approaches suitable for the
H1N1 flu shot prediction dataset.

Feature encoding is applied after imputation but before model training.
Different strategies are suitable for different feature types:
- Ordinal features (opinions, concerns): Preserve order using ordinal encoding
- Categorical features (demographics, geography): One-hot or target encoding
- Binary features: Already 0/1, may be included as-is or with transformations

See PROBLEM_DESCRIPTION.md and CONTEXT_REPORT.md for feature definitions and
analysis of feature cardinality.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np


# Feature groups for organizing encoding choices
FEATURE_GROUPS = {
    "opinions": [
        "opinion_h1n1_vacc_effective",
        "opinion_h1n1_risk",
        "opinion_h1n1_sick_from_vacc",
        "opinion_seas_vacc_effective",
        "opinion_seas_risk",
        "opinion_seas_sick_from_vacc",
    ],
    "behavioral": [
        "behavioral_antiviral_meds",
        "behavioral_avoidance",
        "behavioral_face_mask",
        "behavioral_large_gatherings",
        "behavioral_outside_home",
        "behavioral_touch_face",
    ],
    "medical": [
        "doctor_recc_h1n1",
        "doctor_recc_seasonal",
        "chronic_med_condition",
        "health_worker",
        "health_insurance",
    ],
    "demographics": [
        "age_group",
        "education",
        "race",
        "sex",
        "income_poverty",
        "marital_status",
        "rent_or_own",
        "employment_status",
    ],
    "household": [
        "household_adults",
        "household_children",
    ],
    "geographic": [
        "hhs_geo_region",
        "census_msa",
    ],
    "concern_knowledge": [
        "h1n1_concern",
        "h1n1_knowledge",
    ],
    "employment": [
        "employment_industry",
        "employment_occupation",
    ],
}


class FeatureEncoder(ABC):
    """
    Abstract base class for feature encoding strategies.
    
    Feature encoders transform raw features into representations suitable for
    machine learning models. Different encoders preserve different aspects of
    the original data:
    - Ordinal encoders preserve ordering in ordinal features
    - One-hot encoders create binary indicators for categorical values
    - Target encoders use the target variable for categorical encoding
    - Interaction encoders create cross-product terms
    - Polynomial encoders create polynomial features
    
    Attributes:
        input_features: List of input feature column names
        output_features: List of output feature column names (set after fit)
        fitted: Whether encoder has been fit to training data
        fit_params: Dictionary of parameters learned from training data
        
    Example:
        >>> from src.preprocessing.encoding import OneHotEncoder
        >>> encoder = OneHotEncoder(
        ...     categorical_features=['age_group', 'education'],
        ...     drop_first=True
        ... )
        >>> X_train_encoded = encoder.fit_transform(X_train)
        >>> X_test_encoded = encoder.transform(X_test)
        >>> feature_names = encoder.get_feature_names()
    """

    def __init__(self, input_features: Optional[List[str]] = None):
        """
        Initialize feature encoder.
        
        Args:
            input_features: List of input feature column names (optional)
        """
        self.input_features = input_features
        self.output_features = None
        self.fitted = False
        self.fit_params = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEncoder":
        """
        Fit encoder on training data.
        
        Learns any parameters needed for transformation (e.g., category mappings,
        polynomial terms, interaction pairs). Some encoders (like TargetEncoder)
        may need target variable y.
        
        Args:
            X: Training features DataFrame
            y: Target variable (optional, required for some encoders like TargetEncoder)
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If X is invalid or encoder-specific requirements not met
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset using learned encoding parameters.
        
        Uses parameters from fit() to encode new data. Must call fit() first
        on training data.
        
        Args:
            X: Features DataFrame with input columns
            
        Returns:
            DataFrame with encoded features (column count may change)
            
        Raises:
            ValueError: If encoder not fitted yet
            ValueError: If X has different input columns than training data
        """
        pass

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Fit encoder and transform data in one step.
        
        Convenience method equivalent to fit(X, y).transform(X).
        
        Args:
            X: Training features DataFrame
            y: Target variable (optional)
            
        Returns:
            DataFrame with encoded features
        """
        return self.fit(X, y).transform(X)

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of output feature column names after encoding.
        
        Returns:
            List of output feature names generated by encoding
            
        Raises:
            ValueError: If encoder not fitted yet
        """
        pass

    def get_feature_names_in(self) -> List[str]:
        """
        Get list of input feature column names.
        
        Returns:
            List of input feature names
            
        Raises:
            ValueError: If input_features not set
        """
        if self.input_features is None:
            raise ValueError("Input features not set. Pass input_features to constructor.")
        return self.input_features


class OrdinalEncoder(FeatureEncoder):
    """
    Encoder for ordinal features that preserves order relationships.
    
    For features like opinion scales (1-5) or knowledge/concern levels (1-3),
    preserves the ordinal relationships by keeping numerical values or mapping
    to integers. Suitable for tree-based models that can use the ordering.
    
    For opinion features: Assumes values 1-5 represent increasing agreement/risk
    For concern/knowledge: Assumes values 0-2 or 1-3 represent increasing levels
    
    Configuration:
        - ordinal_features: List of feature names to encode as ordinal
        - keep_as_is: If True, keep original values; if False, map to 0, 1, 2, ...
    
    Trade-offs:
    - Pro: Preserves ordering, no feature explosion, simple
    - Con: May not work well for linear models that don't use ordering
    
    Implementation notes:
        - TODO: In fit(), analyze value distributions for each ordinal feature
        - TODO: Map values to integers preserving order (or keep original if numeric)
        - TODO: Store value mappings in fit_params
        - TODO: In transform(), apply stored mappings
    """

    def __init__(
        self,
        input_features: Optional[List[str]] = None,
        ordinal_features: Optional[List[str]] = None,
        keep_as_is: bool = True,
    ):
        """
        Initialize ordinal encoder.
        
        Args:
            input_features: List of all input feature names
            ordinal_features: List of features to encode as ordinal
            keep_as_is: Keep original ordinal values (True) or remap to 0,1,2... (False)
        """
        super().__init__(input_features)
        self.ordinal_features = ordinal_features or []
        self.keep_as_is = keep_as_is

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OrdinalEncoder":
        """
        Analyze ordinal features and prepare for transformation.
        
        Implementation notes:
            - TODO: For each ordinal feature, analyze unique values and order
            - TODO: Store value-to-integer mapping in fit_params
            - TODO: Handle unseen values in transform
        """
        # TODO: Implement
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ordinal encoding to features.
        
        Implementation notes:
            - TODO: For each ordinal feature, apply learned mapping
            - TODO: Keep non-ordinal features unchanged
            - TODO: Return DataFrame with same columns
        """
        # TODO: Implement
        pass

    def get_feature_names(self) -> List[str]:
        """
        Get output feature names (unchanged for ordinal encoding).
        
        Returns:
            List of feature names (same as input)
        """
        # TODO: Implement
        pass


class OneHotEncoder(FeatureEncoder):
    """
    Encoder for categorical features using one-hot encoding.
    
    Creates binary indicator columns for each category. Suitable for categorical
    features like age_group, education, race, employment_status, etc.
    
    Configuration:
        - categorical_features: List of feature names to encode
        - drop_first: Drop first category to avoid multicollinearity (default: True)
        - handle_unknown: How to handle unseen categories ('error' or 'ignore')
        - sparse_output: Return sparse matrix (True) or dense DataFrame (False)
    
    Trade-offs:
    - Pro: Standard approach, works well with linear models, interpretable
    - Con: Feature explosion for high-cardinality features, not ordinal-aware
    
    Output columns: One per unique category per feature (e.g., age_group_0-18, age_group_19-34, ...)
    
    Implementation notes:
        - TODO: In fit(), identify unique categories for each categorical feature
        - TODO: Store category lists in fit_params
        - TODO: Optionally drop first category (handle_unknown='drop' equivalent)
        - TODO: In transform(), create binary columns for each category
        - TODO: Handle unseen categories in transform (fill with 0s if drop_first=True)
    """

    def __init__(
        self,
        input_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        drop_first: bool = True,
        handle_unknown: str = "error",
        sparse_output: bool = False,
    ):
        """
        Initialize one-hot encoder.
        
        Args:
            input_features: List of all input feature names
            categorical_features: List of features to one-hot encode
            drop_first: Drop first category to avoid multicollinearity (default: True)
            handle_unknown: 'error' or 'ignore' for unseen categories
            sparse_output: Return sparse matrix (default: False for DataFrames)
        """
        super().__init__(input_features)
        self.categorical_features = categorical_features or []
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OneHotEncoder":
        """
        Identify unique categories for each categorical feature.
        
        Implementation notes:
            - TODO: For each categorical feature, find unique categories
            - TODO: Sort categories for consistency
            - TODO: Store category lists in fit_params['categories']
            - TODO: Compute output feature names
        """
        # TODO: Implement
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create one-hot encoded features.
        
        Implementation notes:
            - TODO: For each categorical feature:
            -   TODO: Create binary column for each category
            -   TODO: Handle unseen categories (0 if drop_first, error otherwise)
            - TODO: Keep non-categorical features unchanged
            - TODO: Return DataFrame with one-hot features
        """
        # TODO: Implement
        pass

    def get_feature_names(self) -> List[str]:
        """
        Get output feature names from one-hot encoding.
        
        Returns:
            List of feature names including new one-hot columns
        """
        # TODO: Implement
        pass


class TargetEncoder(FeatureEncoder):
    """
    Encoder for categorical features using target mean encoding.
    
    Encodes categorical features by their mean target value, adding information
    from the target variable. More sophisticated than one-hot, useful for
    high-cardinality features and can improve model performance.
    
    For multilabel problem: Encodes each target separately or as a composite.
    
    Configuration:
        - categorical_features: List of feature names to encode
        - smoothing: Regularization parameter (default: 1.0)
        - handle_unknown: How to handle unseen categories ('return_nan', 'value')
        - unknown_value: Value to use for unseen categories (default: 0)
    
    Trade-offs:
    - Pro: Reduces feature explosion, incorporates target information, interpretable
    - Con: Can cause overfitting if not regularized, requires target variable
    
    Output columns: Same input features but with numerical target means
    
    Implementation notes:
        - TODO: In fit(), compute mean target value per category per feature
        - TODO: Apply smoothing: (count * mean + smoothing * global_mean) / (count + smoothing)
        - TODO: Store category means in fit_params
        - TODO: Handle multilabel targets (compute for each vaccine separately)
        - TODO: In transform(), replace categories with learned means
        - TODO: Handle unseen categories using smoothing parameter
    """

    def __init__(
        self,
        input_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        smoothing: float = 1.0,
        handle_unknown: str = "value",
        unknown_value: float = 0.0,
    ):
        """
        Initialize target encoder.
        
        Args:
            input_features: List of all input feature names
            categorical_features: List of features to target encode
            smoothing: Regularization parameter (default: 1.0)
            handle_unknown: 'value' to use unknown_value, 'return_nan' to return NaN
            unknown_value: Value to use for unseen categories (default: 0.0)
        """
        super().__init__(input_features)
        self.categorical_features = categorical_features or []
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TargetEncoder":
        """
        Compute target mean for each category.
        
        Implementation notes:
            - TODO: Require y (target) to be provided
            - TODO: For each categorical feature:
            -   TODO: Compute mean target value per category
            -   TODO: Apply smoothing formula
            -   TODO: Store mappings in fit_params
        """
        # TODO: Implement
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace categories with learned target means.
        
        Implementation notes:
            - TODO: For each categorical feature:
            -   TODO: Map categories to learned means
            -   TODO: Handle unseen categories using smoothing
            - TODO: Keep non-categorical features unchanged
            - TODO: Return DataFrame with target-encoded features
        """
        # TODO: Implement
        pass

    def get_feature_names(self) -> List[str]:
        """
        Get output feature names (unchanged for target encoding).
        
        Returns:
            List of feature names (same as input)
        """
        # TODO: Implement
        pass


class InteractionEncoder(FeatureEncoder):
    """
    Encoder that creates interaction terms between features.
    
    Generates cross-product terms between specified pairs of features,
    useful for capturing interactions (e.g., age × opinion_risk).
    
    Configuration:
        - pairs: List of (feature1, feature2) tuples to create interactions
        - include_original: Keep original features in output (default: True)
        - interaction_type: 'product' (default) or 'sum'
    
    Trade-offs:
    - Pro: Can capture feature interactions, interpretable
    - Con: Feature explosion with many interactions, prone to overfitting
    
    Output: Original features + interaction columns
    Example columns: age_group, opinion_risk, age_group × opinion_risk
    
    Implementation notes:
        - TODO: In fit(), store interaction pair specifications
        - TODO: In transform():
        -   TODO: For each specified pair, create interaction column
        -   TODO: Column name: feature1_x_feature2 (or sum if interaction_type='sum')
        - TODO: Keep original features if include_original=True
    """

    def __init__(
        self,
        input_features: Optional[List[str]] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
        include_original: bool = True,
        interaction_type: str = "product",
    ):
        """
        Initialize interaction encoder.
        
        Args:
            input_features: List of all input feature names
            pairs: List of (feature1, feature2) tuples for interactions
            include_original: Keep original features in output
            interaction_type: 'product' or 'sum' for interaction computation
        """
        super().__init__(input_features)
        self.pairs = pairs or []
        self.include_original = include_original
        self.interaction_type = interaction_type

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "InteractionEncoder":
        """
        Prepare for creating interactions.
        
        Implementation notes:
            - TODO: Validate that all features in pairs exist in X
            - TODO: Store pair list in fit_params
        """
        # TODO: Implement
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features.
        
        Implementation notes:
            - TODO: For each pair, create interaction column
            - TODO: Use product (default) or sum based on interaction_type
            - TODO: Column naming: feature1_x_feature2
            - TODO: Include original features if include_original=True
            - TODO: Return DataFrame with original + interaction columns
        """
        # TODO: Implement
        pass

    def get_feature_names(self) -> List[str]:
        """
        Get output feature names including interaction columns.
        
        Returns:
            List of original and interaction feature names
        """
        # TODO: Implement
        pass


class PolynomialEncoder(FeatureEncoder):
    """
    Encoder that creates polynomial features.
    
    Generates polynomial terms (e.g., x, x², x³, ...) and interaction terms
    up to a specified degree. Useful for capturing non-linear relationships
    in numerical features.
    
    Configuration:
        - degree: Maximum polynomial degree (default: 2)
        - include_bias: Include bias/constant term (default: False)
        - include_original: Keep original features in output (default: True)
    
    Trade-offs:
    - Pro: Captures non-linear patterns, standard approach for linear models
    - Con: Feature explosion (with degree d, n features → ~n^d features),
            prone to overfitting, requires feature scaling
    
    Output: Original features + polynomial/interaction terms
    Example (degree=2, features x,y): x, y, x², xy, y²
    
    Implementation notes:
        - TODO: In fit(), determine which features to polynomialize
        - TODO: Store degree and feature list in fit_params
        - TODO: In transform():
        -   TODO: Generate all polynomial terms up to degree
        -   TODO: Include original features if include_original=True
        -   TODO: Create appropriate column names (e.g., x2, x_y)
    """

    def __init__(
        self,
        input_features: Optional[List[str]] = None,
        degree: int = 2,
        include_bias: bool = False,
        include_original: bool = True,
    ):
        """
        Initialize polynomial encoder.
        
        Args:
            input_features: List of all input feature names
            degree: Maximum polynomial degree (default: 2)
            include_bias: Include constant bias term (default: False)
            include_original: Keep original features in output (default: True)
        """
        super().__init__(input_features)
        self.degree = degree
        self.include_bias = include_bias
        self.include_original = include_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PolynomialEncoder":
        """
        Prepare for creating polynomial features.
        
        Implementation notes:
            - TODO: Store feature list and degree in fit_params
            - TODO: Compute output feature names
        """
        # TODO: Implement
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Implementation notes:
            - TODO: Generate polynomial terms up to specified degree
            - TODO: Column naming: x2 for x², x_y for x*y, etc.
            - TODO: Include original features if include_original=True
            - TODO: Include bias term if include_bias=True
            - TODO: Return DataFrame with polynomial features
        """
        # TODO: Implement
        pass

    def get_feature_names(self) -> List[str]:
        """
        Get output feature names including polynomial terms.
        
        Returns:
            List of original and polynomial feature names
        """
        # TODO: Implement
        pass
