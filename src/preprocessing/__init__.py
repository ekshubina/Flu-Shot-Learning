"""
Preprocessing module for data transformation.

This module provides interfaces and implementations for missing value imputation
and feature encoding strategies.

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

See docs/SYSTEM_DESIGN.md for detailed component architecture.
"""

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
]
