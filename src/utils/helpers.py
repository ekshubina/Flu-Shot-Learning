"""
General utility helper functions for the ML pipeline.

Provides common utilities like:
- Train-test splitting with stratification
- Class weight computation
- Feature group definitions
- Data transformation helpers

Reference: SYSTEM_DESIGN.md - Component 9: Utilities
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
import random


# Feature group definitions based on problem analysis
# Reference: PROBLEM_DESCRIPTION.md and CONTEXT_REPORT.md

OPINION_FEATURES = [
    'opinion_h1n1_vacc_effective',
    'opinion_h1n1_risk',
    'opinion_h1n1_sick_from_vacc',
    'opinion_seas_vacc_effective',
    'opinion_seas_risk',
]

BEHAVIORAL_FEATURES = [
    'behavioral_antiviral_meds',
    'behavioral_avoidance',
    'behavioral_face_mask',
    'behavioral_hand_washing',
    'behavioral_large_gatherings',
    'behavioral_outside_home',
    'behavioral_touch_face',
]

MEDICAL_FEATURES = [
    'doctor_recc_h1n1',
    'doctor_recc_seasonal',
    'chronic_med_condition',
    'health_worker',
    'health_insurance',
]

DEMOGRAPHIC_FEATURES = [
    'age_group',
    'education',
    'race',
    'sex',
    'income_poverty',
    'marital_status',
    'rent_or_own',
    'employment_status',
]

HOUSEHOLD_FEATURES = [
    'household_adults',
    'household_children',
]

GEOGRAPHIC_FEATURES = [
    'hhs_geo_region',
    'census_msa',
]

CONCERN_FEATURES = [
    'h1n1_concern',
    'h1n1_knowledge',
]

EMPLOYMENT_FEATURES = [
    'employment_industry',
    'employment_occupation',
]

# Complete feature groups dictionary
FEATURE_GROUPS = {
    'opinion': OPINION_FEATURES,
    'behavioral': BEHAVIORAL_FEATURES,
    'medical': MEDICAL_FEATURES,
    'demographic': DEMOGRAPHIC_FEATURES,
    'household': HOUSEHOLD_FEATURES,
    'geographic': GEOGRAPHIC_FEATURES,
    'concern': CONCERN_FEATURES,
    'employment': EMPLOYMENT_FEATURES,
}

ALL_FEATURES = (
    OPINION_FEATURES
    + BEHAVIORAL_FEATURES
    + MEDICAL_FEATURES
    + DEMOGRAPHIC_FEATURES
    + HOUSEHOLD_FEATURES
    + GEOGRAPHIC_FEATURES
    + CONCERN_FEATURES
    + EMPLOYMENT_FEATURES
)


def stratified_train_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split data into train/test with stratification by target class.
    
    Ensures both splits have similar class distributions.
    
    Parameters:
        X (pd.DataFrame): Features
        y (np.ndarray): Binary labels
        test_size (float): Proportion for test set. Default: 0.2
        random_state (int): Random seed. Default: 42
    
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    
    Implementation notes:
        - TODO: Use sklearn.model_selection.train_test_split()
        - TODO: Set stratify=y for stratification
        - TODO: Return tuple of 4 arrays
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def stratified_k_fold_split(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Generator for stratified k-fold cross-validation splits.
    
    Yields train/test indices for each fold, maintaining class distribution.
    
    Parameters:
        X (pd.DataFrame): Features (only used for determining n_samples)
        y (np.ndarray): Binary labels
        n_splits (int): Number of folds. Default: 5
        random_state (int): Random seed. Default: 42
    
    Yields:
        Tuple[np.ndarray, np.ndarray]: (train_idx, test_idx) for each fold
    
    Implementation notes:
        - TODO: Use sklearn.model_selection.StratifiedKFold()
        - TODO: Iterate over splits
        - TODO: Yield (train_idx, test_idx) tuples
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in skf.split(X, y):
        yield train_idx, test_idx


def compute_class_weights(
    y: np.ndarray,
    strategy: str = 'balanced',
) -> Dict[int, float]:
    """
    Compute class weights for imbalanced classification.
    
    Parameters:
        y (np.ndarray): Binary labels
        strategy (str): Weight strategy:
            'balanced': Inverse proportion (1 / class_frequency)
            'balanced_sqrt': Square root of balanced weights
            Default: 'balanced'
    
    Returns:
        Dict mapping class label to weight:
            {0: weight_for_class_0, 1: weight_for_class_1}
    
    Implementation notes:
        - TODO: Count positive and negative samples
        - TODO: If balanced, compute 1/frequency for each class
        - TODO: If balanced_sqrt, take square root
        - TODO: Return dictionary
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    weights = {}
    for label, count in zip(unique, counts):
        if strategy == 'balanced':
            weights[int(label)] = total / (len(unique) * count)
        elif strategy == 'balanced_sqrt':
            weights[int(label)] = np.sqrt(total / (len(unique) * count))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return weights


def get_feature_group(group_name: str) -> List[str]:
    """
    Get list of features for a feature group.
    
    Parameters:
        group_name (str): Group name from FEATURE_GROUPS
            ('opinion', 'behavioral', 'medical', 'demographic', etc.)
    
    Returns:
        List[str]: Feature names in group
    
    Raises:
        ValueError: If group_name not recognized
    
    Implementation notes:
        - TODO: Look up group_name in FEATURE_GROUPS
        - TODO: Return feature list
        - TODO: Raise ValueError if not found
    """
    if group_name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {group_name}. Valid groups: {list(FEATURE_GROUPS.keys())}")
    return FEATURE_GROUPS[group_name].copy()


def get_all_features() -> List[str]:
    """
    Get list of all 35 features in expected order.
    
    Returns:
        List[str]: All feature names
    """
    return ALL_FEATURES.copy()


def create_feature_group_mask(
    all_features: List[str],
    group_name: str,
) -> np.ndarray:
    """
    Create boolean mask for a feature group.
    
    Useful for selecting subsets of features by group.
    
    Parameters:
        all_features (List[str]): All feature names
        group_name (str): Feature group name
    
    Returns:
        np.ndarray: Boolean mask (True where feature in group)
    
    Implementation notes:
        - TODO: Get features for group_name
        - TODO: Create boolean array of same length as all_features
        - TODO: Set True where feature in group
        - TODO: Return boolean array
    """
    group_features = get_feature_group(group_name)
    return np.array([feature in group_features for feature in all_features])


def ensure_parent_directory(file_path: str) -> None:
    """
    Create parent directories if they don't exist.
    
    Parameters:
        file_path (str): Path to file
    
    Implementation notes:
        - TODO: Convert to Path object
        - TODO: Create parent directories if needed
        - TODO: Use parents.mkdir(parents=True, exist_ok=True)
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)


def load_data_safely(
    file_path: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Load CSV with error handling.
    
    Parameters:
        file_path (str): Path to CSV file
        **kwargs: Additional arguments to pd.read_csv()
    
    Returns:
        pd.DataFrame: Loaded data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid CSV
    
    Implementation notes:
        - TODO: Check file exists
        - TODO: Try to read as CSV
        - TODO: Raise FileNotFoundError if missing
        - TODO: Raise ValueError if read fails
        - TODO: Return DataFrame
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {path}: {e}")


def seed_all_random_states(seed: int = 42) -> None:
    """
    Set random seed for numpy and Python random.
    
    Ensures reproducibility across runs.
    
    Parameters:
        seed (int): Random seed. Default: 42
    
    Implementation notes:
        - TODO: Set numpy.random.seed()
        - TODO: Set random.seed()
        - TODO: No return value
    """
    np.random.seed(seed)
    random.seed(seed)


def create_stratification_column(
    h1n1_labels: pd.Series,
    seasonal_labels: pd.Series,
) -> pd.Series:
    """
    Create combined stratification column for multilabel problem.
    
    For multilabel classification with two targets (h1n1_vaccine, seasonal_vaccine),
    creates a synthetic stratification column that encodes all four label combinations:
    - (0, 0) -> 0: No vaccine
    - (1, 0) -> 1: H1N1 only
    - (0, 1) -> 2: Seasonal only
    - (1, 1) -> 3: Both vaccines
    
    This enables StratifiedKFold to create balanced folds respecting both targets.
    
    Parameters:
        h1n1_labels (pd.Series): H1N1 vaccine binary labels (0 or 1)
        seasonal_labels (pd.Series): Seasonal vaccine binary labels (0 or 1)
    
    Returns:
        pd.Series: Combined stratification column with values {0, 1, 2, 3}
        
    Example:
        >>> h1n1 = pd.Series([0, 1, 0, 1])
        >>> seasonal = pd.Series([0, 0, 1, 1])
        >>> strat = create_stratification_column(h1n1, seasonal)
        >>> strat.tolist()
        [0, 1, 2, 3]
    """
    return h1n1_labels + 2 * seasonal_labels


def validate_respondent_ids(
    respondent_ids: pd.Series,
    raise_on_error: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate respondent ID series for data integrity.
    
    Checks:
    - No missing values
    - All values are unique
    - All values are numeric/integers
    
    Parameters:
        respondent_ids (pd.Series): Series of respondent IDs
        raise_on_error (bool): If True, raise exception on validation failure.
                              If False, return (False, errors). Default: True
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
        - is_valid: True if all checks pass
        - error_messages: List of validation error descriptions
        
    Raises:
        ValueError: If raise_on_error=True and validation fails
        
    Example:
        >>> ids = pd.Series([1, 2, 3, 2])  # Duplicate 2
        >>> is_valid, errors = validate_respondent_ids(ids, raise_on_error=False)
        >>> is_valid
        False
        >>> errors[0]
        'Duplicate respondent IDs found'
    """
    errors = []
    
    # Check for missing values
    if respondent_ids.isna().any():
        errors.append("Missing respondent IDs found")
    
    # Check for duplicates
    if respondent_ids.nunique() != len(respondent_ids):
        errors.append("Duplicate respondent IDs found")
    
    # Check for numeric type
    try:
        pd.to_numeric(respondent_ids)
    except (ValueError, TypeError):
        errors.append("Non-numeric respondent IDs found")
    
    is_valid = len(errors) == 0
    
    if not is_valid and raise_on_error:
        raise ValueError(f"Respondent ID validation failed: {'; '.join(errors)}")
    
    return is_valid, errors


def create_output_directory(
    output_dir: str,
    verbose: bool = True,
) -> Path:
    """
    Create output directory structure.
    
    Creates directory and all parent directories if they don't exist.
    
    Parameters:
        output_dir (str): Path to output directory to create
        verbose (bool): If True, log creation status. Default: True
    
    Returns:
        Path: Path object of created directory
        
    Example:
        >>> output_path = create_output_directory("results/experiment_1/")
        >>> output_path.exists()
        True
    """
    path = Path(output_dir)
    
    # Create directories if they don't exist
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            from src.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.info(f"Created output directory: {path}")
    else:
        if verbose:
            from src.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.debug(f"Output directory already exists: {path}")
    
    return path
