"""
General utility helper functions for the ML pipeline.

Provides common utilities like:
- Train-test splitting with stratification
- Class weight computation
- Feature group definitions
- Data transformation helpers

Reference: SYSTEM_DESIGN.md - Component 9: Utilities
"""

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from pathlib import Path


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
    # TODO: Implement
    raise NotImplementedError("stratified_train_test_split() not yet implemented")


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
    # TODO: Implement
    raise NotImplementedError("stratified_k_fold_split() not yet implemented")


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
    # TODO: Implement
    raise NotImplementedError("compute_class_weights() not yet implemented")


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
    # TODO: Implement
    raise NotImplementedError("get_feature_group() not yet implemented")


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
    # TODO: Implement
    raise NotImplementedError("create_feature_group_mask() not yet implemented")


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
    # TODO: Implement
    raise NotImplementedError("ensure_parent_directory() not yet implemented")


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
    # TODO: Implement
    raise NotImplementedError("load_data_safely() not yet implemented")


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
    # TODO: Implement
    raise NotImplementedError("seed_all_random_states() not yet implemented")
