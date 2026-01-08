"""
Prediction engine for test set inference and submission generation.

This module provides interfaces and implementations for:
- Applying preprocessing to test data
- Generating probability predictions on test set
- Formatting predictions as competition submission CSV
- Validating submission format

The competition requires probabilities (0.0-1.0) for both vaccines,
with format: respondent_id, h1n1_vaccine, seasonal_vaccine

Reference: SYSTEM_DESIGN.md - Component 9: Prediction
Reference: PROBLEM_DESCRIPTION.md - Submission format requirements
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


class PredictionEngine(ABC):
    """
    Abstract base class for generating predictions and submissions.
    
    Handles the complete pipeline from test set preprocessing through
    final submission file generation. Validates all outputs match
    competition requirements.
    
    Example:
        ```python
        predictor = PredictionEngine()
        
        # Load test set
        X_test = load_test_features()
        test_ids = X_test['respondent_id']
        
        # Generate predictions
        y_pred_h1n1, y_pred_seasonal = predictor.predict_test_set(
            X_test, trained_model, preprocessing_pipeline
        )
        
        # Format for submission
        submission_df = predictor.format_submission(
            test_ids, y_pred_h1n1, y_pred_seasonal
        )
        
        # Validate format
        predictor.validate_submission(submission_df)
        
        # Save to file
        submission_df.to_csv('submission.csv', index=False)
        ```
    """
    
    @abstractmethod
    def predict_test_set(
        self,
        X_test: pd.DataFrame,
        model: object,
        preprocessing_pipeline: Optional[object] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test set.
        
        Applies preprocessing pipeline (if provided) and trained model to
        generate probability predictions for both vaccines.
        
        Parameters:
            X_test (pd.DataFrame): Test set features (n_samples, 35 features)
                Expected columns: All 35 features from training set
            model (object): Trained model with predict_proba() method
                Must have been trained to output two probabilities:
                one for H1N1 vaccine, one for seasonal vaccine
            preprocessing_pipeline (Optional[object]): Optional preprocessing
                pipeline to apply to X_test before prediction. Must have
                transform() method. If None, X_test used as-is.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                y_pred_h1n1: Predicted probabilities for H1N1 (n_samples,)
                y_pred_seasonal: Predicted probabilities for seasonal (n_samples,)
                Both arrays have shape (n_samples,) with values in [0, 1]
        
        Raises:
            ValueError: If X_test has wrong shape/columns
            TypeError: If model doesn't have predict_proba() method
            
        Implementation notes:
            - TODO: Validate X_test shape and columns
            - TODO: Apply preprocessing if provided
            - TODO: Call model.predict_proba() to get predictions
            - TODO: Extract probabilities for each vaccine
            - TODO: Validate output shape and values [0, 1]
            - TODO: Return two 1D arrays
        """
        pass
    
    @staticmethod
    def format_submission(
        respondent_ids: np.ndarray,
        y_pred_h1n1: np.ndarray,
        y_pred_seasonal: np.ndarray,
    ) -> pd.DataFrame:
        """
        Format predictions as competition submission DataFrame.
        
        Creates a DataFrame with three columns in the required format:
        - respondent_id: The unique ID for each respondent
        - h1n1_vaccine: Predicted probability (0.0-1.0) of H1N1 vaccination
        - seasonal_vaccine: Predicted probability (0.0-1.0) of seasonal vaccination
        
        Parameters:
            respondent_ids (np.ndarray): Respondent IDs (n_samples,)
            y_pred_h1n1 (np.ndarray): Predicted H1N1 probabilities (n_samples,)
            y_pred_seasonal (np.ndarray): Predicted seasonal probabilities (n_samples,)
        
        Returns:
            pd.DataFrame: Submission DataFrame with columns:
                ['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine']
                Shape: (n_samples, 3)
        
        Raises:
            ValueError: If array shapes don't match or lengths are inconsistent
            
        Implementation notes:
            - TODO: Validate that all arrays have same length
            - TODO: Validate y_pred arrays are in [0, 1]
            - TODO: Create DataFrame from arrays
            - TODO: Name columns according to competition spec
            - TODO: Ensure column order is correct
            - TODO: Return DataFrame
        """
        # Validate that all arrays have same length
        if not (len(respondent_ids) == len(y_pred_h1n1) == len(y_pred_seasonal)):
            raise ValueError(
                f"Array lengths don't match: "
                f"respondent_ids={len(respondent_ids)}, "
                f"h1n1={len(y_pred_h1n1)}, "
                f"seasonal={len(y_pred_seasonal)}"
            )
        
        # Validate y_pred arrays are in [0, 1]
        if not (np.all(y_pred_h1n1 >= 0.0) and np.all(y_pred_h1n1 <= 1.0)):
            raise ValueError(
                f"H1N1 predictions not in [0.0, 1.0]: "
                f"min={y_pred_h1n1.min()}, max={y_pred_h1n1.max()}"
            )
        if not (np.all(y_pred_seasonal >= 0.0) and np.all(y_pred_seasonal <= 1.0)):
            raise ValueError(
                f"Seasonal predictions not in [0.0, 1.0]: "
                f"min={y_pred_seasonal.min()}, max={y_pred_seasonal.max()}"
            )
        
        # Create DataFrame from arrays with correct column order
        submission_df = pd.DataFrame({
            'respondent_id': respondent_ids,
            'h1n1_vaccine': y_pred_h1n1,
            'seasonal_vaccine': y_pred_seasonal,
        })
        
        return submission_df
    
    @staticmethod
    def validate_submission(
        submission_df: pd.DataFrame,
        submission_template_path: Optional[str] = None,
    ) -> bool:
        """
        Validate submission format against competition requirements.
        
        Checks:
        - DataFrame has exactly 3 columns: respondent_id, h1n1_vaccine, seasonal_vaccine
        - Column names match exactly (case-sensitive)
        - Column order is correct
        - respondent_id column has no missing values and is integer type
        - Vaccine columns contain only floats in [0.0, 1.0]
        - No NaN or infinite values in vaccine columns
        - Number of rows matches expected test set size (if template provided)
        
        Parameters:
            submission_df (pd.DataFrame): Submission to validate
            submission_template_path (Optional[str]): Path to submission format
                template CSV. If provided, number of rows must match.
        
        Returns:
            bool: True if submission is valid
        
        Raises:
            ValueError: If submission doesn't meet requirements
                Error message describes which requirements failed
            
        Implementation notes:
            - TODO: Check column names exactly match ['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine']
            - TODO: Check column order
            - TODO: Check respondent_id: integer, no NaN, unique (if needed)
            - TODO: Check vaccine columns: float, in [0, 1], no NaN/inf
            - TODO: If template provided, check row count matches
            - TODO: Return True if all checks pass
            - TODO: Raise ValueError with clear message if any check fails
        """
        expected_columns = ['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine']
        
        # Check column names exactly match
        if list(submission_df.columns) != expected_columns:
            raise ValueError(
                f"Column names don't match. Expected {expected_columns}, "
                f"got {list(submission_df.columns)}"
            )
        
        # Check respondent_id column
        if submission_df['respondent_id'].isna().any():
            raise ValueError("respondent_id column contains NaN values")
        
        # Check vaccine columns: float, in [0, 1], no NaN/inf
        for col in ['h1n1_vaccine', 'seasonal_vaccine']:
            if submission_df[col].isna().any():
                raise ValueError(f"{col} column contains NaN values")
            
            if np.isinf(submission_df[col]).any():
                raise ValueError(f"{col} column contains infinite values")
            
            if not (submission_df[col] >= 0.0).all():
                raise ValueError(f"{col} contains values below 0.0")
            
            if not (submission_df[col] <= 1.0).all():
                raise ValueError(f"{col} contains values above 1.0")
        
        # If template provided, check row count matches
        if submission_template_path is not None:
            try:
                template_df = pd.read_csv(submission_template_path)
                if len(submission_df) != len(template_df):
                    raise ValueError(
                        f"Row count mismatch. Expected {len(template_df)}, "
                        f"got {len(submission_df)}"
                    )
            except FileNotFoundError:
                raise ValueError(f"Template file not found: {submission_template_path}")
        
        return True
    
    @staticmethod
    def save_submission(
        submission_df: pd.DataFrame,
        output_path: str,
        validate_before_save: bool = True,
    ) -> None:
        """
        Save submission DataFrame to CSV file.
        
        Optionally validates format before saving. Uses recommended
        CSV settings: no index, UTF-8 encoding.
        
        Parameters:
            submission_df (pd.DataFrame): Submission to save
            output_path (str): Path to output CSV file
            validate_before_save (bool): If True, validate before saving. Default: True
        
        Raises:
            ValueError: If validate_before_save=True and validation fails
            IOError: If file cannot be written
            
        Implementation notes:
            - TODO: If validate_before_save, call validate_submission()
            - TODO: Save DataFrame to CSV with no index
            - TODO: Use UTF-8 encoding
            - TODO: Use float_format='%.10f' for probabilities (10 decimal places)
            - TODO: Create parent directories if needed
            - TODO: Return None (side effect is file creation)
        """
        # If validate_before_save, call validate_submission()
        if validate_before_save:
            PredictionEngine.validate_submission(submission_df)
        
        # Create parent directories if needed
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame to CSV with recommended settings
        submission_df.to_csv(
            output_path,
            index=False,
            encoding='utf-8',
            float_format='%.10f'
        )


class DefaultPredictionEngine(PredictionEngine):
    """
    Default implementation of PredictionEngine.
    
    Concrete implementation that handles standard prediction workflow.
    
    Implementation notes:
        - TODO: In predict_test_set(), apply preprocessing and get predictions
        - TODO: In format_submission(), create properly formatted DataFrame
        - TODO: In validate_submission(), check all format requirements
        - TODO: In save_submission(), write to CSV with proper settings
    """
    
    def predict_test_set(
        self,
        X_test: pd.DataFrame,
        model: object,
        preprocessing_pipeline: Optional[object] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test set.
        
        Implementation notes:
            - TODO: Validate X_test
            - TODO: Apply preprocessing if provided
            - TODO: Get predictions from model
            - TODO: Handle multi-output models (binary classifiers for each vaccine)
            - TODO: Return two 1D probability arrays
        """
        # Validate X_test shape
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError(f"X_test must be DataFrame, got {type(X_test)}")
        
        # Apply preprocessing if provided
        X_test_processed = X_test.copy()
        if preprocessing_pipeline is not None:
            X_test_processed = preprocessing_pipeline.transform(X_test_processed)
        
        # Get predictions from model
        # The model should be a dictionary with h1n1_model and seasonal_model keys
        if isinstance(model, dict):
            h1n1_model = model.get('h1n1_model')
            seasonal_model = model.get('seasonal_model')
            
            if h1n1_model is None or seasonal_model is None:
                raise ValueError("Model dict must contain 'h1n1_model' and 'seasonal_model' keys")
            
            # Get probabilities for positive class (column 1)
            y_pred_h1n1 = h1n1_model.predict_proba(X_test_processed)[:, 1]
            y_pred_seasonal = seasonal_model.predict_proba(X_test_processed)[:, 1]
        else:
            # If model is a single object with two outputs, try to extract them
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test_processed)
                # Assuming proba is (n_samples, 2) for binary classification
                if len(proba.shape) == 2 and proba.shape[1] == 2:
                    y_pred_h1n1 = proba[:, 1]
                    y_pred_seasonal = proba[:, 1]  # Same model for both
                else:
                    raise ValueError(f"Unexpected predict_proba shape: {proba.shape}")
            else:
                raise TypeError("Model must have predict_proba() method")
        
        # Validate output shapes and values
        if len(y_pred_h1n1) != len(X_test_processed):
            raise ValueError(
                f"Prediction length mismatch: got {len(y_pred_h1n1)}, "
                f"expected {len(X_test_processed)}"
            )
        
        if not (np.all(y_pred_h1n1 >= 0.0) and np.all(y_pred_h1n1 <= 1.0)):
            raise ValueError(
                f"H1N1 predictions not in [0.0, 1.0]: "
                f"min={y_pred_h1n1.min()}, max={y_pred_h1n1.max()}"
            )
        
        if not (np.all(y_pred_seasonal >= 0.0) and np.all(y_pred_seasonal <= 1.0)):
            raise ValueError(
                f"Seasonal predictions not in [0.0, 1.0]: "
                f"min={y_pred_seasonal.min()}, max={y_pred_seasonal.max()}"
            )
        
        return y_pred_h1n1, y_pred_seasonal


def load_submission_template(
    template_path: str = 'data/submission_format.csv',
) -> pd.DataFrame:
    """
    Load submission format template from file.
    
    Reads the provided submission_format.csv template to get
    expected structure and row count for validation.
    
    Parameters:
        template_path (str): Path to submission template CSV.
            Default: 'data/submission_format.csv'
    
    Returns:
        pd.DataFrame: Template DataFrame with structure:
            - respondent_id: Test set respondent IDs
            - h1n1_vaccine: Placeholder values (0.5)
            - seasonal_vaccine: Placeholder values (0.5)
    
    Raises:
        FileNotFoundError: If template file doesn't exist
        
    Implementation notes:
        - TODO: Check file exists
        - TODO: Read CSV file
        - TODO: Validate it has expected columns
        - TODO: Return DataFrame
    """
    # Check file exists
    template_path_obj = Path(template_path)
    if not template_path_obj.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Read CSV file
    template_df = pd.read_csv(template_path_obj)
    
    # Validate it has expected columns
    expected_columns = ['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine']
    if list(template_df.columns) != expected_columns:
        raise ValueError(
            f"Template has wrong columns. Expected {expected_columns}, "
            f"got {list(template_df.columns)}"
        )
    
    return template_df


class TestPredictor:
    """
    Predictor for generating test set predictions after training.
    
    This class applies a preprocessing pipeline fitted on full training data
    to test features and generates probability predictions from two independently
    trained models (h1n1 and seasonal vaccines).
    
    Attributes:
        preprocessing_pipeline: Fitted preprocessing (imputation + encoding)
        h1n1_model: Fitted model for h1n1 vaccine predictions
        seasonal_model: Fitted model for seasonal vaccine predictions
    
    Example:
        ```python
        predictor = TestPredictor(
            preprocessing_pipeline=fitted_pipeline,
            h1n1_model=h1n1_model,
            seasonal_model=seasonal_model,
        )
        
        # Generate predictions and submission
        submission_df = predictor.predict(X_test, respondent_ids)
        
        # Validate and save
        PredictionEngine.validate_submission(submission_df)
        PredictionEngine.save_submission(submission_df, 'submission.csv')
        ```
    """
    
    def __init__(
        self,
        preprocessing_pipeline: object,
        h1n1_model: object,
        seasonal_model: object,
    ):
        """
        Initialize test predictor with fitted components.
        
        Parameters:
            preprocessing_pipeline: Fitted preprocessing pipeline with transform() method.
                Must have been fit on full training data.
            h1n1_model: Fitted model for h1n1 vaccine with predict_proba() method.
            seasonal_model: Fitted model for seasonal vaccine with predict_proba() method.
        
        Raises:
            ValueError: If models don't have predict_proba() method
        """
        if not hasattr(h1n1_model, 'predict_proba'):
            raise ValueError("h1n1_model must have predict_proba() method")
        
        if not hasattr(seasonal_model, 'predict_proba'):
            raise ValueError("seasonal_model must have predict_proba() method")
        
        if preprocessing_pipeline is not None and not hasattr(preprocessing_pipeline, 'transform'):
            raise ValueError("preprocessing_pipeline must have transform() method")
        
        self.preprocessing_pipeline = preprocessing_pipeline
        self.h1n1_model = h1n1_model
        self.seasonal_model = seasonal_model
    
    def _validate_unknown_categories(
        self,
        X_test: pd.DataFrame,
        logger: Optional[object] = None,
    ) -> None:
        """
        Validate test data for unknown categorical values not seen in training.
        
        Checks if the preprocessing pipeline contains a OneHotEncoder with
        unknown categories detection capability. If unknown categories are found,
        logs warnings with details about which categories are new and their
        frequency.
        
        Parameters:
            X_test (pd.DataFrame): Test set features to validate
            logger (Optional[object]): Logger instance with warning() method.
                If None, unknown category warnings are not logged.
        
        Implementation notes:
            - Extracts OneHotEncoder from preprocessing pipeline
            - Calls detect_unknown_categories() if available
            - Logs warnings for features with unknown categories
            - Warning includes: feature name, unknown categories, count, percentage
        """
        if self.preprocessing_pipeline is None:
            return
        
        # Try to find the OneHotEncoder in the preprocessing pipeline
        one_hot_encoder = None
        
        # If preprocessing_pipeline has an 'encoder' attribute (from PreprocessingPipeline)
        if hasattr(self.preprocessing_pipeline, 'encoder'):
            one_hot_encoder = self.preprocessing_pipeline.encoder
        # If preprocessing_pipeline is the encoder itself
        elif hasattr(self.preprocessing_pipeline, 'detect_unknown_categories'):
            one_hot_encoder = self.preprocessing_pipeline
        
        if one_hot_encoder is None or not hasattr(one_hot_encoder, 'detect_unknown_categories'):
            return  # No OneHotEncoder found, skip validation
        
        # Detect unknown categories
        try:
            unknown_info = one_hot_encoder.detect_unknown_categories(X_test)
            
            # Log warnings if unknown categories found
            if unknown_info and logger is not None:
                logger.warning(
                    f"Detected {len(unknown_info)} feature(s) with unknown categories in test data"
                )
                for feature, info in unknown_info.items():
                    logger.warning(
                        f"  Feature '{feature}': "
                        f"{info['unknown_count']} samples ({info['unknown_pct']}%) "
                        f"contain unknown categories: {info['unknown_categories']}"
                    )
        except Exception as e:
            # If validation fails, log warning but don't raise (predictions still valid with ignore)
            if logger is not None:
                logger.warning(
                    f"Could not validate unknown categories: {str(e)}"
                )
    
    def predict(
        self,
        X_test: pd.DataFrame,
        respondent_ids: Optional[np.ndarray] = None,
        logger: Optional[object] = None,
    ) -> pd.DataFrame:
        """
        Generate test set predictions and format as submission DataFrame.
        
        Applies fitted preprocessing to test features, generates probability
        predictions from both models, and returns submission-ready DataFrame
        with respondent_id, h1n1_vaccine, and seasonal_vaccine columns.
        
        Parameters:
            X_test (pd.DataFrame): Test set features (n_samples, n_features)
            respondent_ids (Optional[np.ndarray]): Test set respondent IDs.
                If None, uses X_test.index.
            logger (Optional[object]): Logger for warnings about unknown categories.
                If provided, will log warnings about unknown categorical values.
        
        Returns:
            pd.DataFrame: Submission DataFrame with columns:
                ['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine']
                All probabilities are in [0.0, 1.0] range.
        
        Raises:
            ValueError: If preprocessing fails, predictions are invalid, or NaN values detected
            TypeError: If X_test is not a DataFrame
        
        Implementation:
            1. Validate inputs (X_test is DataFrame, has respondent_ids)
            2. Validate for unknown categories in test data and log warnings
            3. Apply fitted preprocessing pipeline to X_test
            4. Generate h1n1 predictions via h1n1_model.predict_proba()
            5. Generate seasonal predictions via seasonal_model.predict_proba()
            6. Extract positive class probabilities (column 1)
            7. Validate no NaN values in predictions
            8. Validate probabilities in [0.0, 1.0] range
            9. Create and return submission DataFrame
        """
        # Validate inputs
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError(f"X_test must be DataFrame, got {type(X_test)}")
        
        # Get respondent IDs
        if respondent_ids is None:
            if hasattr(X_test.index, 'name') and X_test.index.name == 'respondent_id':
                respondent_ids = X_test.index.values
            elif 'respondent_id' in X_test.columns:
                respondent_ids = X_test['respondent_id'].values
                X_test = X_test.drop('respondent_id', axis=1)
            else:
                respondent_ids = np.arange(len(X_test))
        
        # Check for unknown categories before preprocessing
        self._validate_unknown_categories(X_test, logger)
        
        # Apply fitted preprocessing pipeline
        X_test_processed = X_test.copy()
        if self.preprocessing_pipeline is not None:
            X_test_processed = self.preprocessing_pipeline.transform(X_test_processed)
        
        # Generate probability predictions
        h1n1_proba = self.h1n1_model.predict_proba(X_test_processed)
        seasonal_proba = self.seasonal_model.predict_proba(X_test_processed)
        
        # Extract positive class probabilities (column 1)
        # Handle both cases: (n_samples, 2) from sklearn binary classifiers
        if len(h1n1_proba.shape) == 2 and h1n1_proba.shape[1] == 2:
            y_pred_h1n1 = h1n1_proba[:, 1]
        else:
            y_pred_h1n1 = h1n1_proba
        
        if len(seasonal_proba.shape) == 2 and seasonal_proba.shape[1] == 2:
            y_pred_seasonal = seasonal_proba[:, 1]
        else:
            y_pred_seasonal = seasonal_proba
        
        # Validate no NaN values
        if np.isnan(y_pred_h1n1).any():
            raise ValueError("H1N1 predictions contain NaN values")
        if np.isnan(y_pred_seasonal).any():
            raise ValueError("Seasonal predictions contain NaN values")
        
        # Validate probabilities in [0.0, 1.0] range
        if not (np.all(y_pred_h1n1 >= 0.0) and np.all(y_pred_h1n1 <= 1.0)):
            raise ValueError(
                f"H1N1 predictions not in [0.0, 1.0]: "
                f"min={y_pred_h1n1.min()}, max={y_pred_h1n1.max()}"
            )
        
        if not (np.all(y_pred_seasonal >= 0.0) and np.all(y_pred_seasonal <= 1.0)):
            raise ValueError(
                f"Seasonal predictions not in [0.0, 1.0]: "
                f"min={y_pred_seasonal.min()}, max={y_pred_seasonal.max()}"
            )
        
        # Create and return submission DataFrame
        submission_df = PredictionEngine.format_submission(
            respondent_ids,
            y_pred_h1n1,
            y_pred_seasonal,
        )
        
        return submission_df
