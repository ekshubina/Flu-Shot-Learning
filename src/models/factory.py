"""
Machine learning model interfaces and implementations.

This module defines the abstract interface for ML models and provides concrete
implementations for various algorithms suitable for the H1N1 flu shot prediction
multilabel classification task.

All models must support:
- Binary classification (independent predictions for h1n1_vaccine and seasonal_vaccine)
- Probability predictions (probabilities in 0.0-1.0 range for ROC AUC evaluation)
- Feature importance computation for model interpretation
- Parameter getting/setting for hyperparameter optimization

Supported models:
- LogisticRegression: Baseline linear model, fast and interpretable
- XGBoost: Gradient boosting, strong predictive power
- LightGBM: Fast gradient boosting, memory efficient
- RandomForest: Ensemble tree method, good for mixed feature types
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class BaseModel(ABC):
    """
    Abstract base class for machine learning models.
    
    All models must implement the sklearn-like interface:
    - fit(X, y) - Train the model
    - predict_proba(X) - Return probability predictions
    - get_feature_importance() - Return feature importance scores
    - get_params() - Get hyperparameters
    - set_params(**params) - Set hyperparameters
    
    This enables:
    - Consistent interface across different algorithms
    - Easy model swapping and comparison
    - Hyperparameter optimization (GridSearchCV, RandomizedSearchCV, Bayesian)
    - Feature importance analysis
    
    For multilabel classification, each model predicts a single binary target.
    Two independent models should be trained (one for h1n1_vaccine, one for seasonal_vaccine),
    or one model per target, or a single multitask model.
    
    Attributes:
        model_type: Name of the model (e.g., 'logistic_regression', 'xgboost')
        fitted: Whether the model has been fit to training data
        feature_names: List of input feature column names (set during fit)
        feature_importances_: Feature importance scores after fitting
        
    Example:
        >>> from src.models import LogisticRegressionModel
        >>> from src.config import ModelConfig
        >>> 
        >>> config = ModelConfig(
        ...     model_type='logistic_regression',
        ...     hyperparameters={'C': 1.0, 'max_iter': 1000}
        ... )
        >>> model = LogisticRegressionModel(config)
        >>> model.fit(X_train, y_train)
        >>> proba = model.predict_proba(X_test)  # Shape: (n_samples, 2)
        >>> importance = model.get_feature_importance()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model with configuration.
        
        Args:
            config: Configuration dict or ModelConfig dataclass with:
                   - model_type: str name of model type
                   - hyperparameters: dict of model-specific parameters
                   - random_seed: int for reproducibility
                   - n_jobs: int number of parallel jobs
        """
        self.config = config or {}
        self.model_type = self.config.get("model_type", "base")
        self.fitted = False
        self.feature_names = None
        self.feature_importances_ = None
        self._model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """
        Train the model on data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) with values 0 or 1
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If X or y are invalid
            ValueError: If dimensions don't match
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for test data.
        
        Must return probabilities (NOT hard predictions). For binary classification,
        return shape (n_samples, 2) with probabilities for class 0 and 1.
        
        Args:
            X: Test features (n_samples, n_features)
            
        Returns:
            Probability predictions (n_samples, 2) where each row sums to 1.0
            Column 0: P(y=0), Column 1: P(y=1)
            
        Raises:
            ValueError: If model not fitted yet
            ValueError: If X has different features than training data
        """
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict hard class labels (0 or 1).
        
        Converts probabilities to hard predictions using threshold 0.5.
        
        Args:
            X: Test features
            
        Returns:
            Hard predictions (n_samples,) with values 0 or 1
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns a DataFrame with feature names and importance scores, sorted
        by importance (descending). Interpretation depends on model type:
        - Linear models: Absolute coefficient magnitude
        - Tree models: Mean decrease in impurity or gain
        - Permutation: Loss increase when feature is shuffled
        
        Returns:
            DataFrame with columns:
            - 'feature': Feature name
            - 'importance': Importance score
            - Sorted by importance descending
            
        Raises:
            ValueError: If model not fitted yet or doesn't support feature importance
        """
        pass

    @abstractmethod
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get model hyperparameters.
        
        Follows sklearn convention. Used for hyperparameter optimization and
        model reconstruction.
        
        Args:
            deep: If True, return nested parameters (default: True)
            
        Returns:
            Dictionary of parameter names and values
        """
        pass

    @abstractmethod
    def set_params(self, **params) -> "BaseModel":
        """
        Set model hyperparameters.
        
        Follows sklearn convention. Used for hyperparameter optimization.
        
        Args:
            **params: Parameter name=value keyword arguments
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If any parameter is invalid for this model
        """
        pass

    def get_model_name(self) -> str:
        """
        Get human-readable model name.
        
        Returns:
            str: Model name and type (e.g., "LogisticRegression (sklearn)")
        """
        return f"{self.__class__.__name__} ({self.model_type})"

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get hyperparameters as configured (convenience method).
        
        Returns:
            Dictionary of hyperparameters
        """
        return self.get_params()

    def is_fitted(self) -> bool:
        """
        Check if model has been fitted.
        
        Returns:
            bool: True if model has been fit to training data
        """
        return self.fitted


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression model for binary classification.
    
    Fast, interpretable linear model suitable as baseline. Works well for
    this problem because decision boundaries can often be approximated linearly.
    
    Supports:
    - Class weighting to handle imbalance
    - L1/L2 regularization
    - Multiple solvers (liblinear, lbfgs, etc.)
    
    Hyperparameters:
        - C: Inverse regularization strength (default: 1.0)
        - penalty: 'l2' or 'l1' regularization (default: 'l2')
        - solver: 'liblinear', 'lbfgs', 'newton-cg', etc. (default: 'lbfgs')
        - max_iter: Maximum iterations for solver (default: 1000)
        - class_weight: 'balanced' or dict (default: None)
        - random_state: Random seed (default: 42)
    
    Implementation notes:
        - TODO: Wrap sklearn LogisticRegression
        - TODO: In fit(), store feature names and train model
        - TODO: In predict_proba(), return probability matrix
        - TODO: In get_feature_importance(), return coefficients
        - TODO: Implement get_params/set_params
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LogisticRegression model."""
        super().__init__(config)
        self.model_type = "logistic_regression"
        
        # Extract hyperparameters from config
        hyperparams = {}
        if config and isinstance(config, dict) and "hyperparameters" in config:
            hyperparams = config["hyperparameters"]
        
        # Set default hyperparameters with config overrides
        self.C = hyperparams.get("C", 1.0)
        self.penalty = hyperparams.get("penalty", "l2")
        self.solver = hyperparams.get("solver", "lbfgs")
        self.max_iter = hyperparams.get("max_iter", 1000)
        self.class_weight = hyperparams.get("class_weight", "balanced")
        self.random_state = hyperparams.get("random_state", 42)
        
        # Initialize the sklearn model
        self._model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        """
        Train logistic regression model.
        
        Implementation notes:
            - Store feature names from X.columns
            - Create sklearn LogisticRegression with hyperparameters
            - Fit on (X, y)
            - Set self.fitted = True
        """
        # Store feature names for later use
        self.feature_names = list(X.columns)
        
        # Train the sklearn LogisticRegression model
        self._model.fit(X, y)
        
        # Set fitted flag
        self.fitted = True
        
        # Store coefficients as feature importances (absolute value)
        self.feature_importances_ = np.abs(self._model.coef_[0])
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability predictions.
        
        Implementation notes:
            - Check if fitted
            - Call self._model.predict_proba(X)
            - Return (n_samples, 2) array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calling predict_proba(). Call fit() first.")
        
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get logistic regression coefficients as importance.
        
        Implementation notes:
            - Get coefficients from self._model.coef_[0]
            - Use absolute value for importance
            - Create DataFrame with feature names and importance
            - Sort by importance descending
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calling get_feature_importance(). Call fit() first.")
        
        # Get coefficients and convert to absolute value for importance
        importance_scores = np.abs(self._model.coef_[0])
        
        # Create DataFrame with feature names and importance
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance_scores
        })
        
        # Sort by importance descending
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        
        return importance_df

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state
        }
        return params

    def set_params(self, **params) -> "LogisticRegressionModel":
        """Set model parameters."""
        # Update instance attributes
        for key, value in params.items():
            if key == "C":
                self.C = value
            elif key == "penalty":
                self.penalty = value
            elif key == "solver":
                self.solver = value
            elif key == "max_iter":
                self.max_iter = value
            elif key == "class_weight":
                self.class_weight = value
            elif key == "random_state":
                self.random_state = value
        
        # Recreate the sklearn model with updated parameters
        self._model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        
        # Reset fitted flag
        self.fitted = False
        
        return self


class XGBoostModel(BaseModel):
    """
    XGBoost gradient boosting model for binary classification.
    
    Powerful ensemble method that often achieves strong performance. Handles
    mixed feature types, non-linear relationships, and feature interactions well.
    
    Supports:
    - Early stopping to prevent overfitting
    - Feature importance (gain, split, cover)
    - Custom loss functions
    - GPU acceleration
    
    Hyperparameters:
        - n_estimators: Number of boosting rounds (default: 100)
        - max_depth: Tree depth (default: 6)
        - learning_rate: Shrinkage factor (default: 0.1)
        - subsample: Row subsampling (default: 1.0)
        - colsample_bytree: Column subsampling (default: 1.0)
        - lambda: L2 regularization (default: 1.0)
        - alpha: L1 regularization (default: 0.0)
        - random_state: Random seed (default: 42)
    
    Implementation notes:
        - TODO: Wrap xgboost.XGBClassifier (requires xgboost package)
        - TODO: In fit(), train with early stopping if eval_set available
        - TODO: In predict_proba(), return probability matrix
        - TODO: In get_feature_importance(), return feature importance scores
        - TODO: Implement get_params/set_params
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost model."""
        super().__init__(config)
        self.model_type = "xgboost"
        
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "xgboost package required for XGBoostModel. "
                "Install with: pip install xgboost"
            )
        
        # Extract hyperparameters from config
        hyperparams = {}
        if config and isinstance(config, dict) and "hyperparameters" in config:
            hyperparams = config["hyperparameters"]
        
        # Set default hyperparameters with config overrides
        self.n_estimators = hyperparams.get("n_estimators", 100)
        self.max_depth = hyperparams.get("max_depth", 6)
        self.learning_rate = hyperparams.get("learning_rate", 0.1)
        self.subsample = hyperparams.get("subsample", 1.0)
        self.colsample_bytree = hyperparams.get("colsample_bytree", 1.0)
        self.reg_lambda = hyperparams.get("reg_lambda", 1.0)
        self.reg_alpha = hyperparams.get("reg_alpha", 0.0)
        self.random_state = hyperparams.get("random_state", 42)
        
        # Initialize the XGBoost model
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        """
        Train XGBoost model.
        
        Trains the XGBoost classifier on the provided training data and stores
        feature importances.
        """
        # Store feature names for later use
        self.feature_names = list(X.columns)
        
        # Train the XGBoost model
        self._model.fit(X, y)
        
        # Set fitted flag
        self.fitted = True
        
        # Store feature importances
        self.feature_importances_ = self._model.feature_importances_
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability predictions.
        
        Returns probability estimates for binary classification.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calling predict_proba(). Call fit() first.")
        
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get XGBoost feature importance scores.
        
        Returns feature importances as a sorted DataFrame.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calling get_feature_importance(). Call fit() first.")
        
        # Get importance scores
        importance_scores = self.feature_importances_
        
        # Create DataFrame with feature names and importance
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance_scores
        })
        
        # Sort by importance descending
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        
        return importance_df

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "random_state": self.random_state,
        }
        return params

    def set_params(self, **params) -> "XGBoostModel":
        """Set model parameters."""
        import xgboost as xgb
        
        # Update instance attributes
        for key, value in params.items():
            if key == "n_estimators":
                self.n_estimators = value
            elif key == "max_depth":
                self.max_depth = value
            elif key == "learning_rate":
                self.learning_rate = value
            elif key == "subsample":
                self.subsample = value
            elif key == "colsample_bytree":
                self.colsample_bytree = value
            elif key == "reg_lambda":
                self.reg_lambda = value
            elif key == "reg_alpha":
                self.reg_alpha = value
            elif key == "random_state":
                self.random_state = value
        
        # Recreate the XGBoost model with updated parameters
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Reset fitted flag
        self.fitted = False
        
        return self


class LightGBMModel(BaseModel):
    """
    LightGBM gradient boosting model for binary classification.
    
    Fast and memory-efficient gradient boosting alternative to XGBoost.
    Often converges faster with similar or better performance.
    
    Supports:
    - Categorical feature handling (no need for one-hot encoding)
    - Early stopping
    - Feature importance
    - GPU acceleration
    
    Hyperparameters:
        - n_estimators: Number of boosting rounds (default: 100)
        - max_depth: Tree depth (default: -1 for unlimited)
        - learning_rate: Shrinkage factor (default: 0.1)
        - num_leaves: Max leaves per tree (default: 31)
        - subsample: Row subsampling (default: 1.0)
        - colsample_bytree: Column subsampling (default: 1.0)
        - lambda_l2: L2 regularization (default: 0.0)
        - lambda_l1: L1 regularization (default: 0.0)
        - random_state: Random seed (default: 42)
    
    Implementation notes:
        - TODO: Wrap lightgbm.LGBMClassifier (requires lightgbm package)
        - TODO: In fit(), train with early stopping if eval_set available
        - TODO: In predict_proba(), return probability matrix
        - TODO: In get_feature_importance(), return feature importance scores
        - TODO: Implement get_params/set_params
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LightGBM model."""
        super().__init__(config)
        self.model_type = "lightgbm"
        
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm package required for LightGBMModel. "
                "Install with: pip install lightgbm"
            )
        
        # Extract hyperparameters from config
        hyperparams = {}
        if config and isinstance(config, dict) and "hyperparameters" in config:
            hyperparams = config["hyperparameters"]
        
        # Set default hyperparameters with config overrides
        self.n_estimators = hyperparams.get("n_estimators", 100)
        self.max_depth = hyperparams.get("max_depth", -1)
        self.learning_rate = hyperparams.get("learning_rate", 0.1)
        self.num_leaves = hyperparams.get("num_leaves", 31)
        self.subsample = hyperparams.get("subsample", 1.0)
        self.colsample_bytree = hyperparams.get("colsample_bytree", 1.0)
        self.reg_lambda = hyperparams.get("reg_lambda", 0.0)
        self.reg_alpha = hyperparams.get("reg_alpha", 0.0)
        self.random_state = hyperparams.get("random_state", 42)
        
        # Initialize the LightGBM model
        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        """
        Train LightGBM model.
        
        Trains the LightGBM classifier on the provided training data and stores
        feature importances.
        """
        # Store feature names for later use
        self.feature_names = list(X.columns)
        
        # Train the LightGBM model
        self._model.fit(X, y)
        
        # Set fitted flag
        self.fitted = True
        
        # Store feature importances
        self.feature_importances_ = self._model.feature_importances_
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability predictions.
        
        Returns probability estimates for binary classification.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calling predict_proba(). Call fit() first.")
        
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get LightGBM feature importance scores.
        
        Returns feature importances as a sorted DataFrame.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calling get_feature_importance(). Call fit() first.")
        
        # Get importance scores
        importance_scores = self.feature_importances_
        
        # Create DataFrame with feature names and importance
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance_scores
        })
        
        # Sort by importance descending
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        
        return importance_df

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "random_state": self.random_state,
        }
        return params

    def set_params(self, **params) -> "LightGBMModel":
        """Set model parameters."""
        import lightgbm as lgb
        
        # Update instance attributes
        for key, value in params.items():
            if key == "n_estimators":
                self.n_estimators = value
            elif key == "max_depth":
                self.max_depth = value
            elif key == "learning_rate":
                self.learning_rate = value
            elif key == "num_leaves":
                self.num_leaves = value
            elif key == "subsample":
                self.subsample = value
            elif key == "colsample_bytree":
                self.colsample_bytree = value
            elif key == "reg_lambda":
                self.reg_lambda = value
            elif key == "reg_alpha":
                self.reg_alpha = value
            elif key == "random_state":
                self.random_state = value
        
        # Recreate the LightGBM model with updated parameters
        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state
        )
        
        # Reset fitted flag
        self.fitted = False
        
        return self


class RandomForestModel(BaseModel):
    """
    Random Forest ensemble model for binary classification.
    
    Robust ensemble of decision trees. Good for mixed feature types,
    handles non-linearity and interactions. Less prone to overfitting than single trees.
    
    Supports:
    - Feature importance via impurity reduction
    - Out-of-bag error estimation
    - Parallel training
    
    Hyperparameters:
        - n_estimators: Number of trees (default: 100)
        - max_depth: Tree depth (default: None for unlimited)
        - min_samples_split: Min samples to split node (default: 2)
        - min_samples_leaf: Min samples in leaf (default: 1)
        - max_features: Features per split ('sqrt', 'log2', or number) (default: 'sqrt')
        - bootstrap: Use bootstrap samples (default: True)
        - class_weight: 'balanced' or dict for class imbalance (default: None)
        - random_state: Random seed (default: 42)
        - n_jobs: Parallel jobs (default: 1)
    
    Implementation notes:
        - TODO: Wrap sklearn RandomForestClassifier
        - TODO: In fit(), store feature names and train ensemble
        - TODO: In predict_proba(), return probability matrix
        - TODO: In get_feature_importance(), return feature_importances_
        - TODO: Implement get_params/set_params
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Random Forest model."""
        super().__init__(config)
        self.model_type = "random_forest"
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Extract hyperparameters from config
        hyperparams = {}
        if config and isinstance(config, dict) and "hyperparameters" in config:
            hyperparams = config["hyperparameters"]
        
        # Set default hyperparameters with config overrides
        self.n_estimators = hyperparams.get("n_estimators", 100)
        self.max_depth = hyperparams.get("max_depth", None)
        self.min_samples_split = hyperparams.get("min_samples_split", 2)
        self.min_samples_leaf = hyperparams.get("min_samples_leaf", 1)
        self.max_features = hyperparams.get("max_features", "sqrt")
        self.bootstrap = hyperparams.get("bootstrap", True)
        self.class_weight = hyperparams.get("class_weight", None)
        self.random_state = hyperparams.get("random_state", 42)
        self.n_jobs = hyperparams.get("n_jobs", 1)
        
        # Initialize the Random Forest model
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """
        Train random forest model.
        
        Trains the Random Forest classifier on the provided training data and stores
        feature importances.
        """
        # Store feature names for later use
        self.feature_names = list(X.columns)
        
        # Train the Random Forest model
        self._model.fit(X, y)
        
        # Set fitted flag
        self.fitted = True
        
        # Store feature importances
        self.feature_importances_ = self._model.feature_importances_
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability predictions.
        
        Returns probability estimates for binary classification.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calling predict_proba(). Call fit() first.")
        
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get Random Forest feature importance scores.
        
        Returns feature importances as a sorted DataFrame.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calling get_feature_importance(). Call fit() first.")
        
        # Get importance scores
        importance_scores = self.feature_importances_
        
        # Create DataFrame with feature names and importance
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance_scores
        })
        
        # Sort by importance descending
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        
        return importance_df

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        return params

    def set_params(self, **params) -> "RandomForestModel":
        """Set model parameters."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Update instance attributes
        for key, value in params.items():
            if key == "n_estimators":
                self.n_estimators = value
            elif key == "max_depth":
                self.max_depth = value
            elif key == "min_samples_split":
                self.min_samples_split = value
            elif key == "min_samples_leaf":
                self.min_samples_leaf = value
            elif key == "max_features":
                self.max_features = value
            elif key == "bootstrap":
                self.bootstrap = value
            elif key == "class_weight":
                self.class_weight = value
            elif key == "random_state":
                self.random_state = value
            elif key == "n_jobs":
                self.n_jobs = value
        
        # Recreate the Random Forest model with updated parameters
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Reset fitted flag
        self.fitted = False
        
        return self


class ModelFactory:
    """
    Factory for creating model instances by type.
    
    Provides a registry pattern for model creation, enabling easy swapping
    of different models and centralized model configuration.
    
    Supported models:
    - 'logistic_regression': LogisticRegressionModel
    - 'xgboost': XGBoostModel
    - 'lightgbm': LightGBMModel
    - 'random_forest': RandomForestModel
    
    Example:
        >>> from src.models import ModelFactory
        >>> from src.config import ModelConfig
        >>> 
        >>> factory = ModelFactory()
        >>> config = ModelConfig(
        ...     model_type='xgboost',
        ...     hyperparameters={'n_estimators': 100, 'max_depth': 6}
        ... )
        >>> model = factory.create_model(config)
        >>> model.fit(X_train, y_train)
    """

    # Registry of available models
    MODELS = {
        "logistic_regression": LogisticRegressionModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "random_forest": RandomForestModel,
    }

    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on configuration.
        
        Args:
            config: Configuration dict or ModelConfig dataclass with model_type
            
        Returns:
            BaseModel subclass instance
            
        Raises:
            ValueError: If model_type not in supported models
            
        Example:
            >>> config = {'model_type': 'xgboost', 'hyperparameters': {...}}
            >>> model = ModelFactory.create_model(config)
        """
        model_type = config.get("model_type", "logistic_regression")
        
        if model_type not in cls.MODELS:
            available = ", ".join(cls.MODELS.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported models: {available}"
            )
        
        model_class = cls.MODELS[model_type]
        return model_class(config)

    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List of model type names
        """
        return list(cls.MODELS.keys())

    @classmethod
    def register_model(cls, model_type: str, model_class: type) -> None:
        """
        Register a custom model class.
        
        Allows adding new model types to the factory.
        
        Args:
            model_type: Name to register model under
            model_class: Model class (must subclass BaseModel)
            
        Raises:
            TypeError: If model_class doesn't subclass BaseModel
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class} must subclass BaseModel")
        cls.MODELS[model_type] = model_class
