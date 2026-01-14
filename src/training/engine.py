"""
Model training and validation orchestration.

This module provides the training engine that orchestrates the entire training
workflow: cross-validation, hyperparameter search, threshold tuning, and model
evaluation.

The training engine handles:
- Stratified k-fold cross-validation respecting multilabel targets
- Hyperparameter optimization (grid search, random search, Bayesian)
- Class imbalance handling (class weights, SMOTE, threshold tuning)
- Early stopping and model selection
- Caching of fold predictions for ensemble stacking

See SYSTEM_DESIGN.md and PROBLEM_DESCRIPTION.md for context on multilabel
classification and evaluation metrics (ROC AUC).
"""

from abc import abstractmethod
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import itertools
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class FoldResults:
    """
    Results from a single cross-validation fold.
    
    Attributes:
        fold_id: Fold number (0-indexed)
        train_indices: Indices of training samples
        val_indices: Indices of validation samples
        y_val_true: True validation labels
        y_val_proba: Validation probability predictions
        metrics: Dictionary of validation metrics (auroc, accuracy, etc.)
        model: Trained model for this fold
    """
    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    y_val_true: np.ndarray
    y_val_proba: np.ndarray
    metrics: Dict[str, float]
    model: Optional[Any] = None


@dataclass
class CVResults:
    """
    Results from complete cross-validation run.
    
    Attributes:
        fold_results: List of FoldResults for each fold
        mean_metrics: Mean metrics across folds
        std_metrics: Standard deviation of metrics across folds
        best_fold_id: Fold with best validation performance
        best_model: Best model from best fold
    """
    fold_results: List[FoldResults]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    best_fold_id: int
    best_model: Optional[Any] = None

    @property
    def best_auroc(self) -> float:
        """Get best mean AUROC across folds."""
        return self.mean_metrics.get("auroc", 0.0)


class TrainingEngine:
    """
    Orchestrates model training, validation, and hyperparameter optimization.
    
    The training engine manages the full training workflow for the multilabel
    H1N1 flu shot prediction task:
    
    1. **Cross-Validation**: Stratified k-fold splits respecting both vaccine targets
    2. **Model Training**: Train model on each fold with optional early stopping
    3. **Validation**: Evaluate on hold-out validation set, compute ROC AUC
    4. **Hyperparameter Search**: Grid/random/Bayesian search over parameter space
    5. **Threshold Tuning**: Optimize decision threshold on validation set
    6. **Class Imbalance**: Handle imbalance with weights, SMOTE, or threshold
    7. **Metrics Tracking**: Track and aggregate metrics across folds
    8. **Model Selection**: Select best model based on validation performance
    
    Attributes:
        config: TrainingConfig with CV strategy, hyperparameters, search settings
        X: Training features
        y: Training labels
        fold_results: Results from each cross-validation fold
        best_model: Best model selected from CV
        
    Example:
        >>> from src.training import TrainingEngine
        >>> from src.models import ModelFactory
        >>> from src.config import TrainingConfig, ModelConfig
        >>> 
        >>> training_config = TrainingConfig(
        ...     cv_strategy='stratified_kfold',
        ...     hyperparameter_search=True,
        ...     search_strategy='grid'
        ... )
        >>> model_config = ModelConfig(
        ...     model_type='xgboost',
        ...     hyperparameters={'max_depth': [3, 6, 9]}
        ... )
        >>> 
        >>> engine = TrainingEngine(training_config, model_config)
        >>> cv_results = engine.run_cv(X_train, y_train)
        >>> print(f"Best AUROC: {cv_results.best_auroc:.4f}")
        >>> best_model = cv_results.best_model
    """

    def __init__(self, training_config: Dict[str, Any], model_config: Dict[str, Any]):
        """
        Initialize training engine with configuration.
        
        Args:
            training_config: TrainingConfig or dict with CV, SMOTE, early stopping, search settings
            model_config: ModelConfig or dict with model type and hyperparameters
        """
        self.training_config = training_config
        self.model_config = model_config
        self.X = None
        self.y = None
        self.fold_results = []
        self.best_model = None
        self.best_auroc = 0.0

    def run_cv(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        preprocessor: Optional[Any] = None,
        model_factory: Optional[Any] = None,
        return_train_preds: bool = False,
    ) -> CVResults:
        """
        Execute complete cross-validation workflow.
        
        Splits data into k folds, trains model on each fold, validates on
        held-out fold, and aggregates results.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples, 2) for multilabel targets
            preprocessor: PreprocessingPipeline instance for imputation and encoding
            model_factory: ModelFactory for creating models
            return_train_preds: If True, include training predictions in results
            
        Returns:
            CVResults with fold_results, mean/std metrics, best model
            
        Example output:
            >>> cv_results = engine.run_cv(X_train, y_train)
            >>> print(f"Mean AUROC: {cv_results.mean_metrics['auroc']:.4f}")
            >>> print(f"Std AUROC: {cv_results.std_metrics['auroc']:.4f}")
            >>> for fold_result in cv_results.fold_results:
            ...     print(f"Fold {fold_result.fold_id}: {fold_result.metrics}")
        
        Implementation:
            - Create stratified k-fold splits based on combined labels
            - For each fold:
              - Get fold train/val indices
              - Fit preprocessing on training fold only (no data leakage)
              - Train h1n1 model on training fold
              - Train seasonal model on training fold
              - Generate validation predictions for both models
              - Log fold timing and class distributions
            - Aggregate metrics across folds
            - Return CVResults with all fold results
        """
        from src.utils.helpers import create_stratification_column
        from src.models.factory import ModelFactory
        from src.preprocessing import PreprocessingPipeline
        
        # Store training data
        self.X = X
        self.y = y
        
        # Extract h1n1 and seasonal labels
        h1n1_labels = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y[0]
        seasonal_labels = y.iloc[:, 1] if isinstance(y, pd.DataFrame) else y[1]
        
        # Create stratification column: h1n1 + 2*seasonal (4 classes)
        strat_column = create_stratification_column(h1n1_labels, seasonal_labels)
        
        # Get number of folds from config
        n_splits = self.training_config.get("cv_folds", 5) if isinstance(self.training_config, dict) else getattr(self.training_config, "cv_folds", 5)
        random_state = self.training_config.get("random_seed", 42) if isinstance(self.training_config, dict) else getattr(self.training_config, "random_seed", 42)
        
        # Create StratifiedKFold splitter
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Use model factory if provided
        if model_factory is None:
            model_factory = ModelFactory
        
        fold_results = []
        fold_metrics_list = []
        val_indices_list = []
        val_proba_h1n1_list = []
        val_proba_seasonal_list = []
        val_true_h1n1_list = []
        val_true_seasonal_list = []
        
        logger.info(f"Starting {n_splits}-fold stratified cross-validation")
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"H1N1 distribution: {h1n1_labels.sum()} positive, {(1-h1n1_labels).sum()} negative")
        logger.info(f"Seasonal distribution: {seasonal_labels.sum()} positive, {(1-seasonal_labels).sum()} negative")
        
        # Iterate through folds
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, strat_column)):
            fold_start_time = time.time()
            logger.info(f"\n--- Fold {fold_id + 1}/{n_splits} ---")
            
            # Get train and validation splits
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_h1n1 = h1n1_labels.iloc[train_idx]
            y_train_seasonal = seasonal_labels.iloc[train_idx]
            y_val_h1n1 = h1n1_labels.iloc[val_idx]
            y_val_seasonal = seasonal_labels.iloc[val_idx]
            
            # Log fold distributions
            logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
            logger.info(f"Train H1N1: {y_train_h1n1.sum()}/{len(y_train_h1n1)} positive")
            logger.info(f"Train Seasonal: {y_train_seasonal.sum()}/{len(y_train_seasonal)} positive")
            logger.info(f"Val H1N1: {y_val_h1n1.sum()}/{len(y_val_h1n1)} positive")
            logger.info(f"Val Seasonal: {y_val_seasonal.sum()}/{len(y_val_seasonal)} positive")
            
            # Step 1: Fit preprocessing on training fold only (no data leakage)
            if preprocessor is not None:
                if isinstance(preprocessor, dict):
                    # If preprocessor is config, create instance
                    from src.preprocessing import PreprocessingPipeline
                    from src.config import ImputationConfig, EncodingConfig
                    impute_cfg = preprocessor.get("imputation", {})
                    encode_cfg = preprocessor.get("encoding", {})
                    pipeline = PreprocessingPipeline(impute_cfg, encode_cfg)
                else:
                    # Clone preprocessor for this fold
                    from copy import deepcopy
                    pipeline = deepcopy(preprocessor)
                
                # Fit preprocessing on training fold only
                X_train_processed = pipeline.fit_transform(X_train)
                X_val_processed = pipeline.transform(X_val)
            else:
                X_train_processed = X_train
                X_val_processed = X_val
            
            # Step 2: Create and train h1n1 model
            h1n1_config = self.model_config
            h1n1_model = model_factory.create_model(h1n1_config)
            h1n1_model.fit(X_train_processed, y_train_h1n1)
            logger.info(f"H1N1 model trained: {h1n1_model.get_model_name()}")
            
            # Step 3: Create and train seasonal model
            seasonal_config = self.model_config
            seasonal_model = model_factory.create_model(seasonal_config)
            seasonal_model.fit(X_train_processed, y_train_seasonal)
            logger.info(f"Seasonal model trained: {seasonal_model.get_model_name()}")
            
            # Step 4: Generate validation predictions
            y_val_proba_h1n1 = h1n1_model.predict_proba(X_val_processed)[:, 1]
            y_val_proba_seasonal = seasonal_model.predict_proba(X_val_processed)[:, 1]
            
            # Collect validation data for later aggregation
            val_indices_list.append(val_idx)
            val_proba_h1n1_list.append(y_val_proba_h1n1)
            val_proba_seasonal_list.append(y_val_proba_seasonal)
            val_true_h1n1_list.append(y_val_h1n1.values)
            val_true_seasonal_list.append(y_val_seasonal.values)
            
            # Step 5: Compute validation metrics
            fold_metrics = self.compute_fold_metrics_dual(
                y_val_h1n1.values, y_val_proba_h1n1,
                y_val_seasonal.values, y_val_proba_seasonal
            )
            fold_metrics_list.append(fold_metrics)
            
            # Log fold metrics
            logger.info(f"H1N1 AUROC: {fold_metrics['auroc_h1n1']:.4f}")
            logger.info(f"Seasonal AUROC: {fold_metrics['auroc_seasonal']:.4f}")
            logger.info(f"Mean AUROC: {fold_metrics['auroc_mean']:.4f}")
            
            fold_time = time.time() - fold_start_time
            logger.info(f"Fold time: {fold_time:.2f}s")
            
            # Store FoldResults
            fold_result = FoldResults(
                fold_id=fold_id,
                train_indices=train_idx,
                val_indices=val_idx,
                y_val_true=np.column_stack([y_val_h1n1.values, y_val_seasonal.values]),
                y_val_proba=np.column_stack([y_val_proba_h1n1, y_val_proba_seasonal]),
                metrics=fold_metrics,
                model=h1n1_model  # Store h1n1 model as representative
            )
            fold_results.append(fold_result)
        
        # Step 6: Aggregate metrics across folds
        mean_metrics, std_metrics = self.aggregate_fold_metrics(fold_metrics_list)
        
        # Find best fold (by mean AUROC)
        best_fold_id = int(np.argmax([m['auroc_mean'] for m in fold_metrics_list]))
        best_model = fold_results[best_fold_id].model
        self.best_model = best_model
        self.best_auroc = mean_metrics['auroc_mean']
        
        logger.info(f"\n=== Cross-Validation Results ===")
        logger.info(f"Best fold: {best_fold_id + 1}")
        logger.info(f"Mean AUROC: {mean_metrics['auroc_mean']:.4f} ± {std_metrics['auroc_mean']:.4f}")
        logger.info(f"Mean H1N1 AUROC: {mean_metrics['auroc_h1n1']:.4f} ± {std_metrics['auroc_h1n1']:.4f}")
        logger.info(f"Mean Seasonal AUROC: {mean_metrics['auroc_seasonal']:.4f} ± {std_metrics['auroc_seasonal']:.4f}")
        
        # Create CVResults
        cv_results = CVResults(
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            best_fold_id=best_fold_id,
            best_model=best_model
        )
        
        return cv_results

    def hyperparameter_search(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        cv_folds: int = 3,
        preprocessor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Search over hyperparameter space using configured strategy.
        
        Performs grid, random, or Bayesian hyperparameter search with inner
        cross-validation to find best hyperparameter combination.
        
        Args:
            X: Training features
            y: Training labels
            param_grid: Dictionary of parameter names to lists of values
                       (for grid/random search) or distributions (for Bayesian)
            cv_folds: Number of folds for inner cross-validation
            preprocessor: Optional PreprocessingPipeline for imputation and encoding
            
        Returns:
            Dictionary with:
            - 'best_params': Best hyperparameters found
            - 'best_score': Best mean CV score
            - 'results': List of parameter combinations and scores
            - 'search_time': Total search time in seconds
            
        Raises:
            ValueError: If search_strategy not recognized
            
        Example:
            >>> param_grid = {
            ...     'max_depth': [3, 6, 9],
            ...     'learning_rate': [0.01, 0.1, 0.3],
            ...     'n_estimators': [50, 100, 200],
            ... }
            >>> results = engine.hyperparameter_search(X, y, param_grid)
            >>> print(f"Best params: {results['best_params']}")
            >>> print(f"Best score: {results['best_score']:.4f}")
        
        Implementation notes (strategy-dependent):
            - Grid Search:
            -   TODO: Generate all combinations of parameters
            -   TODO: For each combination, run CV and record score
            -   TODO: Return results sorted by score
            - Random Search:
            -   TODO: Randomly sample parameter combinations
            -   TODO: For each, run CV and record score
            -   TODO: Return top-K results
            - Bayesian Search:
            -   TODO: Use Gaussian process to model parameter → score relationship
            -   TODO: Iteratively suggest promising parameter combinations
            -   TODO: Explore-exploit tradeoff to balance search
            -   TODO: Return results
        """
        logger.info(f"\n=== Hyperparameter Search ({self.training_config['search_strategy'].upper()}) ===")
        search_start = time.time()
        
        search_strategy = self.training_config.get('search_strategy', 'grid')
        
        if search_strategy == 'grid':
            results = self._grid_search(X, y, param_grid, cv_folds, preprocessor)
        elif search_strategy == 'random':
            results = self._random_search(X, y, param_grid, cv_folds, preprocessor)
        elif search_strategy == 'bayesian':
            results = self._bayesian_search(X, y, param_grid, cv_folds, preprocessor)
        else:
            raise ValueError(f"Unknown search_strategy: {search_strategy}. Must be 'grid', 'random', or 'bayesian'")
        
        search_time = time.time() - search_start
        results['search_time'] = search_time
        
        logger.info(f"Best hyperparameters: {results['best_params']}")
        logger.info(f"Best AUROC: {results['best_score']:.4f}")
        logger.info(f"Search time: {search_time:.1f}s")
        
        return results

    def _grid_search(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        cv_folds: int,
        preprocessor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Grid search: exhaustively evaluate all parameter combinations.
        
        Args:
            X: Training features
            y: Training labels
            param_grid: Dict of parameter names to lists of values
            cv_folds: Number of CV folds for evaluation
            
        Returns:
            Dict with best_params, best_score, and all results
        """
        logger.info(f"Grid search over {len(param_grid)} parameters")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Total combinations to evaluate: {len(param_combinations)}")
        
        results = []
        best_score = -np.inf
        best_params = None
        
        for i, param_combo in enumerate(param_combinations):
            # Create param dict from combination
            params = dict(zip(param_names, param_combo))
            
            logger.info(f"[{i+1}/{len(param_combinations)}] Evaluating: {params}")
            
            # Update model config with new hyperparameters
            original_hyperparams = self.model_config.get('hyperparameters', {}).copy()
            self.model_config['hyperparameters'].update(params)
            
            try:
                # Run CV with these hyperparameters
                from src.models.factory import ModelFactory
                
                # Inner CV loop to evaluate this parameter set
                cv_scores = []
                skf = StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=self.training_config.get('random_seed', 42)
                )
                
                # For multilabel, we need to stratify based on combined labels
                # Create a combined label for stratification
                y_combined = y.iloc[:, 0].astype(str) + '_' + y.iloc[:, 1].astype(str)
                
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_combined)):
                    X_train_fold = X.iloc[train_idx].reset_index(drop=True)
                    y_train_fold = y.iloc[train_idx].reset_index(drop=True)
                    X_val_fold = X.iloc[val_idx].reset_index(drop=True)
                    y_val_fold = y.iloc[val_idx].reset_index(drop=True)
                    
                    # Create config for this trial
                    model_config = {
                        'model_type': self.model_config.get('model_type', 'logistic_regression'),
                        'hyperparameters': self.model_config.get('hyperparameters', {})
                    }
                    
                    # Train models for both vaccines
                    model_h1n1 = ModelFactory.create_model(model_config)
                    model_seasonal = ModelFactory.create_model(model_config)
                    
                    model_h1n1.fit(X_train_fold, y_train_fold.iloc[:, 0])
                    model_seasonal.fit(X_train_fold, y_train_fold.iloc[:, 1])
                    
                    # Predict on validation fold
                    y_val_proba_h1n1 = model_h1n1.predict_proba(X_val_fold)
                    y_val_proba_seasonal = model_seasonal.predict_proba(X_val_fold)
                    
                    # Handle predictions (could be 1D or 2D)
                    if len(y_val_proba_h1n1.shape) == 2:
                        y_val_proba_h1n1 = y_val_proba_h1n1[:, 1]
                    if len(y_val_proba_seasonal.shape) == 2:
                        y_val_proba_seasonal = y_val_proba_seasonal[:, 1]
                    
                    # Compute AUROC
                    auroc_h1n1 = roc_auc_score(y_val_fold.iloc[:, 0], y_val_proba_h1n1)
                    auroc_seasonal = roc_auc_score(y_val_fold.iloc[:, 1], y_val_proba_seasonal)
                    mean_auroc = (auroc_h1n1 + auroc_seasonal) / 2.0
                    
                    cv_scores.append(mean_auroc)
                
                # Compute mean CV score
                mean_cv_score = np.mean(cv_scores)
                
                results.append({
                    'params': params,
                    'score': mean_cv_score,
                    'cv_scores': cv_scores
                })
                
                logger.info(f"  Score: {mean_cv_score:.4f} (std: {np.std(cv_scores):.4f})")
                
                # Update best if needed
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_params = params.copy()
                    
            finally:
                # Restore original hyperparameters
                self.model_config['hyperparameters'] = original_hyperparams
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'results': results,
        }

    def _random_search(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        cv_folds: int,
        preprocessor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Random search: randomly sample parameter combinations.
        
        Args:
            X: Training features
            y: Training labels
            param_grid: Dict of parameter names to lists of values
            cv_folds: Number of CV folds for evaluation
            
        Returns:
            Dict with best_params, best_score, and all results
        """
        search_params = self.training_config.get('search_params', {})
        n_trials = search_params.get('n_trials', 20)
        random_seed = self.training_config.get('random_seed', 42)
        
        logger.info(f"Random search with {n_trials} trials")
        
        rng = np.random.RandomState(random_seed)
        param_names = list(param_grid.keys())
        
        results = []
        best_score = -np.inf
        best_params = None
        
        for trial in range(n_trials):
            # Randomly sample parameters
            params = {}
            for param_name in param_names:
                param_values = param_grid[param_name]
                params[param_name] = param_values[rng.randint(0, len(param_values))]
            
            logger.info(f"[Trial {trial+1}/{n_trials}] Evaluating: {params}")
            
            # Update model config with new hyperparameters
            original_hyperparams = self.model_config.get('hyperparameters', {}).copy()
            self.model_config['hyperparameters'].update(params)
            
            try:
                # Run CV with these hyperparameters
                from src.models.factory import ModelFactory
                
                # Inner CV loop to evaluate this parameter set
                cv_scores = []
                skf = StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=random_seed
                )
                
                # For multilabel, we need to stratify based on combined labels
                y_combined = y.iloc[:, 0].astype(str) + '_' + y.iloc[:, 1].astype(str)
                
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_combined)):
                    X_train_fold = X.iloc[train_idx].reset_index(drop=True)
                    y_train_fold = y.iloc[train_idx].reset_index(drop=True)
                    X_val_fold = X.iloc[val_idx].reset_index(drop=True)
                    y_val_fold = y.iloc[val_idx].reset_index(drop=True)
                    
                    # Create config for this trial
                    model_config = {
                        'model_type': self.model_config.get('model_type', 'logistic_regression'),
                        'hyperparameters': self.model_config.get('hyperparameters', {})
                    }
                    
                    # Train models for both vaccines
                    model_h1n1 = ModelFactory.create_model(model_config)
                    model_seasonal = ModelFactory.create_model(model_config)
                    
                    model_h1n1.fit(X_train_fold, y_train_fold.iloc[:, 0])
                    model_seasonal.fit(X_train_fold, y_train_fold.iloc[:, 1])
                    
                    # Predict on validation fold
                    y_val_proba_h1n1 = model_h1n1.predict_proba(X_val_fold)
                    y_val_proba_seasonal = model_seasonal.predict_proba(X_val_fold)
                    
                    # Handle predictions (could be 1D or 2D)
                    if len(y_val_proba_h1n1.shape) == 2:
                        y_val_proba_h1n1 = y_val_proba_h1n1[:, 1]
                    if len(y_val_proba_seasonal.shape) == 2:
                        y_val_proba_seasonal = y_val_proba_seasonal[:, 1]
                    
                    # Compute AUROC
                    auroc_h1n1 = roc_auc_score(y_val_fold.iloc[:, 0], y_val_proba_h1n1)
                    auroc_seasonal = roc_auc_score(y_val_fold.iloc[:, 1], y_val_proba_seasonal)
                    mean_auroc = (auroc_h1n1 + auroc_seasonal) / 2.0
                    
                    cv_scores.append(mean_auroc)
                
                # Compute mean CV score
                mean_cv_score = np.mean(cv_scores)
                
                results.append({
                    'params': params,
                    'score': mean_cv_score,
                    'cv_scores': cv_scores
                })
                
                logger.info(f"  Score: {mean_cv_score:.4f} (std: {np.std(cv_scores):.4f})")
                
                # Update best if needed
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_params = params.copy()
                    
            finally:
                # Restore original hyperparameters
                self.model_config['hyperparameters'] = original_hyperparams
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'results': results,
        }

    def _bayesian_search(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        cv_folds: int,
        preprocessor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Bayesian search using Optuna for hyperparameter optimization.
        
        Args:
            X: Training features
            y: Training labels
            param_grid: Dict of parameter names to lists of values
            cv_folds: Number of CV folds for evaluation
            
        Returns:
            Dict with best_params, best_score, and all results
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            logger.warning("Optuna not installed. Falling back to random search.")
            return self._random_search(X, y, param_grid, cv_folds)
        
        search_params = self.training_config.get('search_params', {})
        n_trials = search_params.get('n_trials', 40)
        timeout = search_params.get('timeout', None)
        n_jobs = search_params.get('n_jobs', 1)
        random_seed = self.training_config.get('random_seed', 42)
        
        logger.info(f"Bayesian search (Optuna) with {n_trials} trials")
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Create objective function
        def objective(trial):
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values[0], int):
                    # Integer parameter
                    params[param_name] = trial.suggest_int(
                        param_name,
                        min(param_values),
                        max(param_values)
                    )
                elif isinstance(param_values[0], float):
                    # Float parameter
                    params[param_name] = trial.suggest_float(
                        param_name,
                        min(param_values),
                        max(param_values)
                    )
                else:
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_values
                    )
            
            # Update model config with suggested hyperparameters
            original_hyperparams = self.model_config.get('hyperparameters', {}).copy()
            self.model_config['hyperparameters'].update(params)
            
            try:
                # Run CV with these hyperparameters
                from src.models.factory import ModelFactory
                
                # Inner CV loop to evaluate this parameter set
                cv_scores = []
                skf = StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=random_seed
                )
                
                # For multilabel, we need to stratify based on combined labels
                y_combined = y.iloc[:, 0].astype(str) + '_' + y.iloc[:, 1].astype(str)
                
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_combined)):
                    X_train_fold = X.iloc[train_idx].reset_index(drop=True)
                    y_train_fold = y.iloc[train_idx].reset_index(drop=True)
                    X_val_fold = X.iloc[val_idx].reset_index(drop=True)
                    y_val_fold = y.iloc[val_idx].reset_index(drop=True)
                    
                    # Apply preprocessing if provided (bayesian search)
                    if preprocessor is not None:
                        from copy import deepcopy
                        pipeline = deepcopy(preprocessor)
                        X_train_fold = pipeline.fit_transform(X_train_fold)
                        X_val_fold = pipeline.transform(X_val_fold)
                    
                    # Create config for this trial
                    model_config = {
                        'model_type': self.model_config.get('model_type', 'logistic_regression'),
                        'hyperparameters': self.model_config.get('hyperparameters', {})
                    }
                    
                    # Train models for both vaccines
                    model_h1n1 = ModelFactory.create_model(model_config)
                    model_seasonal = ModelFactory.create_model(model_config)
                    
                    model_h1n1.fit(X_train_fold, y_train_fold.iloc[:, 0])
                    model_seasonal.fit(X_train_fold, y_train_fold.iloc[:, 1])
                    
                    # Predict on validation fold
                    y_val_proba_h1n1 = model_h1n1.predict_proba(X_val_fold)
                    y_val_proba_seasonal = model_seasonal.predict_proba(X_val_fold)
                    
                    # Handle predictions (could be 1D or 2D)
                    if len(y_val_proba_h1n1.shape) == 2:
                        y_val_proba_h1n1 = y_val_proba_h1n1[:, 1]
                    if len(y_val_proba_seasonal.shape) == 2:
                        y_val_proba_seasonal = y_val_proba_seasonal[:, 1]
                    
                    # Compute AUROC
                    auroc_h1n1 = roc_auc_score(y_val_fold.iloc[:, 0], y_val_proba_h1n1)
                    auroc_seasonal = roc_auc_score(y_val_fold.iloc[:, 1], y_val_proba_seasonal)
                    mean_auroc = (auroc_h1n1 + auroc_seasonal) / 2.0
                    
                    cv_scores.append(mean_auroc)
                
                # Compute mean CV score for bayesian search
                mean_cv_score = np.mean(cv_scores)
                return mean_cv_score
                
            finally:
                # Restore original hyperparameters
                self.model_config['hyperparameters'] = original_hyperparams
        
        # Create Optuna study
        sampler = TPESampler(seed=random_seed)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        # Run optimization
        study.optimize(
            lambda trial: float(objective(trial)),  # Type: convert to float
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1,  # n_jobs in Optuna doesn't work well with our nested CV
            show_progress_bar=False
        )
        
        # Extract best trial info
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        # Collect all results for consistency with other search methods
        results = []
        for trial in study.trials:
            results.append({
                'params': trial.params,
                'score': trial.value if trial.value is not None else -np.inf,
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'results': results,
        }

    def get_fold_predictions(
        self,
        X_test: pd.DataFrame,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get test predictions by averaging across all folds (ensemble).
        
        For each fold model, predict on test set and average the probability
        predictions across models. This reduces variance and often improves
        performance compared to single model.
        
        Args:
            X_test: Test features
            return_std: If True, also return standard deviation of predictions
            
        Returns:
            If return_std=False: Array of shape (n_test, 2) with ensemble probabilities
            If return_std=True: Tuple of (proba, std) where std is (n_test, 2)
            
        Example:
            >>> cv_results = engine.run_cv(X_train, y_train)
            >>> test_proba = engine.get_fold_predictions(X_test)
            >>> test_proba_std = engine.get_fold_predictions(X_test, return_std=True)
        
        Implementation notes:
            - TODO: Ensure models from all folds are stored
            - TODO: For each fold model:
            -   TODO: Generate predictions on X_test
            -   TODO: Stack predictions
            - TODO: Compute mean across folds
            - TODO: If return_std, compute standard deviation
            - TODO: Return averaged predictions (and std if requested)
        """
        if not self.fold_results:
            raise ValueError("No fold results available. Run run_cv() first.")
        
        all_proba = []
        
        # Collect predictions from all fold models
        for fold_result in self.fold_results:
            if fold_result.model is None:
                logger.warning(f"Fold {fold_result.fold_id} has no model stored")
                continue
            
            # Predict with this fold's model
            proba = fold_result.model.predict_proba(X_test)
            all_proba.append(proba)
        
        if not all_proba:
            raise ValueError("No fold models available for prediction")
        
        # Stack predictions from all folds
        all_proba = np.array(all_proba)  # Shape: (n_folds, n_test, 2)
        
        # Compute mean across folds
        mean_proba = np.mean(all_proba, axis=0)
        
        if return_std:
            std_proba = np.std(all_proba, axis=0)
            return mean_proba, std_proba
        else:
            return mean_proba

    def get_best_model(self) -> Optional[Any]:
        """
        Get the best model from training.
        
        Returns the model from the best-performing cross-validation fold.
        
        Returns:
            Best model fitted on its training fold
            
        Raises:
            ValueError: If CV not run yet
        """
        if self.best_model is None:
            raise ValueError("No model trained yet. Run run_cv() first.")
        return self.best_model

    def apply_class_weights(
        self,
        y: pd.DataFrame,
    ) -> Optional[Dict[int, float]]:
        """
        Compute class weights to handle imbalance.
        
        For imbalanced classes, compute weights inversely proportional to
        class frequencies. Minority class gets higher weight.
        
        Args:
            y: Labels (n_samples, 2) for multilabel targets
            
        Returns:
            Dictionary {0: weight_0, 1: weight_1}
            
        Example:
            >>> weights = engine.apply_class_weights(y_train)
            >>> # Pass to model as class_weight parameter
        
        Implementation notes:
            - TODO: For each target (h1n1, seasonal):
            -   TODO: Compute class distribution
            -   TODO: Compute inverse weights: 1 / frequency
            -   TODO: Normalize so weights sum to 1 or some constant
            - TODO: Average weights across targets (or use per-target)
            - TODO: Return as dict
        """
        strategy = self.training_config.get('class_weight_strategy', 'balanced')
        
        if strategy == 'none':
            logger.info("No class weights applied")
            return None
        
        if strategy != 'balanced':
            logger.warning(f"Unknown class_weight_strategy: {strategy}. Using 'balanced'.")
            strategy = 'balanced'
        
        logger.info(f"Computing class weights (strategy: {strategy})")
        
        # Compute weights for each target separately
        weights_h1n1 = self._compute_weights_for_target(y.iloc[:, 0])
        weights_seasonal = self._compute_weights_for_target(y.iloc[:, 1])
        
        # Average weights across targets
        weights = {
            0: (weights_h1n1[0] + weights_seasonal[0]) / 2.0,
            1: (weights_h1n1[1] + weights_seasonal[1]) / 2.0,
        }
        
        logger.info(f"Class weights: {weights}")
        return weights

    def _compute_weights_for_target(self, y: pd.Series) -> Dict[int, float]:
        """
        Compute class weights for a single target (binary classification).
        
        Args:
            y: Target labels (0 or 1)
            
        Returns:
            Dict with weights for class 0 and 1
        """
        # Count class occurrences
        n_class_0 = (y == 0).sum()
        n_class_1 = (y == 1).sum()
        total = len(y)
        
        # Inverse of class frequency
        weight_0 = total / (2 * n_class_0) if n_class_0 > 0 else 1.0
        weight_1 = total / (2 * n_class_1) if n_class_1 > 0 else 1.0
        
        return {0: weight_0, 1: weight_1}

    def apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sampling_strategy: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply SMOTE to handle class imbalance via oversampling.
        
        Synthetic Minority Over-sampling Technique: Creates synthetic samples
        of minority class by interpolating between nearest neighbors. Can
        improve model performance on imbalanced data.
        
        Args:
            X: Training features
            y: Training labels (n_samples, 2)
            sampling_strategy: Ratio of minority to majority class after SMOTE (0-1)
            
        Returns:
            Tuple of (X_smote, y_smote) with synthetic samples added
            
        Example:
            >>> X_smote, y_smote = engine.apply_smote(X_train, y_train)
            >>> model.fit(X_smote, y_smote)
        
        Implementation notes:
            - TODO: Check if imbalanced-learn package is available
            - TODO: For each target (h1n1, seasonal):
            -   TODO: Apply SMOTE with sampling_strategy
            -   TODO: Update both X and y
            - TODO: Return augmented X, y
            - TODO: Handle multilabel case (apply SMOTE per target or jointly?)
        """
        use_smote = self.training_config.get('use_smote', False)
        
        if not use_smote:
            logger.info("SMOTE not enabled")
            return X, y
        
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
            return X, y
        
        logger.info(f"Applying SMOTE (sampling_strategy={sampling_strategy})")
        
        # For multilabel case, apply SMOTE to both targets using combined labels
        # Create combined label (0,0), (0,1), (1,0), (1,1)
        y_combined = y.iloc[:, 0].astype(str) + '_' + y.iloc[:, 1].astype(str)
        
        # Apply SMOTE
        # sampling_strategy should be 'auto' or a float between 0 and 1
        strategy_param = 'auto' if sampling_strategy == 1.0 else sampling_strategy
        
        smote = SMOTE(
            sampling_strategy=strategy_param,  # type: ignore
            random_state=self.training_config.get('random_seed', 42),
            k_neighbors=5
        )
        
        try:
            result = smote.fit_resample(X, y_combined)
            X_resampled = result[0]
            y_combined_resampled = result[1]
            
            # Reconstruct separate target columns from combined labels
            y_h1n1 = []
            y_seasonal = []
            for label_val in y_combined_resampled:
                label_str = str(label_val)  # Type: ensure it's a string
                h1n1, seasonal = label_str.split('_')
                y_h1n1.append(int(h1n1))
                y_seasonal.append(int(seasonal))
            
            # Create new y dataframe
            y_resampled = pd.DataFrame({
                y.columns[0]: y_h1n1,
                y.columns[1]: y_seasonal,
            })
            
            logger.info(f"SMOTE complete. New shape: X {X_resampled.shape}, y {y_resampled.shape}")
            return pd.DataFrame(X_resampled, columns=X.columns), y_resampled
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y

    def tune_threshold(
        self,
        y_val_true: np.ndarray,
        y_val_proba: np.ndarray,
        metric: str = "auc",
    ) -> float:
        """
        Find optimal classification threshold on validation set.
        
        By default, sklearn uses threshold 0.5. But optimal threshold may differ
        based on metric and class distribution. This searches for threshold
        that maximizes the specified metric.
        
        Args:
            y_val_true: True validation labels (n_val,)
            y_val_proba: Validation probabilities for positive class (n_val,)
            metric: Metric to optimize ('auc', 'f1', 'precision', 'recall', etc.)
            
        Returns:
            Optimal threshold value
            
        Example:
            >>> threshold = engine.tune_threshold(y_val_true, y_val_proba[:, 1])
            >>> y_pred = (y_val_proba[:, 1] > threshold).astype(int)
        
        Implementation notes:
            - TODO: Generate range of thresholds (e.g., 0.1 to 0.9 in 0.05 steps)
            - TODO: For each threshold:
            -   TODO: Convert probabilities to hard predictions
            -   TODO: Compute specified metric
            - TODO: Find threshold maximizing metric
            - TODO: Return optimal threshold
        """
        from sklearn.metrics import (
            roc_auc_score,
            f1_score,
            precision_score,
            recall_score,
            accuracy_score,
        )
        
        # Generate candidate thresholds
        thresholds = np.linspace(0.0, 1.0, 101)
        best_threshold = 0.5
        best_score = -np.inf
        
        metric_lower = metric.lower()
        
        for threshold in thresholds:
            # Convert probabilities to binary predictions
            y_pred = (y_val_proba > threshold).astype(int)
            
            # Compute specified metric
            if metric_lower == 'auc':
                # For AUC, we use raw probabilities
                score = roc_auc_score(y_val_true, y_val_proba)
            elif metric_lower == 'f1':
                score = f1_score(y_val_true, y_pred, zero_division=0)
            elif metric_lower == 'precision':
                score = precision_score(y_val_true, y_pred, zero_division=0)
            elif metric_lower == 'recall':
                score = recall_score(y_val_true, y_pred, zero_division=0)
            elif metric_lower == 'accuracy':
                score = accuracy_score(y_val_true, y_pred)
            else:
                logger.warning(f"Unknown metric: {metric}. Using accuracy.")
                score = accuracy_score(y_val_true, y_pred)
            
            # Update best if needed
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Clamp threshold to valid range
        best_threshold = np.clip(best_threshold, 0.0, 1.0)
        
        return best_threshold

    def apply_threshold_tuning(
        self,
        fold_results: List[FoldResults],
        metric: str = "auc",
    ) -> Dict[str, float]:
        """
        Find optimal classification thresholds per vaccine using CV fold predictions.
        
        Takes accumulated validation predictions from all CV folds and finds
        the optimal threshold for each vaccine that maximizes the specified metric.
        
        Args:
            fold_results: List of FoldResults from cross-validation
            metric: Metric to optimize ('auc', 'f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Dictionary with thresholds for each vaccine:
            {'h1n1_vaccine': float, 'seasonal_vaccine': float}
            
        Example:
            >>> cv_results = engine.run_cv(X, y)
            >>> thresholds = engine.apply_threshold_tuning(cv_results.fold_results)
            >>> print(f"H1N1 threshold: {thresholds['h1n1_vaccine']:.3f}")
            >>> print(f"Seasonal threshold: {thresholds['seasonal_vaccine']:.3f}")
        """
        logger.info(f"\n=== Threshold Tuning ({metric.upper()}) ===")
        
        # Collect all validation predictions and labels across folds
        y_val_all = np.vstack([fold.y_val_true for fold in fold_results])
        y_val_proba_all = np.vstack([fold.y_val_proba for fold in fold_results])
        
        # For each vaccine, find optimal threshold
        thresholds = {}
        
        # H1N1 vaccine (column 0)
        y_h1n1_true = y_val_all[:, 0]
        y_h1n1_proba = y_val_proba_all[:, 0]
        threshold_h1n1 = self.tune_threshold(y_h1n1_true, y_h1n1_proba, metric)
        thresholds['h1n1_vaccine'] = threshold_h1n1
        
        logger.info(f"H1N1 vaccine threshold: {threshold_h1n1:.4f}")
        
        # Seasonal vaccine (column 1)
        y_seasonal_true = y_val_all[:, 1]
        y_seasonal_proba = y_val_proba_all[:, 1]
        threshold_seasonal = self.tune_threshold(y_seasonal_true, y_seasonal_proba, metric)
        thresholds['seasonal_vaccine'] = threshold_seasonal
        
        logger.info(f"Seasonal vaccine threshold: {threshold_seasonal:.4f}")
        
        return thresholds

    def compute_fold_metrics_dual(
        self,
        y_true_h1n1: np.ndarray,
        y_proba_h1n1: np.ndarray,
        y_true_seasonal: np.ndarray,
        y_proba_seasonal: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for a fold with two independent targets.
        
        Args:
            y_true_h1n1: True H1N1 labels (n_samples,) with values 0 or 1
            y_proba_h1n1: Probability predictions for H1N1 (n_samples,)
            y_true_seasonal: True seasonal labels (n_samples,) with values 0 or 1
            y_proba_seasonal: Probability predictions for seasonal (n_samples,)
            
        Returns:
            Dictionary with metrics for both targets and mean
            
        Example:
            >>> metrics = engine.compute_fold_metrics_dual(y_h1n1_true, y_h1n1_proba, 
            ...                                            y_seas_true, y_seas_proba)
            >>> print(f"H1N1 AUROC: {metrics['auroc_h1n1']:.4f}")
            >>> print(f"Seasonal AUROC: {metrics['auroc_seasonal']:.4f}")
        """
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss
        
        # Compute AUROC for both targets
        auroc_h1n1 = roc_auc_score(y_true_h1n1, y_proba_h1n1)
        auroc_seasonal = roc_auc_score(y_true_seasonal, y_proba_seasonal)
        auroc_mean = (auroc_h1n1 + auroc_seasonal) / 2.0
        
        # Compute accuracy (threshold 0.5)
        pred_h1n1 = (y_proba_h1n1 > 0.5).astype(int)
        pred_seasonal = (y_proba_seasonal > 0.5).astype(int)
        accuracy_h1n1 = accuracy_score(y_true_h1n1, pred_h1n1)
        accuracy_seasonal = accuracy_score(y_true_seasonal, pred_seasonal)
        accuracy_mean = (accuracy_h1n1 + accuracy_seasonal) / 2.0
        
        # Compute F1 score (macro)
        f1_h1n1 = f1_score(y_true_h1n1, pred_h1n1, zero_division=0)
        f1_seasonal = f1_score(y_true_seasonal, pred_seasonal, zero_division=0)
        f1_mean = (f1_h1n1 + f1_seasonal) / 2.0
        
        # Compute Brier score
        brier_h1n1 = brier_score_loss(y_true_h1n1, y_proba_h1n1)
        brier_seasonal = brier_score_loss(y_true_seasonal, y_proba_seasonal)
        brier_mean = (brier_h1n1 + brier_seasonal) / 2.0
        
        return {
            'auroc_h1n1': float(auroc_h1n1),
            'auroc_seasonal': float(auroc_seasonal),
            'auroc_mean': float(auroc_mean),
            'accuracy_h1n1': float(accuracy_h1n1),
            'accuracy_seasonal': float(accuracy_seasonal),
            'accuracy_mean': float(accuracy_mean),
            'f1_h1n1': float(f1_h1n1),
            'f1_seasonal': float(f1_seasonal),
            'f1_mean': float(f1_mean),
            'brier_h1n1': float(brier_h1n1),
            'brier_seasonal': float(brier_seasonal),
            'brier_mean': float(brier_mean),
        }

    def aggregate_fold_metrics(
        self,
        fold_metrics_list: List[Dict[str, float]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Aggregate metrics from multiple folds into mean and std.
        
        Args:
            fold_metrics_list: List of metric dicts from each fold
            
        Returns:
            Tuple of (mean_metrics, std_metrics) dicts
            
        Example:
            >>> metrics_list = [fold1_metrics, fold2_metrics, ...]
            >>> mean_m, std_m = engine.aggregate_fold_metrics(metrics_list)
            >>> print(f"AUROC: {mean_m['auroc_mean']:.4f} ± {std_m['auroc_mean']:.4f}")
        """
        if not fold_metrics_list:
            raise ValueError("No fold metrics to aggregate")
        
        mean_metrics = {}
        std_metrics = {}
        
        # Get all metric keys from first fold
        for key in fold_metrics_list[0].keys():
            values = np.array([m[key] for m in fold_metrics_list])
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))
        
        return mean_metrics, std_metrics

    def save_cv_results(self, cv_results: CVResults, output_path: str) -> None:
        """
        Save cross-validation results to file.
        
        Persists CVResults for later analysis or comparison.
        
        Args:
            cv_results: CVResults object from run_cv()
            output_path: Path to save results (CSV or pickle)
        """
        # TODO: Implement
        pass

    def load_best_model_from_cv(self, cv_results: CVResults) -> Any:
        """
        Extract and cache best model from CV results.
        
        Args:
            cv_results: CVResults from run_cv()
            
        Returns:
            Best model
        """
        self.best_model = cv_results.best_model
        return self.best_model
