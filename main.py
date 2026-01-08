"""
ML Pipeline Orchestrator for Flu Shot Prediction.

This module demonstrates the data flow through all components of the ML pipeline:

    Data Loading
        ↓
    Imputation
        ↓
    Feature Encoding
        ↓
    Model Training (Cross-Validation)
        ↓
    Probability Calibration
        ↓
    Evaluation & Metrics
        ↓
    Experiment Tracking
        ↓
    Test Set Predictions

Reference: SYSTEM_DESIGN.md - Component 10: Integration & Orchestration

Architecture:
- src.config: Configuration system with PipelineConfig dataclass
- src.data: Data loading and splitting (CSVDataLoader)
- src.preprocessing: Imputation and encoding strategies
- src.models: ML model implementations and factory
- src.training: Training orchestration with cross-validation
- src.calibration: Probability calibration methods
- src.evaluation: Metrics computation and visualization
- src.tracking: Experiment tracking and logging
- src.prediction: Test set inference and submission formatting
- src.utils: Helper functions for logging, validation, metrics, plots

Usage:
    python main.py --config examples/config_baseline.yaml
    python main.py --config examples/config_xgboost.yaml --run-name "experiment_001"

Example Configuration:
    See examples/config_baseline.yaml for a basic logistic regression pipeline
    See examples/config_xgboost.yaml for an advanced XGBoost pipeline
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# Lazy imports: Import heavy dependencies only when pipeline runs
# This allows --help to work without all dependencies installed
try:
    from src.config import PipelineConfig
    from src.data.loader import CSVDataLoader
    from src.preprocessing.imputation import ImputationStrategy
    from src.preprocessing.encoding import FeatureEncoder
    from src.models.factory import ModelFactory
    from src.training.engine import TrainingEngine
    from src.calibration.calibrator import create_calibrator
    from src.evaluation.metrics import Evaluator
    from src.tracking.logger import CSVExperimentLogger
    from src.prediction.predictor import DefaultPredictionEngine
    from src.utils.logging import setup_logging, get_logger
    from src.utils.helpers import seed_all_random_states
    
    # Imports that might fail if optional dependencies missing
    try:
        from src.evaluation.plots import plot_roc_curves, plot_calibration_curve, plot_feature_importance
    except ImportError:
        plot_roc_curves = None
        plot_calibration_curve = None
        plot_feature_importance = None
    
except ImportError as e:
    # Allow argument parsing to work even if imports fail
    pass

# Configure logging
try:
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration from file or use default.
    
    Parameters:
        config_path (Optional[str]): Path to YAML config file.
            If None, uses default configuration.
    
    Returns:
        PipelineConfig: Configuration object for pipeline
    
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML syntax is invalid
        ValueError: If config is missing required fields
    
    Implementation notes:
        - If config_path provided, load from YAML file with error handling
        - If config_path is None, create default PipelineConfig
        - Validate config has required fields
        - Return config object
    """
    logger.info(f"Loading configuration from: {config_path or 'default'}")
    
    try:
        import yaml
        
        if config_path:
            config_path_obj = Path(config_path)
            
            # Check file exists
            if not config_path_obj.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            # Check file is readable
            if not config_path_obj.is_file():
                raise FileNotFoundError(f"Config path is not a file: {config_path}")
            
            try:
                with open(config_path_obj, 'r') as f:
                    config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Invalid YAML syntax in {config_path}: {e}")
            except OSError as e:
                raise FileNotFoundError(f"Cannot read config file {config_path}: {e}")
            
            if config_dict is None:
                raise ValueError(f"Config file is empty: {config_path}")
            
            config = PipelineConfig.from_dict(config_dict)
            
            # Validate required config sections exist
            required_sections = ['data', 'model', 'training']
            for section in required_sections:
                if not hasattr(config, section) or getattr(config, section) is None:
                    raise ValueError(f"Config missing required section: {section}")
        else:
            # Create default PipelineConfig
            config = PipelineConfig()
        
        logger.info("Configuration loaded successfully")
        return config
    
    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Configuration validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def run_pipeline(config: PipelineConfig, run_name: str = "default_run") -> dict:
    """
    Execute complete ML pipeline from data loading to prediction.
    
    Pipeline stages:
        1. Load training and test data
        2. Create preprocessing pipeline (imputation + encoding)
        3. Run stratified cross-validation with training engine
        4. Apply calibration to CV predictions
        5. Compute evaluation metrics
        6. Generate visualizations
        7. Log metrics to experiment tracking CSV
        8. Refit preprocessing on full training data
        9. Train final models on full training data
        10. Generate test predictions and submission CSV
    
    Parameters:
        config (PipelineConfig): Pipeline configuration
        run_name (str): Name for tracking this run
    
    Returns:
        dict: Results dictionary with metrics, predictions, and run metadata
    """
    import time
    
    logger.info(f"Starting pipeline run: {run_name}")
    pipeline_start = time.time()
    
    results = {
        'run_name': run_name,
        'config': config,
        'stages_completed': [],
        'errors': [],
        'metrics': {},
        'paths': {},
    }
    
    try:
        # ============================================================================
        # STAGE 1: Data Loading
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 1: Loading data...")
        stage_start = time.time()
        
        data_loader = CSVDataLoader(config.data)
        X_train, y_train = data_loader.load_train()
        X_test, test_respondent_ids = data_loader.load_test()
        
        # Separate respondent_id from features if included
        if 'respondent_id' in X_train.columns:
            train_respondent_ids = X_train['respondent_id'].copy()
            X_train = X_train.drop(columns=['respondent_id'])
        
        if 'respondent_id' in y_train.columns:
            y_train = y_train.drop(columns=['respondent_id'])
        
        if 'respondent_id' in X_test.columns:
            test_respondent_ids = X_test['respondent_id'].copy()
            X_test = X_test.drop(columns=['respondent_id'])
        
        # Basic validation
        if X_train.empty or y_train.empty or X_test.empty:
            raise ValueError("Loaded data is empty")
        
        # Validate data shape matching
        if len(X_train) != len(y_train):
            raise ValueError(
                f"Data shape mismatch: X_train has {len(X_train)} rows but "
                f"y_train has {len(y_train)} rows"
            )
        
        # Validate test features match training features
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"Feature count mismatch: training features have {X_train.shape[1]} "
                f"columns but test features have {X_test.shape[1]} columns"
            )
        
        # Validate target columns exist and contain valid values
        if not all(col in y_train.columns for col in ['h1n1_vaccine', 'seasonal_vaccine']):
            raise ValueError(
                f"Target columns missing. Expected: h1n1_vaccine, seasonal_vaccine. "
                f"Got: {list(y_train.columns)}"
            )
        
        # Validate target values are binary (0 or 1)
        for target_col in ['h1n1_vaccine', 'seasonal_vaccine']:
            unique_vals = y_train[target_col].unique()
            valid_vals = {0, 1, 0.0, 1.0}
            if not all(v in valid_vals for v in unique_vals):
                raise ValueError(
                    f"Invalid values in {target_col}: {unique_vals}. "
                    f"Expected only 0 or 1"
                )
        
        logger.info(f"  • Training features: {X_train.shape}")
        logger.info(f"  • Training labels: {y_train.shape}")
        logger.info(f"  • Test features: {X_test.shape}")
        logger.info(f"  ✓ Data loaded in {time.time() - stage_start:.2f}s")
        
        results['stages_completed'].append('data_loading')
        
        # ============================================================================
        # STAGE 2: Create Preprocessing Pipeline
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 2: Creating preprocessing pipeline...")
        stage_start = time.time()
        
        # Import preprocessing components
        from src.preprocessing import PreprocessingPipeline
        
        preprocessing_pipeline = PreprocessingPipeline(
            config.imputation,
            config.encoding
        )
        
        logger.info(f"  • Imputation strategy: {config.imputation.strategy}")
        logger.info(f"  • Encoding strategy: {config.encoding}")
        logger.info(f"  ✓ Preprocessing pipeline created in {time.time() - stage_start:.2f}s")
        
        results['stages_completed'].append('preprocessing_pipeline')
        
        # ============================================================================
        # STAGE 3: Training with Cross-Validation
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 3: Running stratified cross-validation...")
        stage_start = time.time()
        
        # Create training engine
        # Convert config objects to dicts for factory compatibility
        from dataclasses import asdict
        training_config_dict = asdict(config.training) if hasattr(config.training, '__dataclass_fields__') else config.training
        model_config_dict = asdict(config.model) if hasattr(config.model, '__dataclass_fields__') else config.model
        
        training_engine = TrainingEngine(training_config_dict, model_config_dict)
        
        # Run cross-validation
        cv_results = training_engine.run_cv(
            X_train,
            y_train,
            preprocessing_pipeline,
        )
        
        # Reconstruct out-of-fold predictions from fold results
        # Sort by original indices to get OOF predictions in original order
        cv_preds_h1n1 = np.zeros(len(X_train))
        cv_preds_seasonal = np.zeros(len(X_train))
        
        for fold_result in cv_results.fold_results:
            val_idx = fold_result.val_indices
            # Extract h1n1 and seasonal predictions from combined y_val_proba
            cv_preds_h1n1[val_idx] = fold_result.y_val_proba[:, 0]
            cv_preds_seasonal[val_idx] = fold_result.y_val_proba[:, 1]
        
        # Validate CV predictions
        if np.any(np.isnan(cv_preds_h1n1)):
            raise ValueError(
                f"CV H1N1 predictions contain NaN values. "
                f"Count: {np.sum(np.isnan(cv_preds_h1n1))}/{len(cv_preds_h1n1)}"
            )
        if np.any(np.isnan(cv_preds_seasonal)):
            raise ValueError(
                f"CV seasonal predictions contain NaN values. "
                f"Count: {np.sum(np.isnan(cv_preds_seasonal))}/{len(cv_preds_seasonal)}"
            )
        
        # Validate predictions are in valid probability range [0.0, 1.0]
        if np.any((cv_preds_h1n1 < 0.0) | (cv_preds_h1n1 > 1.0)):
            invalid_h1n1 = np.sum((cv_preds_h1n1 < 0.0) | (cv_preds_h1n1 > 1.0))
            raise ValueError(
                f"CV H1N1 predictions out of range [0.0, 1.0]. "
                f"Min: {cv_preds_h1n1.min():.6f}, Max: {cv_preds_h1n1.max():.6f}, "
                f"Invalid count: {invalid_h1n1}/{len(cv_preds_h1n1)}"
            )
        if np.any((cv_preds_seasonal < 0.0) | (cv_preds_seasonal > 1.0)):
            invalid_seasonal = np.sum((cv_preds_seasonal < 0.0) | (cv_preds_seasonal > 1.0))
            raise ValueError(
                f"CV seasonal predictions out of range [0.0, 1.0]. "
                f"Min: {cv_preds_seasonal.min():.6f}, Max: {cv_preds_seasonal.max():.6f}, "
                f"Invalid count: {invalid_seasonal}/{len(cv_preds_seasonal)}"
            )
        
        logger.info(f"  • CV folds: {config.data.cv_folds}")
        logger.info(f"  • CV predictions generated for h1n1: {cv_preds_h1n1.shape}")
        logger.info(f"  • CV predictions generated for seasonal: {cv_preds_seasonal.shape}")
        logger.info(f"  • H1N1 CV predictions - min: {cv_preds_h1n1.min():.6f}, max: {cv_preds_h1n1.max():.6f}, mean: {cv_preds_h1n1.mean():.6f}")
        logger.info(f"  • Seasonal CV predictions - min: {cv_preds_seasonal.min():.6f}, max: {cv_preds_seasonal.max():.6f}, mean: {cv_preds_seasonal.mean():.6f}")
        logger.info(f"  ✓ Cross-validation complete in {time.time() - stage_start:.2f}s")
        
        results['stages_completed'].append('training')
        
        # ============================================================================
        # STAGE 4: Probability Calibration
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 4: Calibrating predictions...")
        stage_start = time.time()
        
        calibrator_h1n1 = create_calibrator(config.calibration.method)
        calibrator_seasonal = create_calibrator(config.calibration.method)
        
        # Fit calibrators on CV predictions - note: fit(y_true, y_proba)
        calibrator_h1n1.fit(y_train['h1n1_vaccine'].values, cv_preds_h1n1)
        calibrator_seasonal.fit(y_train['seasonal_vaccine'].values, cv_preds_seasonal)
        
        # Calibrate CV predictions
        cv_preds_h1n1_calibrated = calibrator_h1n1.transform(cv_preds_h1n1)
        cv_preds_seasonal_calibrated = calibrator_seasonal.transform(cv_preds_seasonal)
        
        # Validate calibrated predictions
        if np.any(np.isnan(cv_preds_h1n1_calibrated)):
            raise ValueError(
                f"Calibrated H1N1 predictions contain NaN values. "
                f"Count: {np.sum(np.isnan(cv_preds_h1n1_calibrated))}/{len(cv_preds_h1n1_calibrated)}"
            )
        if np.any(np.isnan(cv_preds_seasonal_calibrated)):
            raise ValueError(
                f"Calibrated seasonal predictions contain NaN values. "
                f"Count: {np.sum(np.isnan(cv_preds_seasonal_calibrated))}/{len(cv_preds_seasonal_calibrated)}"
            )
        
        # Validate calibrated predictions are in valid range [0.0, 1.0]
        if np.any((cv_preds_h1n1_calibrated < 0.0) | (cv_preds_h1n1_calibrated > 1.0)):
            invalid_h1n1_cal = np.sum((cv_preds_h1n1_calibrated < 0.0) | (cv_preds_h1n1_calibrated > 1.0))
            raise ValueError(
                f"Calibrated H1N1 predictions out of range [0.0, 1.0]. "
                f"Min: {cv_preds_h1n1_calibrated.min():.6f}, Max: {cv_preds_h1n1_calibrated.max():.6f}, "
                f"Invalid count: {invalid_h1n1_cal}/{len(cv_preds_h1n1_calibrated)}"
            )
        if np.any((cv_preds_seasonal_calibrated < 0.0) | (cv_preds_seasonal_calibrated > 1.0)):
            invalid_seasonal_cal = np.sum((cv_preds_seasonal_calibrated < 0.0) | (cv_preds_seasonal_calibrated > 1.0))
            raise ValueError(
                f"Calibrated seasonal predictions out of range [0.0, 1.0]. "
                f"Min: {cv_preds_seasonal_calibrated.min():.6f}, Max: {cv_preds_seasonal_calibrated.max():.6f}, "
                f"Invalid count: {invalid_seasonal_cal}/{len(cv_preds_seasonal_calibrated)}"
            )
        
        logger.info(f"  • Calibration method: {config.calibration.method}")
        logger.info(f"  • H1N1 predictions - Original mean: {cv_preds_h1n1.mean():.4f}, Calibrated mean: {cv_preds_h1n1_calibrated.mean():.4f}")
        logger.info(f"  • H1N1 predictions - Calibrated min: {cv_preds_h1n1_calibrated.min():.6f}, max: {cv_preds_h1n1_calibrated.max():.6f}")
        logger.info(f"  • Seasonal predictions - Original mean: {cv_preds_seasonal.mean():.4f}, Calibrated mean: {cv_preds_seasonal_calibrated.mean():.4f}")
        logger.info(f"  • Seasonal predictions - Calibrated min: {cv_preds_seasonal_calibrated.min():.6f}, max: {cv_preds_seasonal_calibrated.max():.6f}")
        logger.info(f"  ✓ Calibration complete in {time.time() - stage_start:.2f}s")
        
        results['stages_completed'].append('calibration')
        
        # ============================================================================
        # STAGE 5: Evaluation & Metrics
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 5: Computing evaluation metrics...")
        stage_start = time.time()
        
        evaluator = Evaluator()
        
        # Calculate metrics using the Evaluator.get_diagnostics method
        metrics_full = evaluator.get_diagnostics(
            y_train['h1n1_vaccine'].values,
            y_train['seasonal_vaccine'].values,
            cv_preds_h1n1_calibrated,
            cv_preds_seasonal_calibrated,
        )
        
        # Extract the key metrics
        metrics = {
            'auroc_h1n1': metrics_full.get('auroc_h1n1', 0.0),
            'auroc_seasonal': metrics_full.get('auroc_seasonal', 0.0),
            'auroc_mean': metrics_full.get('auroc_mean', 0.0),
            'calibration_error_h1n1': metrics_full.get('h1n1_ece', np.nan),
            'calibration_error_seasonal': metrics_full.get('seasonal_ece', np.nan),
            'brier_score_h1n1': metrics_full.get('h1n1_brier', np.nan),
            'brier_score_seasonal': metrics_full.get('seasonal_brier', np.nan),
        }
        
        results['metrics'] = metrics
        
        logger.info(f"  • AUROC (H1N1): {metrics['auroc_h1n1']:.4f}")
        logger.info(f"  • AUROC (Seasonal): {metrics['auroc_seasonal']:.4f}")
        logger.info(f"  • AUROC (Mean): {metrics['auroc_mean']:.4f}")
        if not np.isnan(metrics['calibration_error_h1n1']):
            logger.info(f"  • Calibration Error (H1N1): {metrics['calibration_error_h1n1']:.4f}")
        if not np.isnan(metrics['calibration_error_seasonal']):
            logger.info(f"  • Calibration Error (Seasonal): {metrics['calibration_error_seasonal']:.4f}")
        if not np.isnan(metrics['brier_score_h1n1']):
            logger.info(f"  • Brier Score (H1N1): {metrics['brier_score_h1n1']:.4f}")
        if not np.isnan(metrics['brier_score_seasonal']):
            logger.info(f"  • Brier Score (Seasonal): {metrics['brier_score_seasonal']:.4f}")
        logger.info(f"  ✓ Metrics computed in {time.time() - stage_start:.2f}s")
        
        results['stages_completed'].append('evaluation')
        
        # ============================================================================
        # STAGE 6: Visualization
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 6: Creating visualizations...")
        stage_start = time.time()
        
        try:
            output_dir = Path(config.evaluation.output_dir) if hasattr(config.evaluation, 'output_dir') else Path('./outputs')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if plot_roc_curves is not None:
                plot_roc_curves(
                    y_train['h1n1_vaccine'].values,
                    cv_preds_h1n1_calibrated,
                    y_train['seasonal_vaccine'].values,
                    cv_preds_seasonal_calibrated,
                    output_path=output_dir / 'roc_curves.png'
                )
                logger.info(f"  • ROC curves saved to {output_dir / 'roc_curves.png'}")
            
            if plot_calibration_curve is not None:
                plot_calibration_curve(
                    y_train['h1n1_vaccine'].values,
                    cv_preds_h1n1_calibrated,
                    y_train['seasonal_vaccine'].values,
                    cv_preds_seasonal_calibrated,
                    output_path=output_dir / 'calibration_curves.png'
                )
                logger.info(f"  • Calibration curves saved to {output_dir / 'calibration_curves.png'}")
            
            # Plot confidence distribution
            from src.evaluation.plots import plot_confidence_distribution
            plot_confidence_distribution(
                cv_preds_h1n1_calibrated,
                cv_preds_seasonal_calibrated,
                output_path=output_dir / 'confidence_distribution.png'
            )
            logger.info(f"  • Confidence distribution saved to {output_dir / 'confidence_distribution.png'}")
            
            results['paths']['visualizations'] = str(output_dir)
            logger.info(f"  ✓ Visualizations created in {time.time() - stage_start:.2f}s")
            results['stages_completed'].append('visualization')
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")
            results['errors'].append(f"Visualization failed: {e}")
        
        # ============================================================================
        # STAGE 7: Experiment Tracking
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 7: Logging experiment results...")
        stage_start = time.time()
        
        try:
            log_path = config.tracking.log_path if hasattr(config.tracking, 'log_path') else 'experiments.csv'
            tracker = CSVExperimentLogger(log_path)
            
            run_id = tracker.create_run_id()
            tracker.log_run(
                run_id=run_id,
                model_type=config.model.model_type,
                config=config.to_dict() if hasattr(config, 'to_dict') else str(config),
                hyperparameters=config.model.hyperparameters if hasattr(config.model, 'hyperparameters') else {},
                metrics=metrics,
            )
            
            logger.info(f"  • Run ID: {run_id}")
            logger.info(f"  • Metrics logged to: {log_path}")
            results['paths']['tracking'] = log_path
            logger.info(f"  ✓ Tracking complete in {time.time() - stage_start:.2f}s")
            results['stages_completed'].append('tracking')
        except Exception as e:
            logger.warning(f"Could not log to tracking CSV: {e}")
            results['errors'].append(f"Tracking failed: {e}")
        
        # ============================================================================
        # STAGE 8: Refit Preprocessing on Full Training Data
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 8: Refitting preprocessing on full training data...")
        stage_start = time.time()
        
        from src.preprocessing import PreprocessingPipeline
        
        preprocessing_pipeline_full = PreprocessingPipeline(
            config.imputation,
            config.encoding
        )
        
        # Fit on full training data
        preprocessing_pipeline_full.fit(X_train)
        X_train_full_processed = preprocessing_pipeline_full.transform(X_train)
        X_test_processed = preprocessing_pipeline_full.transform(X_test)
        
        logger.info(f"  • Preprocessing refitted on all {len(X_train)} training samples")
        logger.info(f"  • Processed training features: {X_train_full_processed.shape}")
        logger.info(f"  • Processed test features: {X_test_processed.shape}")
        logger.info(f"  ✓ Preprocessing refitted in {time.time() - stage_start:.2f}s")
        
        results['stages_completed'].append('preprocessing_refit')
        
        # ============================================================================
        # STAGE 9: Train Final Models on Full Data
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 9: Training final models on full training data...")
        stage_start = time.time()
        
        # Train h1n1 model
        model_h1n1_final = ModelFactory.create_model(model_config_dict)
        model_h1n1_final.fit(X_train_full_processed, y_train['h1n1_vaccine'].values)
        
        # Train seasonal model
        model_seasonal_final = ModelFactory.create_model(model_config_dict)
        model_seasonal_final.fit(X_train_full_processed, y_train['seasonal_vaccine'].values)
        
        logger.info(f"  • H1N1 model trained on {len(X_train_full_processed)} samples")
        logger.info(f"  • Seasonal model trained on {len(X_train_full_processed)} samples")
        logger.info(f"  ✓ Final models trained in {time.time() - stage_start:.2f}s")
        
        results['stages_completed'].append('final_training')
        
        # ============================================================================
        # STAGE 10: Test Predictions & Submission
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 10: Generating test predictions and submission...")
        stage_start = time.time()
        
        # Generate test predictions
        test_pred_h1n1_proba = model_h1n1_final.predict_proba(X_test_processed)
        test_pred_seasonal_proba = model_seasonal_final.predict_proba(X_test_processed)
        
        # Handle both 1D and 2D outputs from predict_proba
        if test_pred_h1n1_proba.ndim == 2:
            test_pred_h1n1 = test_pred_h1n1_proba[:, 1]  # Extract probability of class 1
        else:
            test_pred_h1n1 = test_pred_h1n1_proba
        
        if test_pred_seasonal_proba.ndim == 2:
            test_pred_seasonal = test_pred_seasonal_proba[:, 1]  # Extract probability of class 1
        else:
            test_pred_seasonal = test_pred_seasonal_proba
        
        # Validate test predictions before calibration
        if np.any(np.isnan(test_pred_h1n1)):
            raise ValueError(
                f"H1N1 predictions contain NaN values. "
                f"Count: {np.sum(np.isnan(test_pred_h1n1))}/{len(test_pred_h1n1)}"
            )
        if np.any(np.isnan(test_pred_seasonal)):
            raise ValueError(
                f"Seasonal predictions contain NaN values. "
                f"Count: {np.sum(np.isnan(test_pred_seasonal))}/{len(test_pred_seasonal)}"
            )
        
        # Validate predictions are in valid probability range [0.0, 1.0]
        if np.any((test_pred_h1n1 < 0.0) | (test_pred_h1n1 > 1.0)):
            invalid_h1n1 = np.sum((test_pred_h1n1 < 0.0) | (test_pred_h1n1 > 1.0))
            raise ValueError(
                f"H1N1 predictions out of range [0.0, 1.0]. "
                f"Min: {test_pred_h1n1.min():.6f}, Max: {test_pred_h1n1.max():.6f}, "
                f"Invalid count: {invalid_h1n1}/{len(test_pred_h1n1)}"
            )
        if np.any((test_pred_seasonal < 0.0) | (test_pred_seasonal > 1.0)):
            invalid_seasonal = np.sum((test_pred_seasonal < 0.0) | (test_pred_seasonal > 1.0))
            raise ValueError(
                f"Seasonal predictions out of range [0.0, 1.0]. "
                f"Min: {test_pred_seasonal.min():.6f}, Max: {test_pred_seasonal.max():.6f}, "
                f"Invalid count: {invalid_seasonal}/{len(test_pred_seasonal)}"
            )
        
        # Apply calibration to test predictions
        test_pred_h1n1_calibrated = calibrator_h1n1.transform(test_pred_h1n1)
        test_pred_seasonal_calibrated = calibrator_seasonal.transform(test_pred_seasonal)
        
        # Validate calibrated predictions
        if isinstance(test_pred_h1n1_calibrated, np.ndarray):
            if test_pred_h1n1_calibrated.ndim == 2:
                test_pred_h1n1_calibrated = test_pred_h1n1_calibrated[:, 1]
            
            if np.any(np.isnan(test_pred_h1n1_calibrated)):
                raise ValueError(
                    f"Calibrated H1N1 predictions contain NaN values. "
                    f"Count: {np.sum(np.isnan(test_pred_h1n1_calibrated))}/{len(test_pred_h1n1_calibrated)}"
                )
            if np.any((test_pred_h1n1_calibrated < 0.0) | (test_pred_h1n1_calibrated > 1.0)):
                invalid_h1n1_cal = np.sum((test_pred_h1n1_calibrated < 0.0) | (test_pred_h1n1_calibrated > 1.0))
                raise ValueError(
                    f"Calibrated H1N1 predictions out of range [0.0, 1.0]. "
                    f"Min: {test_pred_h1n1_calibrated.min():.6f}, "
                    f"Max: {test_pred_h1n1_calibrated.max():.6f}, "
                    f"Invalid count: {invalid_h1n1_cal}/{len(test_pred_h1n1_calibrated)}"
                )
        
        if isinstance(test_pred_seasonal_calibrated, np.ndarray):
            if test_pred_seasonal_calibrated.ndim == 2:
                test_pred_seasonal_calibrated = test_pred_seasonal_calibrated[:, 1]
            
            if np.any(np.isnan(test_pred_seasonal_calibrated)):
                raise ValueError(
                    f"Calibrated seasonal predictions contain NaN values. "
                    f"Count: {np.sum(np.isnan(test_pred_seasonal_calibrated))}/{len(test_pred_seasonal_calibrated)}"
                )
            if np.any((test_pred_seasonal_calibrated < 0.0) | (test_pred_seasonal_calibrated > 1.0)):
                invalid_seasonal_cal = np.sum((test_pred_seasonal_calibrated < 0.0) | (test_pred_seasonal_calibrated > 1.0))
                raise ValueError(
                    f"Calibrated seasonal predictions out of range [0.0, 1.0]. "
                    f"Min: {test_pred_seasonal_calibrated.min():.6f}, "
                    f"Max: {test_pred_seasonal_calibrated.max():.6f}, "
                    f"Invalid count: {invalid_seasonal_cal}/{len(test_pred_seasonal_calibrated)}"
                )
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'respondent_id': test_respondent_ids.values,
            'h1n1_vaccine': test_pred_h1n1_calibrated,
            'seasonal_vaccine': test_pred_seasonal_calibrated,
        })
        
        # Create output directory
        output_dir = Path(config.prediction.output_dir) if hasattr(config.prediction, 'output_dir') else Path('./submissions')
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create output directory {output_dir}: {e}")
        
        # Validate output directory is writable
        if not os.access(output_dir, os.W_OK):
            raise OSError(f"Output directory is not writable: {output_dir}")
        
        # Save submission
        submission_filename = config.prediction.output_filename if hasattr(config.prediction, 'output_filename') else 'submission.csv'
        submission_path = output_dir / submission_filename
        
        try:
            submission_df.to_csv(submission_path, index=False)
        except OSError as e:
            raise OSError(f"Cannot write submission file to {submission_path}: {e}")
        except Exception as e:
            raise IOError(f"Error writing submission file {submission_path}: {e}")
        
        logger.info(f"  • Test predictions generated for {len(submission_df)} samples")
        logger.info(f"  • H1N1 predictions - mean: {test_pred_h1n1_calibrated.mean():.4f}, range: [{test_pred_h1n1_calibrated.min():.4f}, {test_pred_h1n1_calibrated.max():.4f}]")
        logger.info(f"  • Seasonal predictions - mean: {test_pred_seasonal_calibrated.mean():.4f}, range: [{test_pred_seasonal_calibrated.min():.4f}, {test_pred_seasonal_calibrated.max():.4f}]")
        logger.info(f"  • Submission saved to: {submission_path}")
        logger.info(f"  ✓ Test predictions complete in {time.time() - stage_start:.2f}s")
        
        results['paths']['submission'] = str(submission_path)
        results['stages_completed'].append('submission')
        
        # ============================================================================
        # Pipeline Complete
        # ============================================================================
        logger.info("=" * 80)
        total_time = time.time() - pipeline_start
        logger.info(f"✓ PIPELINE SUCCESSFUL")
        logger.info(f"  • Run Name: {run_name}")
        logger.info(f"  • Stages Completed: {len(results['stages_completed'])}/10")
        logger.info(f"  • AUROC (mean): {metrics['auroc_mean']:.4f}")
        logger.info(f"  • Total Time: {total_time:.2f}s")
        logger.info(f"  • Submission: {results['paths'].get('submission', 'N/A')}")
        logger.info("=" * 80)
        
        results['success'] = True
        results['total_time'] = total_time
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        results['errors'].append(str(e))
        results['success'] = False
        
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        results['errors'].append(str(e))
        results['success'] = False
    
    except (OSError, IOError) as e:
        logger.error(f"File I/O error: {e}")
        results['errors'].append(str(e))
        results['success'] = False
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        results['errors'].append(str(e))
        results['success'] = False
    
    return results


def main():
    """
    Main entry point for the pipeline.
    
    Parses command-line arguments:
        --config: Path to YAML configuration file (optional)
        --run-name: Name for this experiment run (optional, default: "default_run")
        --verbose: Enable verbose logging (optional)
        --seed: Random seed for reproducibility (optional, default: 42)
    
    Example:
        python main.py --config examples/config_baseline.yaml --run-name "baseline_001"
        python main.py --seed 123 --verbose
    """
    
    # ============================================================================
    # Argument Parsing
    # ============================================================================
    parser = argparse.ArgumentParser(
        description="ML Pipeline for Flu Shot Vaccination Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config examples/config_baseline.yaml
  python main.py --config examples/config_xgboost.yaml --run-name exp_001 --verbose
  python main.py --seed 42 --verbose
        """,
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file (default: use built-in config)',
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default='default_run',
        help='Name for this experiment run (default: default_run)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (default: False)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    
    args = parser.parse_args()
    
    # ============================================================================
    # Setup Logging
    # ============================================================================
    log_level_str = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level_str)
    logger.info(f"Pipeline initialized with seed={args.seed}")
    
    # ============================================================================
    # Set Random Seeds
    # ============================================================================
    seed_all_random_states(args.seed)
    
    # ============================================================================
    # Load Configuration
    # ============================================================================
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # ============================================================================
    # Run Pipeline
    # ============================================================================
    try:
        results = run_pipeline(config, run_name=args.run_name)
        
        # Log final status
        if results.get('success'):
            logger.info(f"✓ PIPELINE SUCCESSFUL")
            logger.info(f"  Run Name: {results['run_name']}")
            logger.info(f"  Stages Completed: {len(results['stages_completed'])}/10")
            if 'auroc_mean' in results.get('metrics', {}):
                logger.info(f"  AUROC (mean): {results['metrics']['auroc_mean']:.4f}")
            logger.info(f"  Total Time: {results.get('total_time', 'N/A'):.2f}s" if isinstance(results.get('total_time'), (int, float)) else "")
            
            # Log output paths
            if results.get('paths'):
                logger.info(f"  Output Paths:")
                for path_type, path in results['paths'].items():
                    logger.info(f"    • {path_type}: {path}")
            
            return 0
        else:
            logger.error(f"✗ PIPELINE FAILED")
            if results.get('errors'):
                logger.error(f"  Errors encountered:")
                for i, error in enumerate(results['errors'], 1):
                    logger.error(f"    {i}. {error}")
            return 1
    
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
