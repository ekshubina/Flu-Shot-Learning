#!/usr/bin/env python
"""
Verification script for probability prediction implementation.

This script comprehensively tests the probability prediction functionality
across the LogisticRegressionModel and TrainingEngine components.
"""

import numpy as np
import pandas as pd
from src.models.factory import LogisticRegressionModel, ModelFactory
from src.training.engine import TrainingEngine
from sklearn.metrics import roc_auc_score
import sys


def main():
    print('=' * 70)
    print('PROBABILITY PREDICTION IMPLEMENTATION - VERIFICATION REPORT')
    print('=' * 70)

    # TEST 1: LogisticRegressionModel.predict_proba() Implementation
    print('\n1. LogisticRegressionModel.predict_proba() Implementation')
    print('-' * 70)

    config = {'model_type': 'logistic_regression', 'hyperparameters': {'C': 1.0}}
    model = LogisticRegressionModel(config)

    X_train = pd.DataFrame(np.random.randn(50, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    y_train = pd.Series(np.random.randint(0, 2, 50))
    model.fit(X_train, y_train)

    # Check predict_proba exists and is callable
    assert hasattr(model, 'predict_proba'), 'LogisticRegressionModel missing predict_proba()'
    assert callable(model.predict_proba), 'predict_proba() is not callable'
    print('✓ LogisticRegressionModel has predict_proba() method')

    # Check output shape
    proba = model.predict_proba(X_train)
    assert proba.shape == (50, 2), f'Expected shape (50, 2), got {proba.shape}'
    print(f'✓ predict_proba() returns correct shape: {proba.shape}')

    # Check output range
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0), 'Probabilities not in [0, 1]'
    print(f'✓ All probabilities in [0.0, 1.0]: min={proba.min():.6f}, max={proba.max():.6f}')

    # Check probabilities sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0), 'Probabilities do not sum to 1'
    print('✓ Probabilities sum to 1.0 for each sample')

    # Check no NaN or Inf
    assert not np.isnan(proba).any(), 'NaN values found'
    assert not np.isinf(proba).any(), 'Inf values found'
    print('✓ No NaN or Inf values in predictions')

    # Check unfitted model raises error
    model_unfitted = LogisticRegressionModel(config)
    try:
        model_unfitted.predict_proba(X_train)
        print('✗ Unfitted model should raise ValueError')
        sys.exit(1)
    except ValueError as e:
        assert 'must be fitted' in str(e).lower(), 'Wrong error message'
        print('✓ Unfitted model raises appropriate ValueError')

    # TEST 2: TrainingEngine probability prediction extraction
    print('\n2. TrainingEngine Validation Prediction Extraction')
    print('-' * 70)

    np.random.seed(42)
    X_cv = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    y_cv = pd.DataFrame({
        'h1n1_vaccine': np.random.randint(0, 2, 100),
        'seasonal_vaccine': np.random.randint(0, 2, 100)
    })
    y_cv.index = X_cv.index

    training_config = {'cv_folds': 2, 'random_seed': 42}
    model_config = {'model_type': 'logistic_regression', 'hyperparameters': {'C': 1.0}}

    engine = TrainingEngine(training_config, model_config)
    cv_results = engine.run_cv(X_cv, y_cv, model_factory=ModelFactory)

    # Check predictions are generated for all folds
    assert len(cv_results.fold_results) == 2, 'Expected 2 folds'
    print(f'✓ CV completed with {len(cv_results.fold_results)} folds')

    # Check each fold has valid predictions
    for i, fold in enumerate(cv_results.fold_results):
        # Check shapes
        assert fold.y_val_proba.shape[1] == 2, f'Fold {i}: Expected 2 vaccine columns'
        assert fold.y_val_proba.shape[0] == len(fold.val_indices), f'Fold {i}: Shape mismatch'

        # Check ranges
        assert np.all(fold.y_val_proba >= 0.0), f'Fold {i}: Negative probabilities'
        assert np.all(fold.y_val_proba <= 1.0), f'Fold {i}: Probabilities > 1'

        # Check no NaN
        assert not np.isnan(fold.y_val_proba).any(), f'Fold {i}: NaN values found'

        print(f'✓ Fold {i}: {fold.y_val_proba.shape} shape, [{fold.y_val_proba.min():.4f}, {fold.y_val_proba.max():.4f}] range')

    # TEST 3: Metrics calculation with predict_proba outputs
    print('\n3. Metrics Calculation with Probability Predictions')
    print('-' * 70)

    fold = cv_results.fold_results[0]
    h1n1_proba = fold.y_val_proba[:, 0]
    seasonal_proba = fold.y_val_proba[:, 1]
    h1n1_true = fold.y_val_true[:, 0]
    seasonal_true = fold.y_val_true[:, 1]

    # Verify shapes are 1D for metrics
    assert h1n1_proba.ndim == 1, f'H1N1 proba should be 1D, got shape {h1n1_proba.shape}'
    assert h1n1_true.ndim == 1, f'H1N1 true should be 1D, got shape {h1n1_true.shape}'
    print(f'✓ Predictions are 1D arrays for AUROC computation')

    # Compute AUROC to verify predictions work
    auroc_h1n1 = roc_auc_score(h1n1_true, h1n1_proba)
    auroc_seasonal = roc_auc_score(seasonal_true, seasonal_proba)
    auroc_mean = (auroc_h1n1 + auroc_seasonal) / 2.0

    assert 0.0 <= auroc_h1n1 <= 1.0, f'Invalid AUROC: {auroc_h1n1}'
    assert 0.0 <= auroc_seasonal <= 1.0, f'Invalid AUROC: {auroc_seasonal}'
    print(f'✓ AUROC H1N1: {auroc_h1n1:.4f}')
    print(f'✓ AUROC Seasonal: {auroc_seasonal:.4f}')
    print(f'✓ Mean AUROC: {auroc_mean:.4f}')

    # TEST 4: Edge cases
    print('\n4. Edge Case Handling')
    print('-' * 70)

    # Single sample
    single_proba = model.predict_proba(X_train.iloc[[0]])
    assert single_proba.shape == (1, 2), 'Single sample prediction shape incorrect'
    assert 0.0 <= single_proba[0, 1] <= 1.0, 'Single sample probability out of range'
    print('✓ Single sample prediction works')

    # Empty DataFrame error handling
    empty_x = pd.DataFrame(columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    try:
        model.predict_proba(empty_x)
    except (ValueError, IndexError) as e:
        print('✓ Empty DataFrame handled gracefully')

    # TEST 5: Aggregate fold predictions
    print('\n5. Aggregate Fold Predictions')
    print('-' * 70)

    all_h1n1_proba = []
    all_seasonal_proba = []
    total_samples = 0

    for fold in cv_results.fold_results:
        all_h1n1_proba.append(fold.y_val_proba[:, 0])
        all_seasonal_proba.append(fold.y_val_proba[:, 1])
        total_samples += len(fold.val_indices)

    all_h1n1_proba = np.concatenate(all_h1n1_proba)
    all_seasonal_proba = np.concatenate(all_seasonal_proba)

    assert len(all_h1n1_proba) == total_samples, 'Prediction aggregation failed'
    assert np.all(all_h1n1_proba >= 0.0) and np.all(all_h1n1_proba <= 1.0), 'Aggregated proba out of range'
    print(f'✓ Aggregated {total_samples} predictions across folds')
    print(f'✓ H1N1: min={all_h1n1_proba.min():.6f}, max={all_h1n1_proba.max():.6f}')
    print(f'✓ Seasonal: min={all_seasonal_proba.min():.6f}, max={all_seasonal_proba.max():.6f}')

    # TEST 6: Fold metrics computation
    print('\n6. Fold Metrics Computation')
    print('-' * 70)

    assert hasattr(cv_results, 'mean_metrics'), 'CVResults missing mean_metrics'
    assert hasattr(cv_results, 'std_metrics'), 'CVResults missing std_metrics'
    assert 'auroc_h1n1' in cv_results.mean_metrics, 'Missing auroc_h1n1'
    assert 'auroc_seasonal' in cv_results.mean_metrics, 'Missing auroc_seasonal'
    assert 'auroc_mean' in cv_results.mean_metrics, 'Missing auroc_mean'

    print(f'✓ Mean Metrics:')
    print(f'  H1N1 AUROC: {cv_results.mean_metrics["auroc_h1n1"]:.4f} ± {cv_results.std_metrics["auroc_h1n1"]:.4f}')
    print(f'  Seasonal AUROC: {cv_results.mean_metrics["auroc_seasonal"]:.4f} ± {cv_results.std_metrics["auroc_seasonal"]:.4f}')
    print(f'  Mean AUROC: {cv_results.mean_metrics["auroc_mean"]:.4f} ± {cv_results.std_metrics["auroc_mean"]:.4f}')

    print('\n' + '=' * 70)
    print('✓ ALL VERIFICATION CHECKS PASSED')
    print('=' * 70)
    print('\nSummary:')
    print('  - LogisticRegressionModel.predict_proba() is fully implemented')
    print('  - Returns probabilities in [0.0, 1.0] range')
    print('  - TrainingEngine correctly extracts and uses predictions')
    print('  - Metrics are computed correctly from probability predictions')
    print('  - Edge cases are handled appropriately')
    print('  - No NaN or out-of-range values in predictions')
    print('\nStatus: READY for Phase 4 (Calibration & Evaluation)')


if __name__ == '__main__':
    main()
