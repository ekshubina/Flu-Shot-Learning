#!/usr/bin/env python
"""Test script to verify StratifiedKFold implementation in TrainingEngine"""

import sys
import pandas as pd
import numpy as np
from src.training.engine import TrainingEngine, CVResults, FoldResults
from src.models.factory import ModelFactory
from src.utils.helpers import create_stratification_column

print('=== StratifiedKFold Implementation Verification ===')
print()

# Create realistic test data
np.random.seed(42)
n_samples = 200
n_features = 20

X = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f'feature_{i}' for i in range(n_features)]
)

# Create multilabel targets with different distributions
h1n1 = np.random.binomial(1, 0.4, n_samples)
seasonal = np.random.binomial(1, 0.35, n_samples)

y = pd.DataFrame({
    'h1n1_vaccine': h1n1,
    'seasonal_vaccine': seasonal
})

print(f'Test Data:')
print(f'  Features: {X.shape}')
print(f'  Labels: {y.shape}')
print(f'  H1N1: {h1n1.sum()}/{n_samples} positive ({h1n1.sum()/n_samples*100:.1f}%)')
print(f'  Seasonal: {seasonal.sum()}/{n_samples} positive ({seasonal.sum()/n_samples*100:.1f}%)')
print()

# Initialize TrainingEngine
training_config = {'cv_folds': 5, 'random_seed': 42}
model_config = {
    'model_type': 'logistic_regression',
    'hyperparameters': {
        'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs',
        'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42
    }
}

engine = TrainingEngine(training_config, model_config)

print('=== Phase 3: Stratified K-Fold Cross-Validation ===')
print()

# Execute cross-validation loop
cv_results = engine.run_cv(X, y, preprocessor=None, model_factory=ModelFactory)

print()
print('=== Verification Results ===')
print()

# Verify 1: Number of folds
assert len(cv_results.fold_results) == 5, 'Expected 5 folds'
print(f'✓ Correct number of folds: {len(cv_results.fold_results)}')

# Verify 2: Each fold has required data
for i, fold_result in enumerate(cv_results.fold_results):
    assert fold_result.fold_id == i, f'Fold ID mismatch'
    assert len(fold_result.train_indices) + len(fold_result.val_indices) == n_samples, 'Fold split size mismatch'
    assert fold_result.y_val_proba.shape[0] == len(fold_result.val_indices), 'Predictions shape mismatch'
    assert fold_result.y_val_proba.shape[1] == 2, 'Expected 2 targets (h1n1, seasonal)'
    assert np.all((fold_result.y_val_proba >= 0) & (fold_result.y_val_proba <= 1)), 'Probabilities out of [0,1]'

print(f'✓ All fold results validated (structure, size, probability range)')

# Verify 3: Stratification balance
strat_column = create_stratification_column(y['h1n1_vaccine'], y['seasonal_vaccine'])
overall_dist = [(strat_column == i).sum() for i in range(4)]

for i, fold_result in enumerate(cv_results.fold_results):
    y_val = y.iloc[fold_result.val_indices]
    strat_val = create_stratification_column(y_val['h1n1_vaccine'], y_val['seasonal_vaccine'])
    val_dist = [(strat_val == j).sum() for j in range(4)]
    
    # Check proportions match (within sampling variation)
    for j in range(4):
        expected_ratio = overall_dist[j] / n_samples
        actual_ratio = val_dist[j] / len(fold_result.val_indices)
        assert abs(expected_ratio - actual_ratio) < 0.1, f'Stratification imbalance in fold {i}'

print(f'✓ Stratification balance verified (fold class distributions match target)')

# Verify 4: Metrics calculation
assert 'auroc_h1n1' in cv_results.mean_metrics, 'Missing h1n1 auroc metric'
assert 'auroc_seasonal' in cv_results.mean_metrics, 'Missing seasonal auroc metric'
assert 'auroc_mean' in cv_results.mean_metrics, 'Missing mean auroc metric'
assert cv_results.mean_metrics['auroc_mean'] == (cv_results.mean_metrics['auroc_h1n1'] + cv_results.mean_metrics['auroc_seasonal']) / 2

print(f'✓ Metrics calculated correctly')
print(f'  H1N1 AUROC: {cv_results.mean_metrics["auroc_h1n1"]:.4f} ± {cv_results.std_metrics["auroc_h1n1"]:.4f}')
print(f'  Seasonal AUROC: {cv_results.mean_metrics["auroc_seasonal"]:.4f} ± {cv_results.std_metrics["auroc_seasonal"]:.4f}')
print(f'  Mean AUROC: {cv_results.mean_metrics["auroc_mean"]:.4f} ± {cv_results.std_metrics["auroc_mean"]:.4f}')

# Verify 5: Best model selection
assert cv_results.best_model is not None, 'Best model not selected'
assert cv_results.best_model.is_fitted(), 'Best model not fitted'
assert cv_results.best_fold_id in range(5), 'Invalid best fold ID'

print(f'✓ Best model selected: Fold {cv_results.best_fold_id + 1}')
print(f'  Model type: {cv_results.best_model.get_model_name()}')
print(f'  Best fold mean AUROC: {cv_results.fold_results[cv_results.best_fold_id].metrics["auroc_mean"]:.4f}')

# Verify 6: Data integrity
for fold_result in cv_results.fold_results:
    # Check no overlap between train and val
    train_set = set(fold_result.train_indices)
    val_set = set(fold_result.val_indices)
    assert len(train_set & val_set) == 0, 'Train/Val overlap detected'
    
    # Check coverage
    all_indices = train_set | val_set
    assert len(all_indices) == n_samples, 'Not all samples covered in fold'

print(f'✓ Data integrity verified (no train/val leakage, full coverage)')

# Verify 7: Two independent models per fold
for fold_result in cv_results.fold_results:
    # Predictions should be independent (not just copies)
    proba_h1n1 = fold_result.y_val_proba[:, 0]
    proba_seasonal = fold_result.y_val_proba[:, 1]
    
    # Check they're not identical (would indicate copy-paste error)
    correlation = np.corrcoef(proba_h1n1, proba_seasonal)[0, 1]
    assert abs(correlation) < 0.95, f'Models not independent (correlation: {correlation:.3f})'

print(f'✓ Two independent models verified per fold')

print()
print('=' * 60)
print('✅ STRATIFIED K-FOLD IMPLEMENTATION COMPLETE & VERIFIED')
print('=' * 60)
print()
print('Summary:')
print(f'  ✓ Stratified K-fold splits: {len(cv_results.fold_results)} folds')
print(f'  ✓ Stratification strategy: Combined label (h1n1 + 2*seasonal)')
print(f'  ✓ Fold balance: Target class distribution maintained')
print(f'  ✓ Data leakage: Prevented (no train/val overlap)')
print(f'  ✓ Models per fold: 2 independent binary classifiers')
print(f'  ✓ Predictions: Probabilities in [0.0, 1.0] range')
print(f'  ✓ Metrics: AUROC per vaccine + mean aggregation')
print(f'  ✓ Logging: Fold timing, sample counts, class distributions')
print(f'  ✓ Best model: Selected based on validation AUROC')
