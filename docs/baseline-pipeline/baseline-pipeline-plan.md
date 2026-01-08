# Baseline Pipeline Execution Implementation Plan

## Overview

Implement a complete end-to-end ML pipeline that executes all 10 stages of the flu shot prediction workflow with the baseline configuration: load training data, apply preprocessing (imputation + encoding), train independent binary classifiers for each vaccine target using stratified cross-validation, calibrate probability predictions, evaluate performance, generate visualizations, track experiments, and produce a submission-ready CSV with test set predictions. The entire pipeline is driven by a single YAML configuration file with no hardcoded parameters.

## Goals

1. Create a working pipeline orchestrator that loads the baseline YAML config and executes all preprocessing, training, and evaluation stages in sequence
2. Implement data loading that reads CSV files and maintains respondent ID alignment across train/validation/test splits
3. Build a preprocessing pipeline that applies imputation and encoding consistently across all data splits without data leakage
4. Develop a stratified cross-validation loop that properly handles the multilabel nature (two independent targets) by creating balanced folds
5. Train two independent logistic regression models (one per vaccine) with proper probability calibration using Platt scaling
6. Generate comprehensive evaluation metrics (AUROC per vaccine, mean AUROC, calibration error, Brier score) and visualizations (ROC curves, calibration plots)
7. Create a test set prediction engine that generates the submission CSV in the required format (respondent_id, h1n1_vaccine, seasonal_vaccine)
8. Implement experiment tracking that logs metrics to CSV for comparing multiple pipeline runs

## Non-Goals

1. Hyperparameter search/optimization (configuration has stubs for this but not required for baseline execution)
2. Advanced imputation strategies beyond mean/mode/KNN/MICE skeleton implementations
3. Model types other than logistic regression (XGBoost, LightGBM support is structural but not required)
4. Threshold optimization or custom class weighting beyond the balanced strategy
5. Production deployment or API serving
6. Handling of streaming or online learning scenarios

## Implementation Steps

### Phase 1: Configuration & Data Foundation

1. **Complete config.py implementation** — Ensure YAML/JSON parsing works, validate all dataclass fields, implement config loading from `examples/config_baseline.yaml`
2. **Implement CSVDataLoader** — Load training features/labels and test features from CSV, return DataFrames with respondent_id tracking, handle missing files gracefully
3. **Set up logging and utilities** — Ensure logging is configured per config specifications, create utility functions for stratification, directory creation

### Phase 2: Preprocessing Pipeline

4. **Implement imputation strategies** — Build MeanImputer, ModeImputer, and KNNImputer classes that fit on training data and transform all splits consistently; handle both numeric and categorical columns
5. **Implement feature encoding** — Build OrdinalEncoder and OneHotEncoder that handle multiple feature groups (opinions, behavioral, medical, demographic, etc.), respect `drop_first=true`, and use `handle_unknown=ignore` for test set compatibility
6. **Create preprocessing orchestrator** — Combine imputation and encoding into a single PreprocessingPipeline that fits on training fold, transforms validation and test folds, prevents data leakage

### Phase 3: Model Training & Cross-Validation

7. **Implement model factory** — Ensure LogisticRegressionModel factory creates two independent binary classifiers with parameters from config (C, penalty, solver, class_weight=balanced)
8. **Implement training engine** — Build StratifiedKFold loop that:
   - Creates stratification column from combined labels: `h1n1 + 2*seasonal` (4 classes for balance)
   - For each fold: fits preprocessing on training subset only, trains models, generates validation predictions
   - Returns out-of-fold predictions for all validation instances
9. **Add probability prediction** — Ensure models output probabilities (0.0–1.0) not binary labels

### Phase 4: Calibration & Evaluation

10. **Implement Platt scaling calibrator** — Fit logistic regression on validation fold predictions vs actual labels; apply to test set predictions to improve probability calibration
11. **Implement evaluation metrics** — Calculate ROC AUC per vaccine (using sklearn.metrics.roc_auc_score), mean AUC across both vaccines, calibration error, Brier score
12. **Implement visualizations** — Generate ROC curves (per vaccine + combined), calibration curves (predicted vs actual probability), prediction confidence distributions
13. **Implement experiment tracking** — Log metrics to CSV file path specified in config for tracking multiple runs

### Phase 5: Test Prediction & Submission

14. **Implement test prediction engine** — Apply full preprocessing pipeline (fit on all training data) to test set, generate predictions from both models, handle any unknown categories
15. **Create submission generator** — Format predictions as CSV with columns: respondent_id, h1n1_vaccine, seasonal_vaccine; validate format against [data/submission_format.csv](data/submission_format.csv)
16. **Add validation layer** — Ensure predictions are valid probabilities (0.0–1.0 range), no NaN values, correct number of rows

### Phase 6: Main Orchestrator & Integration

17. **Implement main.py** — Single entry point that orchestrates all 10 pipeline stages:
    1. Load and parse config from command-line argument (default: `examples/config_baseline.yaml`)
    2. Initialize logger with config settings
    3. Load data via CSVDataLoader
    4. Run training engine (10-fold cross-validation)
    5. Execute calibration on validation fold predictions
    6. Generate evaluation metrics and visualizations
    7. Log metrics to experiment tracking CSV
    8. Run test prediction pipeline
    9. Generate and validate submission CSV
    10. Log completion and output file paths
18. **Add error handling** — Gracefully handle missing files, invalid configs, mismatched data shapes, and log errors with context

## Success Criteria

1. **Config Parsing**: Baseline YAML config loads without errors and populates all config objects
2. **Data Loading**: Training features (891×35), labels (891×2), and test features (417×35) load correctly with respondent_id integrity maintained
3. **Preprocessing**: Data shapes remain consistent after imputation and encoding; no NaN values in processed features; test set handles unseen categories
4. **Training**: 5-fold stratified CV completes successfully; out-of-fold predictions generated for all training instances; cross-validation uses proper stratification with 4 balanced classes
5. **Calibration**: Platt scaling calibrator trains and transforms predictions without errors
6. **Metrics**: 
   - AUROC calculated for h1n1 and seasonal vaccines separately
   - Mean AUROC computed as average of both
   - Calibration error and Brier score generated
   - Values are reasonable (AUROC ~0.80–0.82 per baseline spec)
7. **Visualizations**: ROC curves, calibration plots, and confidence distribution plots created and saved
8. **Submission**: CSV file generated with 418 rows (1 header + 417 test samples), valid probabilities (0.0–1.0), no NaN values, matches required format
9. **Tracking**: Metrics logged to CSV file; each row represents one pipeline run with timestamp and config identifier
10. **End-to-End Execution**: Running `python main.py examples/config_baseline.yaml` completes all 10 stages without manual intervention

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data leakage via preprocessing fit on full data | High | Implement strict separation: fit imputation/encoding ONLY on training fold, never on validation or test; use scikit-learn Pipeline or custom orchestrator to enforce this |
| Stratification fails for multilabel problem | High | Use combined stratification column: `h1n1 + 2*seasonal` creating 4 classes; verify fold distributions are balanced per vaccine |
| Class imbalance causes poor minority predictions | Medium | Config specifies `class_weight=balanced` in logistic regression; monitor per-vaccine AUROC separately to detect imbalance; log class distributions |
| Unknown categories in test set break encoding | Medium | Use `handle_unknown=ignore` in OneHotEncoder; pre-impute test set categorical features with training mode values before encoding |
| Probability calibration overfits to validation fold | Low | Use nested cross-validation (3 folds within training folds) or reserve separate calibration set; Platt scaling is robust to moderate overfitting |
| Missing values in employment features cause failures | Medium | Config allows `drop_columns` strategy; implement threshold-based dropping (e.g., >50% missing); alternatively use flag-as-missing imputation |
| Test set has different feature distributions | Low | Monitor predictions (should be probabilities, not binary); apply same imputation and encoding transformations as training; log test prediction statistics |

