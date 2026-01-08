# Baseline Pipeline Execution Task Checklist

**Last Updated**: 2026-01-07  
**Status**: Draft — Ready for Implementation

## Summary

- **Goal**: Implement complete end-to-end ML pipeline that loads baseline config and executes all preprocessing, training, evaluation, and submission stages
- **Key Phases**: Config & Data (Phase 1) → Preprocessing (Phase 2) → Training & CV (Phase 3) → Calibration & Evaluation (Phase 4) → Test Prediction (Phase 5) → Orchestration (Phase 6)
- **Dependencies**: All core module interfaces exist; implementations are stubbed with TODO comments
- **Test Data**: Training set 891×35 features + 891×2 labels; test set 417×35 features
- **Success Metric**: Running `python main.py examples/config_baseline.yaml` completes all stages; submission CSV generated with valid probabilities

---

## Phase 1: Configuration & Data Foundation

- [ ] **Parse baseline YAML config** — Implement config loading in [src/config.py](src/config.py); ensure all sections (data, imputation, encoding, model, training, calibration, evaluation, tracking, prediction, logging) parse correctly; validate required fields

- [ ] **Implement CSVDataLoader** — Complete [src/data/loader.py](src/data/loader.py); load training features, labels, and test features from CSV files; return pandas DataFrames with respondent_id as index; handle file not found errors

- [ ] **Set up logging infrastructure** — Configure logging in [src/utils/logging.py](src/utils/logging.py) per config specs (log level, log file path, format); ensure all modules log with proper hierarchy

- [ ] **Create utility helpers** — Implement in [src/utils/helpers.py](src/utils/helpers.py): stratification column creation (`h1n1 + 2*seasonal`), output directory creation, respondent ID validation

---

## Phase 2: Preprocessing Pipeline

- [ ] **Implement MeanImputer** — Complete [src/preprocessing/imputation.py](src/preprocessing/imputation.py); fit on training fold only, impute numeric columns with mean, handle categorical separately

- [ ] **Implement ModeImputer** — Complete mode-based imputation for categorical features; fit on training fold only, apply to validation and test

- [ ] **Implement KNNImputer** — Complete KNN-based imputation (sklearn.impute.KNNImputer wrapper); fit on training fold, use n_neighbors from config

- [ ] **Implement OrdinalEncoder** — Complete [src/preprocessing/encoding.py](src/preprocessing/encoding.py); handle ordinal features (opinions, behavioral, concern, knowledge), preserve order, fit on training fold only

- [ ] **Implement OneHotEncoder** — Complete one-hot encoding for categorical features (demographics, geographic, employment); respect `drop_first=true` and `handle_unknown=ignore` for test compatibility; fit on training fold only

- [ ] **Implement PreprocessingPipeline orchestrator** — Combine imputation and encoding into single pipeline; fit on training fold, apply consistently to validation and test; ensure no data leakage

---

## Phase 3: Model Training & Cross-Validation

- [ ] **Implement LogisticRegressionModel factory** — Complete [src/models/factory.py](src/models/factory.py); create sklearn LogisticRegression with params from config (C, penalty, solver, class_weight=balanced, max_iter)

- [ ] **Implement StratifiedKFold loop** — Complete [src/training/engine.py](src/training/engine.py):
  - Create stratification column from combined labels
  - Split data into K folds using StratifiedKFold with stratification
  - For each fold: fit preprocessing on training fold, train two independent models (h1n1, seasonal)
  - Collect validation fold predictions for all samples
  - Verify fold distributions are balanced

- [ ] **Implement probability prediction** — Ensure both models output probabilities (0.0–1.0 range) using `.predict_proba()`, not binary labels; verify no NaN or out-of-range values

- [ ] **Add fold timing and logging** — Log time for each fold, sample counts, class distributions per fold; identify any warnings or errors

---

## Phase 4: Calibration & Evaluation

- [ ] **Implement PlattScaler calibrator** — Complete [src/calibration/calibrator.py](src/calibration/calibrator.py):
  - Fit logistic regression on (validation_predictions, actual_labels) per vaccine
  - Apply fitted calibrator to test predictions
  - Verify calibrated predictions remain in [0.0, 1.0] range

- [ ] **Implement AUROC metric calculation** — Complete [src/evaluation/metrics.py](src/evaluation/metrics.py):
  - Calculate ROC AUC for h1n1 vaccine on validation fold
  - Calculate ROC AUC for seasonal vaccine on validation fold
  - Calculate mean AUROC (average of both)
  - Verify values match expected range (0.80–0.82 for baseline)

- [ ] **Implement calibration error metric** — Calculate Expected Calibration Error (ECE) or similar; measure gap between predicted and actual probabilities

- [ ] **Implement Brier score** — Calculate mean squared error between predicted probabilities and actual binary labels per vaccine

- [ ] **Implement ROC curve visualization** — Complete [src/evaluation/plots.py](src/evaluation/plots.py):
  - Plot individual ROC curves for h1n1 and seasonal
  - Plot combined ROC curve (average FPR/TPR across both)
  - Save to output_dir; include AUROC in title

- [ ] **Implement calibration curve visualization** — Plot predicted probability vs. actual frequency; show perfect calibration diagonal; save to output_dir

- [ ] **Implement confidence distribution plot** — Show histogram of predicted probabilities for both vaccines; save to output_dir

- [ ] **Implement experiment tracking** — Complete [src/tracking/logger.py](src/tracking/logger.py):
  - Create CSV logger that appends one row per pipeline run
  - Log timestamp, config identifier (e.g., "baseline"), AUROC per vaccine, mean AUROC, calibration error, Brier score
  - Append to file path specified in config (`experiments_baseline.csv`)

---

## Phase 5: Test Prediction & Submission

- [ ] **Fit preprocessing on full training data** — After CV loop, refit imputation and encoding on all training data (union of all CV folds); no holdout for preprocessing fit

- [ ] **Implement test set predictor** — Complete [src/prediction/predictor.py](src/prediction/predictor.py):
  - Apply fitted preprocessing to test features
  - Generate probability predictions from both trained models (h1n1, seasonal)
  - Return DataFrame with respondent_id, h1n1_vaccine, seasonal_vaccine

- [ ] **Handle unknown test categories** — Verify one-hot encoder uses `handle_unknown=ignore` to handle unseen categorical values; pre-fill or warn if significant unknown categories detected

- [ ] **Implement submission CSV generator** — Format predictions as: respondent_id, h1n1_vaccine, seasonal_vaccine; validate:
  - Correct number of rows (417 test samples)
  - No NaN values
  - All probabilities in [0.0, 1.0] range
  - respondent_id values match test set
  - Save to path specified in config (`submissions/submission_baseline.csv`)

- [ ] **Validate submission format** — Compare against [data/submission_format.csv](data/submission_format.csv); log validation result

---

## Phase 6: Main Orchestrator & Integration

- [ ] **Implement main.py orchestrator** — Create single entry point that orchestrates all 10 pipeline stages:
  1. Parse command-line argument for config path (default: `examples/config_baseline.yaml`)
  2. Load and validate config
  3. Initialize logger
  4. Load data via CSVDataLoader
  5. Run training engine (stratified CV loop)
  6. Apply calibration to validation fold predictions
  7. Calculate evaluation metrics on validation folds
  8. Generate visualizations (ROC, calibration, confidence)
  9. Log metrics to experiment tracking CSV
  10. Run test prediction and generate submission CSV
  11. Log completion with output file paths

- [ ] **Add error handling and validation** — Catch and log errors for:
  - Missing config file or invalid YAML
  - Missing data files
  - Data shape mismatches
  - NaN or out-of-range predictions
  - File write permissions
  - Gracefully exit with informative error messages

- [ ] **Add timing and progress logging** — Log:
  - Start/end time for each phase
  - Duration for each CV fold
  - Total pipeline runtime
  - Output paths for results and submission

---

## Phase 7: Testing & Validation

- [ ] **Manual end-to-end test** — Run `python main.py examples/config_baseline.yaml` and verify:
  - No errors or exceptions
  - Log file created at path specified in config
  - Submission CSV created and valid
  - Metrics logged to experiment tracking CSV
  - Visualizations created (ROC, calibration, confidence)
  - AUROC values in expected range (~0.80–0.82)

- [ ] **Validate output files** — Check:
  - Submission CSV has correct columns, row count, value ranges
  - ROC curves have expected shape (h1n1 and seasonal AUROC visible)
  - Calibration plots show reasonable fit
  - Confidence distribution plots have expected shapes
  - Log file contains no errors (only INFO and above)

- [ ] **Verify data integrity** — Confirm:
  - No data leakage (preprocessing fit only on training fold)
  - Respondent IDs preserved across all stages
  - No missing values in final predictions
  - CV fold distributions are balanced per vaccine

- [ ] **Test error cases** — Verify graceful handling of:
  - Missing config file
  - Invalid YAML syntax
  - Missing data files
  - Empty or corrupted CSV files

---

## Implementation Order

1. **Phase 1** — Config & Data Foundation (enables all downstream work)
   - Parse config, load data, setup utilities
   
2. **Phase 2** — Preprocessing Pipeline (required before training)
   - Imputation, encoding, orchestrator
   
3. **Phase 3** — Model Training & CV (core machine learning)
   - Model factory, training engine, stratification
   
4. **Phase 4** — Calibration & Evaluation (metrics and diagnostics)
   - Calibrator, metrics, visualizations, tracking
   
5. **Phase 5** — Test Prediction & Submission (generate submission)
   - Test predictor, submission formatter, validation
   
6. **Phase 6** — Main Orchestrator (tie everything together)
   - main.py implementation, error handling, logging
   
7. **Phase 7** — Testing & Validation (verify correctness)
   - Manual testing, output validation, error handling tests

---

## Acceptance Criteria

Each task should be:

- **Testable**: Can be verified independently (e.g., CSVDataLoader can be tested by loading a single CSV and checking shape/columns)
- **Atomic**: Can be completed as a standalone unit (e.g., PlattScaler does not depend on ROC visualization)
- **Specific**: Focused on a single deliverable (e.g., "implement MeanImputer" not "implement all preprocessing")
- **Actionable**: Has clear implementation steps (e.g., "use sklearn.impute.SimpleImputer for mean strategy")

All tasks should result in working, tested code with minimal technical debt.

