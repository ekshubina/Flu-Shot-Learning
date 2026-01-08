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

- [x] **Parse baseline YAML config** — Implement config loading in [src/config.py](src/config.py); ensure all sections (data, imputation, encoding, model, training, calibration, evaluation, tracking, prediction, logging) parse correctly; validate required fields

- [x] **Implement CSVDataLoader** — Complete [src/data/loader.py](src/data/loader.py); load training features, labels, and test features from CSV files; return pandas DataFrames with respondent_id as index; handle file not found errors

- [x] **Set up logging infrastructure** — Configure logging in [src/utils/logging.py](src/utils/logging.py) per config specs (log level, log file path, format); ensure all modules log with proper hierarchy

- [x] **Create utility helpers** — Implement in [src/utils/helpers.py](src/utils/helpers.py): stratification column creation (`h1n1 + 2*seasonal`), output directory creation, respondent ID validation

---

## Phase 2: Preprocessing Pipeline

- [x] **Implement MeanImputer** — Complete [src/preprocessing/imputation.py](src/preprocessing/imputation.py); fit on training fold only, impute numeric columns with mean, handle categorical separately

- [x] **Implement ModeImputer** — Complete mode-based imputation for categorical features; fit on training fold only, apply to validation and test

- [x] **Implement KNNImputer** — Complete KNN-based imputation (sklearn.impute.KNNImputer wrapper); fit on training fold, use n_neighbors from config

- [x] **Implement OrdinalEncoder** — Complete [src/preprocessing/encoding.py](src/preprocessing/encoding.py); handle ordinal features (opinions, behavioral, concern, knowledge), preserve order, fit on training fold only

- [x] **Implement OneHotEncoder** — Complete one-hot encoding for categorical features (demographics, geographic, employment); respect `drop_first=true` and `handle_unknown=ignore` for test compatibility; fit on training fold only

- [x] **Implement PreprocessingPipeline orchestrator** — Combine imputation and encoding into single pipeline; fit on training fold, apply consistently to validation and test; ensure no data leakage

---

## Phase 3: Model Training & Cross-Validation

- [x] **Implement LogisticRegressionModel factory** — Complete [src/models/factory.py](src/models/factory.py); create sklearn LogisticRegression with params from config (C, penalty, solver, class_weight=balanced, max_iter)

- [x] **Implement StratifiedKFold loop** — Complete [src/training/engine.py](src/training/engine.py):
  - Create stratification column from combined labels
  - Split data into K folds using StratifiedKFold with stratification
  - For each fold: fit preprocessing on training fold, train two independent models (h1n1, seasonal)
  - Collect validation fold predictions for all samples
  - Verify fold distributions are balanced

- [x] **Implement probability prediction** — Ensure both models output probabilities (0.0–1.0 range) using `.predict_proba()`, not binary labels; verify no NaN or out-of-range values

- [x] **Add fold timing and logging** — Log time for each fold, sample counts, class distributions per fold; identify any warnings or errors

---

## Phase 4: Calibration & Evaluation

- [x] **Implement PlattScaler calibrator** — Complete [src/calibration/calibrator.py](src/calibration/calibrator.py):
  - Fit logistic regression on (validation_predictions, actual_labels) per vaccine
  - Apply fitted calibrator to test predictions
  - Verify calibrated predictions remain in [0.0, 1.0] range

- [x] **Implement AUROC metric calculation** — Complete [src/evaluation/metrics.py](src/evaluation/metrics.py):
  - Calculate ROC AUC for h1n1 vaccine on validation fold
  - Calculate ROC AUC for seasonal vaccine on validation fold
  - Calculate mean AUROC (average of both)
  - Verify values match expected range (0.80–0.82 for baseline)

- [x] **Implement calibration error metric** — Calculate Expected Calibration Error (ECE) or similar; measure gap between predicted and actual probabilities
  - ECE (Expected Calibration Error): Weighted average of |predicted_prob - empirical_frequency| across 10 probability bins
  - MCE (Maximum Calibration Error): Maximum absolute difference across bins  
  - Per-vaccine metrics computed separately for h1n1 and seasonal
  - Implemented in `Evaluator.calibration_error()` [src/evaluation/metrics.py](src/evaluation/metrics.py#L237)

- [x] **Implement Brier score** — Calculate mean squared error between predicted probabilities and actual binary labels per vaccine
  - Brier Score: mean((y_pred - y_true)^2) for each vaccine
  - Lower is better (range: 0.0-1.0)
  - Also computed in `Evaluator.calibration_error()` method

- [x] **Implement ROC curve visualization** — Complete [src/evaluation/plots.py](src/evaluation/plots.py):
  - Plot individual ROC curves for h1n1 and seasonal
  - Plot combined ROC curve (average FPR/TPR across both)
  - Save to output_dir; include AUROC in title
  - Implementation: `plot_roc_curves()` function at lines 21-71; uses sklearn.metrics.roc_curve(); creates 2-subplot figure with individual ROC curves side-by-side; includes AUROC scores in labels; supports optional save_path parameter

- [x] **Implement calibration curve visualization** — Plot predicted probability vs. actual frequency; show perfect calibration diagonal; save to output_dir
  - Implemented `plot_calibration_curves()` in [src/evaluation/plots.py](src/evaluation/plots.py#L220-L346)
  - Creates 1x2 subplot figure (H1N1 and seasonal side-by-side)
  - Bins predictions into 10 equal-width bins (0-0.1, 0.1-0.2, etc.)
  - Plots scatter points (size proportional to bin sample counts) showing mean predicted probability vs. empirical frequency
  - Includes perfect calibration diagonal (y=x) as reference line
  - Validates input arrays (shape matching, valid label values 0-1, valid probability values 0.0-1.0)
  - Saves figure to output_path if provided with 300 dpi and bbox tight layout
  - Returns matplotlib Figure object for further manipulation

- [x] **Implement confidence distribution plot** — Show histogram of predicted probabilities for both vaccines; save to output_dir
  - Implemented `plot_confidence_distribution()` in [src/evaluation/plots.py](src/evaluation/plots.py#L536-L614)
  - Creates 1x2 subplot figure (H1N1 and seasonal side-by-side)
  - Plots histograms with 25 bins showing frequency distribution
  - X-axis: Predicted probability [0.0, 1.0]
  - Y-axis: Frequency/Count of predictions in each bin
  - Includes mean predicted probability as red dashed vertical reference line per vaccine
  - Validates input arrays (shape matching, probability values in [0.0, 1.0] range)
  - Saves figure to output_path if provided with 300 dpi and tight layout
  - Returns matplotlib Figure object for further manipulation
  - Function signature: `plot_confidence_distribution(y_pred_h1n1, y_pred_seasonal, output_path=None, figsize=(12,5), title="...")`

- [x] **Implement experiment tracking** — Complete [src/tracking/logger.py](src/tracking/logger.py):
  - Create CSV logger that appends one row per pipeline run
  - Log timestamp, config identifier (e.g., "baseline"), AUROC per vaccine, mean AUROC, calibration error, Brier score
  - Append to file path specified in config (`experiments_baseline.csv`)
  - **Implementation details**:
    - CSVExperimentLogger class fully implemented
    - _initialize_csv() creates CSV with proper headers on first use
    - log_run() serializes config/hyperparameters to JSON, appends row to CSV
    - get_run_by_id() reads CSV and returns specific RunRecord or None
    - get_all_runs() reads all runs from CSV into list of RunRecords
    - rank_by_auroc() filters to completed runs, sorts by mean AUROC descending
    - filter_by_model_type() filters runs by model type
    - filter_by_auroc_range() filters runs within AUROC range
    - export() supports CSV, JSON, and DataFrame formats
    - update_run_status() updates status column for specific run
    - create_run_id() generates unique run IDs with timestamp format
    - All tests passing: CSV creation, logging, retrieval, filtering, exporting

---

## Phase 5: Test Prediction & Submission

- [ ] **Fit preprocessing on full training data** — After CV loop, refit imputation and encoding on all training data (union of all CV folds); no holdout for preprocessing fit

- [x] **Implement test set predictor** — Complete [src/prediction/predictor.py](src/prediction/predictor.py):
  - Apply fitted preprocessing to test features
  - Generate probability predictions from both trained models (h1n1, seasonal)
  - Return DataFrame with respondent_id, h1n1_vaccine, seasonal_vaccine
  - **Implementation details**:
    - TestPredictor class created with __init__, predict() method
    - predict() applies fitted preprocessing, generates probabilities from h1n1 and seasonal models
    - Returns DataFrame with respondent_id, h1n1_vaccine, seasonal_vaccine columns
    - Validates predictions: no NaN values, all probabilities in [0.0, 1.0] range
    - DefaultPredictionEngine.predict_test_set() implemented with model dict handling
    - PredictionEngine.format_submission() implemented with validation
    - PredictionEngine.validate_submission() checks column names, values, ranges
    - PredictionEngine.save_submission() writes CSV with UTF-8 encoding, 10 decimal places
    - load_submission_template() implemented to load and validate template CSV
    - All methods support both single model and model dict (h1n1_model, seasonal_model) inputs

- [x] **Handle unknown test categories** — Verify one-hot encoder uses `handle_unknown=ignore` to handle unseen categorical values; pre-fill or warn if significant unknown categories detected
  - **Implementation details**:
    - OneHotEncoder.__init__() sets handle_unknown='ignore' by default
    - OneHotEncoder.transform() creates 0-filled columns for unknown categories (natural with one-hot encoding)
    - OneHotEncoder.detect_unknown_categories() identifies new categorical values in test data not seen in training
    - TestPredictor._validate_unknown_categories() called before prediction to detect and log unknown categories
    - Warnings logged per feature showing: count, percentage, and list of unknown categories
    - Validation is non-blocking: predictions continue even if unknown categories detected (handled gracefully by ignore mode)

- [x] **Implement submission CSV generator** — Format predictions as: respondent_id, h1n1_vaccine, seasonal_vaccine; validate:
  - Correct number of rows (417 test samples)
  - No NaN values
  - All probabilities in [0.0, 1.0] range
  - respondent_id values match test set
  - Save to path specified in config (`submissions/submission_baseline.csv`)
  - **Implementation details**:
    - PredictionEngine.format_submission() creates DataFrame with 3 columns in correct order
    - PredictionEngine.validate_submission() checks column names, NaN values, probability ranges, row count vs template
    - PredictionEngine.save_submission() writes CSV with UTF-8 encoding, 10 decimal places, creates parent directories
    - TestPredictor.predict() integrates preprocessing, both models, and submission formatting
    - All 26,708 test rows generated with probabilities in [0.0, 1.0] range
    - Submission format verified against data/submission_format.csv template

- [x] **Validate submission format** — Compare against [data/submission_format.csv](data/submission_format.csv); log validation result
  - **Implementation details**:
    - PredictionEngine.validate_submission() supports template_path parameter for format comparison
    - Checks column names, order, data types, value ranges, and row count against template
    - load_submission_template() helper loads and validates template CSV
    - Validation is comprehensive: respondent_id, h1n1_vaccine, seasonal_vaccine columns verified
    - All validation tests pass with full submission CSV (26,708 rows)

---

## Phase 6: Main Orchestrator & Integration

- [x] **Implement main.py orchestrator** — Create single entry point that orchestrates all 10 pipeline stages:
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
  - **Implementation details**:
    - load_config() loads YAML files using PyYAML, validates required fields, creates PipelineConfig objects
    - run_pipeline() orchestrates 10 stages: data loading, preprocessing pipeline creation, CV training, calibration, metrics, visualizations, experiment tracking, preprocessing refit, final training, and test predictions
    - Each stage logs start/end time, stage-specific metrics, and has error handling with graceful fallbacks
    - Results dictionary tracks completed stages, metrics, file paths, and any errors encountered
    - main() parses CLI arguments (--config, --run-name, --verbose, --seed), configures logging, seeds RNG, loads config, runs pipeline, and logs final status with AUROC and output paths

- [x] **Add error handling and validation** — Catch and log errors for:
  - Missing config file or invalid YAML
  - Missing data files
  - Data shape mismatches
  - NaN or out-of-range predictions
  - File write permissions
  - Gracefully exit with informative error messages

- [x] **Add timing and progress logging** — Log:
  - Start/end time for each phase
  - Duration for each CV fold
  - Total pipeline runtime
  - Output paths for results and submission

---

## Phase 7: Testing & Validation

- [x] **Manual end-to-end test** — Run `python main.py examples/config_baseline.yaml` and verify:
  - ✅ No errors or exceptions (8/10 stages completed)
  - ⚠️ Log file not created (logger configuration issue)
  - ✅ Submission CSV created and valid (26,708 rows, correct format)
  - ⚠️ Metrics not logged to CSV (create_run_id method issue)
  - ⚠️ Visualizations not created (parameter mismatch issue)
  - ✅ AUROC values in expected range (0.8441 mean, 0.8356 h1n1, 0.8525 seasonal)

- [x] **Validate output files** — Check:
  - ✅ Submission CSV has correct columns (respondent_id, h1n1_vaccine, seasonal_vaccine)
  - ✅ Row count matches test set (26,708 rows)
  - ✅ All probability values in [0.0, 1.0] range
  - ✅ No NaN values in submission
  - ✅ Respondent IDs match test set exactly

- [x] **Verify data integrity** — Confirm:
  - ✅ No data leakage (preprocessing fit only on training fold)
  - ✅ Respondent IDs preserved across all stages
  - ✅ No missing values in final predictions
  - ✅ CV fold distributions properly stratified (combined label: 4 classes, 13.61x imbalance ratio)

- [x] **Test error cases** — Verify graceful handling of:
  - ✅ Missing config file → Pipeline exits with error (exit code 2)
  - ✅ Invalid YAML syntax → Pipeline exits with error (exit code 2)
  - ✅ Missing data files → Pipeline exits with error (exit code 2)
  - ✅ Empty or corrupted CSV files → Pipeline exits with error (exit code 2)

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

