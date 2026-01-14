# Config Implementation Phase 2 Task Checklist

**Last Updated**: January 14, 2026  
**Phase**: Phase 2 (Implementation) — Implementation Stage  
**Dependencies**: Phase 1 ✅ COMPLETE (All config fields implemented)

---

## Summary

Phase 2 focuses on implementing the stub methods and making the pipeline use the configuration fields that were added in Phase 1. This involves implementing hyperparameter search, threshold tuning, complete metric calculations, and configuration-driven visualization.

**Estimated Duration**: 11–16 hours  
**Implementation Order**: A → B → C → D (sequential phases with dependencies)

---

## Phase 2A: Training Engine Implementation

### Hyperparameter Search Methods

- [ ] **Implement grid search strategy** – [src/training/engine.py:336-381](src/training/engine.py#L336)
  - Create exhaustive combinations of hyperparameters from `config.training.search_space`
  - Use itertools.product() to generate all combinations
  - Evaluate each combination with cross-validation
  - Return best hyperparameters and score
  - Target: Complete within 2 hours

- [ ] **Implement random search strategy** – [src/training/engine.py:336-381](src/training/engine.py#L336)
  - Sample random combinations from `config.training.search_space`
  - Number of samples from `config.training.search_params.n_trials` (if provided)
  - Use random seed from `config.data.random_seed` for reproducibility
  - Return best hyperparameters and score
  - Target: Complete within 1 hour

- [ ] **Implement Bayesian search strategy (Optuna)** – [src/training/engine.py:336-381](src/training/engine.py#L336)
  - Create Optuna study with sampler and pruner
  - Use `config.training.search_params` for Optuna config (n_trials, timeout, n_jobs)
  - Map `config.training.search_space` to Optuna suggest calls (suggest_int, suggest_float, suggest_categorical)
  - Define objective function that trains and evaluates model with suggested params
  - Return best hyperparameters and score
  - Verify with config_type_based_boosting.yaml (40 trials)
  - Target: Complete within 3 hours

- [ ] **Dispatch search strategy in `hyperparameter_search()` method** – [src/training/engine.py:336-381](src/training/engine.py#L336)
  - Read `config.training.search_strategy` and call appropriate method
  - Validate search_strategy is one of: "grid", "random", "bayesian"
  - Pass config to each search method
  - Return results in consistent format: `{"best_params": dict, "best_score": float}`
  - Target: Complete within 1 hour

### Threshold Tuning Methods

- [ ] **Implement threshold tuning** – [src/training/engine.py:397-407](src/training/engine.py#L397)
  - Take CV fold validation predictions and labels
  - For each vaccine (h1n1, seasonal):
    - Generate array of candidate thresholds [0.0, 1.0]
    - For each threshold, compute `config.training.threshold_metric` (default: "auc")
    - Find threshold that maximizes metric
  - Return dict: `{"h1n1_vaccine": float, "seasonal_vaccine": float}`
  - Validate thresholds are in range [0.0, 1.0]
  - Target: Complete within 2 hours

### Class Weight & SMOTE Methods

- [ ] **Implement `_apply_class_weights()` helper** – [src/training/engine.py:460+](src/training/engine.py#L460)
  - Read `config.training.class_weight_strategy`
  - If "balanced": compute weights inversely proportional to class frequencies
  - If "balanced_subsample": pass to model (for tree-based models)
  - If "none": no weights
  - Return weights dict for sklearn models
  - Target: Complete within 1 hour

- [ ] **Implement `_apply_smote()` helper** – [src/training/engine.py:460+](src/training/engine.py#L460)
  - If `config.training.use_smote=true`:
    - Create SMOTE sampler with ratio from `config.training.smote_ratio`
    - Apply to training fold
  - If `config.training.use_smote=false`: return data unchanged
  - Return resampled X_train, y_train
  - Target: Complete within 1 hour

- [ ] **Integrate class weights and SMOTE into `run_cv()` workflow** – [src/training/engine.py](src/training/engine.py)
  - Call `_apply_class_weights()` before training each fold
  - Call `_apply_smote()` after train/val split if enabled
  - Pass weights to `model.fit()` if model supports it
  - Verify class balance in validation fold remains unchanged (SMOTE only on train)
  - Target: Complete within 1 hour

---

## Phase 2B: Evaluation Engine Implementation

### Metric Computation Implementations

- [ ] **Complete ROC AUC computation** – [src/evaluation/metrics.py:113-116](src/evaluation/metrics.py#L113)
  - Use `sklearn.metrics.roc_auc_score()`
  - Compute per-vaccine ROC AUC
  - Average using `average="macro"` for multilabel result
  - Return dict: `{"h1n1_vaccine": float, "seasonal_vaccine": float, "mean": float}`
  - Target: Complete within 1 hour

- [ ] **Complete sensitivity, specificity, PPV computation** – [src/evaluation/metrics.py:160+](src/evaluation/metrics.py#L160)
  - Use threshold from `config.evaluation.threshold` (or tuned if "adaptive")
  - Apply threshold to probabilities to get binary predictions
  - Compute TP, FP, TN, FN from confusion matrix
  - Calculate: sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), PPV = TP/(TP+FP)
  - Compute per-vaccine and average
  - Target: Complete within 1 hour

- [ ] **Complete confusion matrix computation** – [src/evaluation/metrics.py:200+](src/evaluation/metrics.py#L200)
  - Use `sklearn.metrics.confusion_matrix()`
  - Compute per-vaccine confusion matrices
  - Return as dict of 2x2 arrays
  - Target: Complete within 30 minutes

- [ ] **Complete calibration error (ECE) computation** – [src/evaluation/metrics.py:273-283](src/evaluation/metrics.py#L273)
  - Implement Expected Calibration Error (ECE) formula:
    - Bin predictions into 10 bins
    - For each bin: compute |mean_predicted_prob - actual_fraction|
    - Average across bins, weighted by bin size
  - Compute per-vaccine and average
  - Target: Complete within 1 hour

- [ ] **Complete Brier score computation** – [src/evaluation/metrics.py:290+](src/evaluation/metrics.py#L290)
  - Use formula: mean((predicted_prob - actual_label)^2)
  - Compute per-vaccine and average
  - Target: Complete within 30 minutes

### Adaptive Threshold Evaluation

- [ ] **Update Evaluator to support adaptive thresholds** – [src/evaluation/metrics.py](src/evaluation/metrics.py)
  - Add parameter: `thresholds: dict = None` (from threshold tuning)
  - If `config.evaluation.threshold == "adaptive"` and thresholds provided:
    - Use thresholds for binary predictions (sensitivity, specificity, PPV)
    - Keep ROC AUC calculation without threshold (uses raw probabilities)
  - If `config.evaluation.threshold` is numeric:
    - Use specified float as threshold for all metrics
  - Target: Complete within 1 hour

---

## Phase 2C: Main Pipeline Integration

### Visualization Configuration

- [ ] **Update plot generation to be config-driven** – [src/evaluation/plots.py](src/evaluation/plots.py)
  - Create mapping of plot names to functions
  - Add validation for recognized plot types
  - Return list of valid plot names from plot module
  - Target: Complete within 1 hour

- [ ] **Update main.py to use visualization config** – [main.py:505-540](main.py#L505)
  - Check `config.evaluation.create_plots` before calling any plot function
  - If `create_plots=false`: skip all visualization, log "visualization disabled"
  - If `create_plots=true`: iterate through `config.evaluation.plot_types`
  - Call plot function only if plot name in config list
  - Log which plots were generated
  - Target: Complete within 1 hour

### Hyperparameter Search Orchestration

- [ ] **Update main.py to call hyperparameter search** – [main.py:377-459](main.py#L377)
  - After data loading and preprocessing:
    - Check `config.training.hyperparameter_search`
    - If true: call `training_engine.hyperparameter_search()` with config and X/y
    - Log best hyperparameters and score
    - Update `config.model.hyperparameters` with best params
  - If false: use hyperparameters from config directly
  - Target: Complete within 1 hour

- [ ] **Update main.py to integrate threshold tuning** – [main.py:377-459](main.py#L377)
  - After CV training, receive threshold tuning results from training engine
  - Store thresholds for use during evaluation
  - Log tuned thresholds per vaccine
  - Pass to evaluator
  - Target: Complete within 1 hour

### Config Validation & Logging

- [ ] **Add config validation layer** – [main.py or src/config.py](src/config.py)
  - Validate that search_space is provided if hyperparameter_search=true
  - Validate that search_strategy is recognized ("grid", "random", "bayesian")
  - Validate that plot_types values are recognized plot names
  - Log warnings for unused parameters (e.g., search_space if hyperparameter_search=false)
  - Target: Complete within 1 hour

- [ ] **Update experiment logging to capture hyperparameter search info** – [src/tracking/logger.py](src/tracking/logger.py)
  - Add fields to CSV logging:
    - `hyperparameter_search_enabled`: bool
    - `best_hyperparameters`: json string
    - `tuned_thresholds`: json string
  - Log whether adaptive thresholds were used
  - Target: Complete within 1 hour

---

## Phase 2D: Testing & Verification

### Unit/Component Tests

- [ ] **Test grid search with small search space** – [tests/ or standalone script](test_knn_neighbors.py)
  - Create minimal search space: `{"max_depth": [3, 4], "learning_rate": [0.1, 0.2]}`
  - Verify all 4 combinations are evaluated
  - Verify best params are returned
  - Verify score is numeric and in valid range
  - Target: Complete within 30 minutes

- [ ] **Test Bayesian search with Optuna** – [tests/](tests/)
  - Create search space matching config_type_based_boosting.yaml
  - Run with n_trials=5 for quick test
  - Verify search completes without errors
  - Verify best params are within search space bounds
  - Target: Complete within 1 hour

- [ ] **Test threshold tuning produces valid thresholds** – [tests/](tests/)
  - Create synthetic predictions and labels
  - Run threshold tuning
  - Verify output is dict with h1n1_vaccine and seasonal_vaccine
  - Verify thresholds are in [0.0, 1.0]
  - Verify thresholds differ if class imbalance differs
  - Target: Complete within 30 minutes

- [ ] **Test metric computations against sklearn baselines** – [tests/](tests/)
  - Create synthetic predictions and labels
  - Compute metrics via Evaluator
  - Compute same metrics via sklearn directly
  - Compare results (should match within rounding)
  - Target: Complete within 1 hour

- [ ] **Test visualization filtering** – [tests/](tests/)
  - Create config with `create_plots=false`
  - Verify no plots are generated
  - Create config with `plot_types=["roc_curves"]`
  - Verify only ROC curves are generated, not calibration curves
  - Target: Complete within 30 minutes

### Integration Tests

- [ ] **Run full pipeline with config_type_based_boosting.yaml** – Terminal
  - Command: `python main.py --config examples/config_type_based_boosting.yaml --run-name "xgboost_bayesian_v1" --verbose`
  - Verify hyperparameter search executes (40 trials)
  - Verify threshold tuning produces results
  - Verify specified plots are generated in results/ directory
  - Verify submission_*.csv is generated with valid format
  - Verify experiments_*.csv logs results with all metrics
  - Target: Complete within 2 hours (including execution time)

- [ ] **Run pipeline with config_type_based_lr_enhanced.yaml** – Terminal
  - Regression test: config without hyperparameter search
  - Verify pipeline completes without errors
  - Verify results are logged
  - Target: Complete within 1 hour (including execution time)

- [ ] **Run pipeline with config_baseline.yaml** – Terminal
  - Regression test: simplest config
  - Verify no errors introduced
  - Verify backward compatibility
  - Target: Complete within 1 hour (including execution time)

### Validation Checklist

- [ ] **Submission format validation**
  - File exists at expected path
  - Columns are: respondent_id, h1n1_vaccine, seasonal_vaccine
  - Row count matches test set (26,708)
  - All values in [0.0, 1.0]
  - No NaN or missing values
  - Target: Complete within 30 minutes

- [ ] **Results CSV validation**
  - experiments_*.csv file created
  - Contains run metadata: timestamp, model_type, config_json
  - Contains all metrics: auroc_h1n1, auroc_seasonal, auroc_mean, ece, brier
  - Contains threshold info: h1n1_threshold, seasonal_threshold
  - Target: Complete within 30 minutes

- [ ] **Reproducibility check**
  - Run same config twice with same random seed
  - Verify identical AUROC scores in results CSV
  - Verify identical best hyperparameters (if search used)
  - Verify identical tuned thresholds
  - Target: Complete within 1 hour (including execution time)

---

## Phase 2E: Documentation & Cleanup

- [ ] **Update README.md with hyperparameter search example** – [README.md](README.md)
  - Add section: "Running Hyperparameter Search"
  - Show example command: `python main.py --config examples/config_type_based_boosting.yaml`
  - Explain what hyperparameter search does
  - Document expected runtime (~30-60 min for 40 Bayesian trials)
  - Target: Complete within 30 minutes

- [ ] **Update IMPLEMENTATION_SUMMARY.md with Phase 2 results** – [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
  - Add section: "Phase 2: Config Implementation"
  - Document which methods were implemented
  - Show example results (AUROC scores, tuned thresholds)
  - Compare vs Phase 1 results (if available)
  - Target: Complete within 1 hour

- [ ] **Clean up any debug code or temporary test scripts** – [src/](src/)
  - Remove print statements and debugging code added during implementation
  - Ensure logging is clean and informative
  - Target: Complete within 30 minutes

- [ ] **Final code review and formatting** – [src/](src/)
  - Ensure consistent indentation and style
  - Add docstrings to new methods
  - Verify imports are organized
  - Target: Complete within 1 hour

---

## Implementation Order

### Critical Path (Must do in order)

1. **Phase 2A Training**: Implement hyperparameter search, threshold tuning, class weights/SMOTE
   - Prerequisite for Phase 2B
   - Prerequisite for Phase 2C integration
   - Estimated: 12 hours

2. **Phase 2B Evaluation**: Complete metric computations and adaptive thresholds
   - Prerequisite for Phase 2C visualization
   - Prerequisite for validation tests
   - Estimated: 5 hours

3. **Phase 2C Main Integration**: Wire up training and evaluation in main.py
   - Prerequisite for integration tests
   - Estimated: 4 hours

4. **Phase 2D Testing**: Run full pipeline tests
   - Should pass after Phases A-C complete
   - Estimated: 5 hours

### Parallel Work (Can do simultaneously if resources available)

- Phase 2E Documentation can start after Phase 2D tests pass
- Unit tests in Phase 2D can be written while Phase 2A/B implementation in progress

---

## Acceptance Criteria

Each task should be:
- **Testable**: Clear success criteria (metrics match sklearn, thresholds in valid range, etc.)
- **Atomic**: Can be completed and verified independently
- **Specific**: Focused on single deliverable (one search strategy, one metric, one plot, etc.)
- **Actionable**: Clear implementation steps and file locations

### Definition of "Phase 2 Complete"

✅ All stub methods are implemented (no more `pass` statements)  
✅ All config fields are actively used in the pipeline  
✅ Full pipeline runs end-to-end with config_type_based_boosting.yaml  
✅ Hyperparameter search completes successfully (40 Bayesian trials)  
✅ Threshold tuning produces valid thresholds for both vaccines  
✅ Metrics are computed correctly (validated vs sklearn)  
✅ Visualization respects create_plots and plot_types config  
✅ Adaptive threshold evaluation works correctly  
✅ Results are logged to CSV with all metrics  
✅ Submission file format is valid  
✅ Same config produces identical results (reproducibility test passes)  
✅ Regression tests pass (other configs still work)  

---

## Notes & Tips

### Important Implementation Gotchas

1. **CV fold handling in hyperparameter search**: Don't use the same validation fold for both hyperparameter search AND threshold tuning — causes data leakage. Solution: Aggregate all fold predictions for threshold tuning.

2. **Optuna callbacks**: May need to add callbacks for progress logging during Bayesian search (40 trials can take a while). Add simple logging every N trials.

3. **Threshold arrays**: When iterating through candidate thresholds, use linspace to generate smooth range: `np.linspace(0.0, 1.0, 101)` gives 101 candidate thresholds.

4. **Metric averaging**: For multilabel, use `average="macro"` in sklearn functions to get per-vaccine average (not micro or weighted).

5. **Config validation timing**: Validate config early in main.py before any heavy computation (data loading, preprocessing).

### Testing Commands

```bash
# Test single config loading
python -c "from src.config import PipelineConfig; config = PipelineConfig.from_yaml('examples/config_type_based_boosting.yaml'); print('✓ Config loaded')"

# Test with verbose logging
python main.py --config examples/config_type_based_boosting.yaml --run-name "test_run" --verbose

# Check submission format
python -c "import pandas as pd; sub = pd.read_csv('submissions/submission_*.csv'); print(f'Shape: {sub.shape}, Columns: {list(sub.columns)}, Value ranges: {sub.iloc[:,1:].describe()}')"

# Compare experiment results
cat experiments_xgboost.csv | column -t -s,
```

---

**Phase 1 Status**: ✅ COMPLETE  
**Phase 2 Status**: ⏳ READY FOR IMPLEMENTATION  
**Next Step**: Begin Phase 2A (Training Engine) implementation
