# Config Implementation Phase 2 Plan

## Overview

The Flu Shot project's configuration system has been audited and all 47 parameters from `config_type_based_boosting.yaml` are now properly captured in dataclasses (Phase 1 ✅ COMPLETE). However, several critical methods are stubbed and don't actually use these configuration parameters. Phase 2 focuses on implementing these methods to make the pipeline fully configuration-driven.

The primary issue: Config fields exist but are not actively used by the training engine, evaluation logic, and visualization system. This prevents features like hyperparameter search, threshold tuning, and adaptive evaluation thresholds from functioning.

## Goals

1. **Implement hyperparameter search** with support for grid, random, and Bayesian optimization strategies
2. **Implement threshold tuning** to find optimal classification thresholds per vaccine
3. **Enable configuration-driven visualization** using `create_plots` and `plot_types` config fields
4. **Implement evaluation with adaptive thresholds** per config specification
5. **Complete missing core methods** in training engine and evaluation pipeline
6. **Verify end-to-end pipeline** execution with XGBoost hyperparameter search config

## Non-Goals

- Implement interaction term or polynomial feature generation (deferred, low priority)
- Implement metric filtering by `log_metrics` config field (deferred, low priority)
- Optimize hyperparameter search performance (implement first, optimize later)
- Add new imputation or encoding strategies (beyond existing stubs)

## Implementation Steps

### Phase 2A: Training Engine Implementation

1. **Implement `TrainingEngine.hyperparameter_search()` method** [src/training/engine.py:336-381]
   - Support three strategies: grid search (exhaustive), random search (sampling), Bayesian optimization (Optuna)
   - Read `config.training.search_space` (hyperparameter grid) and `config.training.search_params` (search config)
   - Return best hyperparameters and score for integration into model training
   - Test with `config_type_based_boosting.yaml` (40 Bayesian trials)

2. **Implement `TrainingEngine.apply_threshold_tuning()` method** [src/training/engine.py:397-407]
   - Take CV fold predictions and find optimal threshold per vaccine
   - Use `config.training.threshold_metric` (default: "auc") to guide optimization
   - Return optimal thresholds for use in evaluation
   - Verify with test set predictions

3. **Implement missing class weight and SMOTE helper methods** [src/training/engine.py:460+]
   - `_apply_class_weights()`: Apply `config.training.class_weight_strategy`
   - `_apply_smote()`: Apply oversampling when `config.training.use_smote=true`
   - Integrate into `run_cv()` workflow

### Phase 2B: Evaluation Engine Implementation

4. **Complete metric computation implementations** [src/evaluation/metrics.py:113-283]
   - ROC AUC calculation (currently TODO at lines 113-116)
   - Sensitivity, specificity, PPV calculations
   - Calibration error (ECE, Brier) currently stubbed at lines 273-283
   - Confusion matrix computation

5. **Implement adaptive threshold evaluation** [src/evaluation/metrics.py]
   - Support `config.evaluation.threshold` as float or "adaptive" string
   - If "adaptive", use threshold tuning result from Phase 2A
   - Apply threshold to probabilities before computing metrics
   - Ensure ROC AUC calculation ignores threshold (uses raw probabilities)

6. **Update plot generation to be config-driven** [src/evaluation/plots.py]
   - Read `config.evaluation.create_plots` boolean flag
   - Filter plots by `config.evaluation.plot_types` list
   - Only generate plots if `create_plots=true` AND plot type in list
   - Integrate into main pipeline [main.py:505-540]

### Phase 2C: Main Pipeline Integration

7. **Update `main.py` to orchestrate new features** [main.py:377-459]
   - Call hyperparameter search when `config.training.hyperparameter_search=true`
   - Pass best hyperparameters to model training
   - Retrieve threshold tuning results from training engine
   - Pass tuned thresholds to evaluator
   - Wire up visualization config to plot generation

8. **Implement configuration logging and validation** [main.py]
   - Log which config fields are actively used during pipeline execution
   - Validate config parameter combinations (e.g., search_strategy vs search_space)
   - Warn if unused parameters are specified

### Phase 2D: Testing & Verification

9. **Create integration tests** [tests/ or inline scripts]
   - Test hyperparameter search with small search space (quick validation)
   - Test threshold tuning produces valid thresholds [0, 1]
   - Test visualization respects `create_plots` and `plot_types`
   - Test end-to-end with `config_type_based_boosting.yaml`

10. **Run full pipeline with XGBoost boosting config** [examples/config_type_based_boosting.yaml]
    - Execute: `python main.py --config examples/config_type_based_boosting.yaml --run-name "xgboost_bayesian_v1" --verbose`
    - Verify hyperparameter search completes (40 trials)
    - Verify threshold tuning produces results
    - Verify specified plots are generated
    - Verify results logged to CSV with metrics
    - Validate submission file format and values

## Success Criteria

1. **Hyperparameter Search**: Bayesian search with 40 trials completes in <1 hour, finds better params than defaults
2. **Threshold Tuning**: Produces valid thresholds [0.0-1.0] for each vaccine
3. **Visualization**: Only specified plot types are generated when `create_plots=true`
4. **Evaluation Metrics**: All metrics (ROC AUC, ECE, Brier, confusion matrix) computed correctly
5. **Adaptive Thresholds**: Evaluation uses tuned thresholds when `threshold="adaptive"`
6. **End-to-End**: Full pipeline runs without errors, logs results, generates submission
7. **Reproducibility**: Same config produces same results across runs (with same random seed)

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hyperparameter search takes too long (Bayesian with 40 trials) | HIGH | Run with smaller trial count for testing; optimize Optuna callback |
| Threshold tuning causes overfitting to validation fold | MEDIUM | Use cross-validation folds instead of single validation set |
| Plot generation breaks existing visualization logic | MEDIUM | Add config-driven filtering layer; keep existing plot logic intact |
| Metric computation bugs propagate to results | MEDIUM | Write unit tests for each metric; validate against sklearn baselines |
| Configuration inconsistencies not caught early | LOW | Add validation layer in config loading; log warnings for unused params |

## Deliverables

- Updated [src/training/engine.py](src/training/engine.py) with hyperparameter search & threshold tuning
- Updated [src/evaluation/metrics.py](src/evaluation/metrics.py) with complete metric implementations
- Updated [src/evaluation/plots.py](src/evaluation/plots.py) with config-driven visualization
- Updated [main.py](main.py) to orchestrate new features
- Test results showing end-to-end pipeline execution with `config_type_based_boosting.yaml`
- Submission CSV with test set predictions using tuned hyperparameters and thresholds

## Timeline Estimate

- Phase 2A (Training engine): ~4-6 hours
- Phase 2B (Evaluation): ~3-4 hours
- Phase 2C (Main pipeline integration): ~2-3 hours
- Phase 2D (Testing & verification): ~2-3 hours
- **Total**: ~11-16 hours of implementation work

---

**Status**: Plan drafted for Phase 2 implementation  
**Phase 1 Status**: ✅ COMPLETE (all 47 config parameters captured)  
**Ready for**: Implementation by developer
