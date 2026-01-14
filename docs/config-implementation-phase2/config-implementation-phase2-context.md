# Config Implementation Phase 2 Context & References

## Key Files

### Files to Modify (Implementation Work)

| File | Current Status | Changes Needed | Priority |
|------|--------|----------------|----------|
| [src/training/engine.py](src/training/engine.py) | Stubs at lines 336-407 + 460+ | Implement hyperparameter_search(), apply_threshold_tuning(), class weight/SMOTE helpers | HIGH |
| [src/evaluation/metrics.py](src/evaluation/metrics.py) | Incomplete metric computation (TODO at lines 113-116, 273-283) | Complete ROC AUC, ECE, Brier, confusion matrix; support adaptive thresholds | HIGH |
| [src/evaluation/plots.py](src/evaluation/plots.py) | Hardcoded visualization | Add config-driven filtering by create_plots and plot_types | MEDIUM |
| [main.py](main.py) | Visualization hardcoded (lines 505-540), missing orchestration (lines 377-459) | Call hyperparameter search, integrate threshold tuning, wire up visualization config | MEDIUM |

### Configuration Files (Reference)

| File | Purpose | Relevance |
|------|---------|-----------|
| [examples/config_type_based_boosting.yaml](examples/config_type_based_boosting.yaml) | Target config with hyperparameter search and adaptive thresholds | End-to-end testing config; exercises all new features |
| [examples/config_type_based_lr_enhanced.yaml](examples/config_type_based_lr_enhanced.yaml) | LR variant without hyperparameter search | Baseline for comparison; should still work after changes |
| [examples/config_baseline.yaml](examples/config_baseline.yaml) | Simple baseline config | Regression test; should still work after changes |
| [src/config.py](src/config.py) | Configuration dataclasses | Reference for available config fields (already complete) |

### Reference Implementations (Patterns)

| File | Relevance | Pattern |
|------|-----------|---------|
| [src/preprocessing/imputation.py](src/preprocessing/imputation.py) | Strategy pattern for swappable algorithms | How to implement grid/random/Bayesian search as strategies |
| [src/models/factory.py](src/models/factory.py) | Factory pattern for model creation | Model instantiation and hyperparameter passing |
| [src/calibration/calibrator.py](src/calibration/calibrator.py) | Callback pattern for post-processing | How calibrators wrap predictions (threshold tuning similar) |
| [src/tracking/logger.py](src/tracking/logger.py) | CSV experiment logging | How results are logged; what metrics to capture |

### Existing Implementations to Study

| File | Implementation | Status |
|------|-----------------|--------|
| [src/preprocessing/__init__.py](src/preprocessing/__init__.py) | PreprocessingPipeline with fit/transform | ✅ Reference for pipeline orchestration |
| [src/calibration/calibrator.py](src/calibration/calibrator.py) | Platt, Isotonic, Temperature scaling | ✅ Reference for probability transformation |
| [src/evaluation/metrics.py](src/evaluation/metrics.py) | Evaluator class skeleton | ⚠️ Incomplete; needs implementation |

---

## Architecture Decisions

### Decision 1: Hyperparameter Search Strategy

**Context**: Config specifies three search strategies (grid, random, Bayesian) with different compute costs. Grid search is exhaustive but slow; random sampling is faster; Bayesian (Optuna) is most efficient for expensive evaluations.

**Decision**: Implement all three strategies as separate methods within `TrainingEngine.hyperparameter_search()`, dispatched by `config.training.search_strategy`.

**Rationale**: 
- Allows users to choose by configuration without code changes
- Each strategy has different pros/cons; supporting all enables flexibility
- Bayesian search (Optuna) is optimal for expensive CV loops with many hyperparameters

**Alternatives Considered**: 
- Use only Bayesian search (too complex for simple experiments)
- Use only grid search (too slow for large spaces)
- Make search strategy hardcoded per config (not flexible)

**Implementation Notes**:
- Grid search: Use itertools.product() to generate all combinations
- Random search: Use numpy random sampling with specified seed
- Bayesian search: Use Optuna with `suggest_int/suggest_float` for each hyperparameter

### Decision 2: Threshold Tuning Approach

**Context**: Two vaccines have different class imbalances (H1N1 ~21%, seasonal ~47%). A single threshold (0.5) may be suboptimal. Threshold tuning finds optimal per-vaccine cutoff via ROC AUC or other metric.

**Decision**: Implement threshold tuning post-CV by analyzing fold predictions. Tune separately per vaccine. Store tuned thresholds for use during evaluation.

**Rationale**:
- Per-vaccine thresholds account for different class balances
- Post-CV approach avoids train/test data leakage
- Tuned thresholds improve ROC AUC if class imbalance is significant
- Threshold tuning is standard practice in multilabel classification

**Alternatives Considered**:
- Single threshold for both vaccines (loses per-vaccine optimization)
- Tune during CV rather than after (potential leakage)
- Use different metrics per vaccine (complexity; not needed)

**Implementation Notes**:
- Use CV fold validation predictions to find threshold that maximizes ROC AUC
- Thresholds must be in range [0.0, 1.0]
- Store as dict: `{"h1n1_vaccine": 0.45, "seasonal_vaccine": 0.48}`

### Decision 3: Visualization Configuration Layer

**Context**: Currently, visualization is hardcoded in main.py (lines 505-540). All plots are generated unconditionally. New config fields `create_plots` and `plot_types` should control which plots are created.

**Decision**: Add a filtering layer in `main.py` that checks config before calling plot functions. If `create_plots=false`, skip all visualization. If `create_plots=true`, only call functions for plots in `plot_types` list.

**Rationale**:
- Configuration-driven visualization enables flexible experiment workflows
- Users can disable plots to save I/O time if results only needed in CSV
- Selective plot generation enables focused diagnostic investigations

**Alternatives Considered**:
- Move plot logic into EvaluationConfig (too much responsibility)
- Use function dispatch table (adds complexity)
- Keep plotting logic in plot functions (insufficient control)

**Implementation Notes**:
- Add validation: ensure `plot_types` contains only recognized plot names
- Map plot names to functions: `{"roc_curves": plot_roc_curves, ...}`
- Call plot function if `create_plots and plot_name in config.evaluation.plot_types`

### Decision 4: Metric Computation Implementation

**Context**: Several metric computation methods are stubbed with TODO comments. These are critical for evaluation but cannot be completed without understanding the full Evaluator API and existing code.

**Decision**: Complete missing metric implementations by studying existing evaluator patterns and using scikit-learn as reference implementation.

**Rationale**:
- Metrics are core to pipeline evaluation and experiment tracking
- Using sklearn ensures correctness and compatibility
- Complete implementation enables end-to-end validation

**Alternatives Considered**:
- Import metrics directly from sklearn (loses custom formatting/validation)
- Implement from scratch (high risk of bugs)

**Implementation Notes**:
- Use `sklearn.metrics.roc_auc_score()` for ROC AUC with average="macro" for multilabel
- Use `sklearn.metrics.confusion_matrix()` for confusion matrices
- Implement ECE and Brier score using standard formulas

---

## Dependencies

### Internal Dependencies

- **`TrainingEngine`** [src/training/engine.py]: Needs hyperparameter search implementation to feed best params to models
- **`Evaluator`** [src/evaluation/metrics.py]: Needs complete metric computations and adaptive threshold support
- **`ModelFactory`** [src/models/factory.py]: Already implemented; will receive best hyperparameters
- **`CSVExperimentLogger`** [src/tracking/logger.py]: Already implemented; will log results with metrics
- **`PipelineConfig`** [src/config.py]: Already complete; all config fields present

### External Dependencies

- **`optuna`** (for Bayesian search): Already in requirements.txt (check: `pip list | grep optuna`)
- **`scikit-learn`** (for metrics, grid search utilities): Already in requirements.txt
- **`numpy`** (for random sampling, threshold operations): Already in requirements.txt
- **`pandas`** (for CV fold handling, result aggregation): Already in requirements.txt
- **`xgboost`** (for hyperparameter search with XGBoost model): Already in requirements.txt

### Configuration Field Dependencies

New features depend on these config fields (all now implemented in [src/config.py](src/config.py)):
- `training.hyperparameter_search` (bool) — enable/disable search
- `training.search_strategy` (str) — "grid", "random", "bayesian"
- `training.search_space` (dict) — hyperparameter grid **[NEWLY ADDED]**
- `training.search_params` (dict) — Bayesian search config **[NEWLY ADDED]**
- `training.threshold_tuning` (bool) — enable/disable threshold optimization
- `training.threshold_metric` (str) — metric for threshold optimization
- `evaluation.create_plots` (bool) — enable/disable visualization **[NEWLY ADDED]**
- `evaluation.plot_types` (list) — which plots to generate **[NEWLY ADDED]**
- `evaluation.threshold` (float or "adaptive") — classification threshold **[NEWLY ADDED]**

---

## Related Documentation

- [CONFIG_AUDIT_QUICK_REF.md](CONFIG_AUDIT_QUICK_REF.md): Summary of Phase 1 (config field additions) completion
- [CONFIG_AUDIT_REPORT.md](CONFIG_AUDIT_REPORT.md): Detailed audit findings with priority rankings
- [CONFIG_IMPLEMENTATION_AUDIT.md](CONFIG_IMPLEMENTATION_AUDIT.md): Parameter-by-parameter verification
- [CONFIG_AUDIT_SUMMARY.md](CONFIG_AUDIT_SUMMARY.md): Executive summary of Phase 1 completion
- [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md): Architecture overview and component responsibilities
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md): Summary of implemented pipelines and results

---

## Open Questions

1. **Optuna availability**: Is optuna already installed? (`pip list | grep optuna` to verify)
   - If not, should we add to requirements.txt? (Likely yes, as it's standard for Bayesian optimization)

2. **Threshold tuning during CV**: Should threshold be tuned on each fold's validation set, or on aggregated CV predictions?
   - Recommendation: Aggregate all fold validation predictions, then tune once (simpler, less variance)

3. **Adaptive threshold mode**: When `evaluation.threshold="adaptive"`, should thresholds be tuned automatically during training, or does user provide pre-tuned thresholds?
   - Recommendation: Tune automatically during training; if manual thresholds needed, use numeric value instead of "adaptive"

4. **Plot file naming**: Should plots be differentiated by vaccine (e.g., `roc_curves_h1n1.png`, `roc_curves_seasonal.png`)?
   - Current code: Both vaccines on same plot
   - Recommendation: Keep current approach (cleaner); expand if needed later

5. **Metric computation for multilabel**: Should metrics be computed per-vaccine, then averaged, or as single multilabel calculation?
   - Recommendation: Per-vaccine then average (follow sklearn's average="macro" pattern for multilabel)

---

## Testing Strategy

### Unit Tests (if created)

- Test hyperparameter search with small search space (quick validation)
- Test threshold tuning produces valid thresholds [0.0-1.0]
- Test metric computations against sklearn baselines
- Test visualization filtering with different create_plots/plot_types values

### Integration Tests

- Run full pipeline with `config_type_based_boosting.yaml`
- Run full pipeline with `config_type_based_lr_enhanced.yaml` (without hyperparameter search)
- Run full pipeline with `config_baseline.yaml` (regression test)
- Verify submission file is valid (format, column names, value ranges)

### Validation Checklist

- [ ] Hyperparameter search completes without errors
- [ ] Best hyperparameters are different from defaults
- [ ] Threshold tuning produces thresholds in [0.0, 1.0]
- [ ] Specified plots are generated, unspecified plots are not
- [ ] Metrics are correctly computed (compare vs sklearn)
- [ ] Adaptive threshold evaluation works correctly
- [ ] Results are logged to CSV with all metrics
- [ ] Submission file has correct format and values
- [ ] Same config produces same results (reproducibility)

---

## Success Metrics

1. **Code coverage**: All previously-stubbed methods are implemented and called
2. **Test results**: Full pipeline runs end-to-end with all three example configs
3. **Performance**: XGBoost Bayesian search (40 trials) completes in <1 hour
4. **Correctness**: Metrics match sklearn baselines; thresholds in valid range
5. **Reproducibility**: Same config produces identical results across runs
6. **Feature completeness**: All config fields in YAML are actively used

---

**Last Updated**: January 14, 2026  
**Phase Status**: Phase 1 Complete ✅ | Phase 2 Planning ⏳  
**Ready for**: Implementation Sprint
