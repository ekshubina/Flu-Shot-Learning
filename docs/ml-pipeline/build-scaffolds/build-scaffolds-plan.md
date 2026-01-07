# ML Pipeline Scaffolding Implementation Plan

## Overview

Build a modular, component-based Python package structure that mirrors the [SYSTEM_DESIGN.md](../../SYSTEM_DESIGN.md) architecture. The scaffolds define clear interfaces and module boundaries without implementing business logic, enabling parallel development, easy component swapping, and systematic testing of different preprocessing, encoding, and modeling strategies.

This is the structural foundation for the entire ML pipeline—no actual ML happens here, just well-organized, type-hinted placeholder classes and functions that future implementation will fill in.

## Goals

1. Create modular directory structure (`src/`) with 9 component packages (data, preprocessing, models, training, calibration, evaluation, tracking, prediction, utils)
2. Define abstract interfaces and protocol classes for each component with method signatures and docstrings
3. Wire components into an orchestrator (`main.py`) that demonstrates data flow and configuration passing
4. Establish configuration system (`config.py`) for all pipeline parameters (imputation, encoding, model hyperparameters, CV strategy)
5. Update `requirements.txt` with complete ML dependencies (scikit-learn, numpy, matplotlib, seaborn, optional XGBoost/LightGBM)
6. Add utility modules for logging, validation, metrics, and visualization helpers
7. Create structured test/fixture stubs for future unit testing

## Non-Goals

- Implement actual ML logic (data loading, model training, feature encoding, etc.)
- Train any models or generate predictions
- Perform exploratory data analysis (EDA) or feature engineering
- Create notebooks or interactive experiments
- Set up external experiment tracking (MLflow, W&B) integration

## Implementation Steps

### Phase 1: Directory Structure & Package Setup

1. Create `src/` directory at project root
2. Create subdirectories: `src/data/`, `src/preprocessing/`, `src/models/`, `src/training/`, `src/calibration/`, `src/evaluation/`, `src/tracking/`, `src/prediction/`, `src/utils/`
3. Add `__init__.py` to each package (initially empty or with imports)
4. Create `src/config.py` for configuration dataclasses
5. Add `.gitignore` entries for `__pycache__/`, `*.pyc`, `.pytest_cache/`

### Phase 2: Define Component Interfaces

1. **Data Module** (`src/data/loader.py`): `DataLoader` abstract class with `load_train()`, `load_test()`, `create_splits()`, `validate()` methods
2. **Preprocessing Module** (`src/preprocessing/imputation.py` and `encoding.py`):
   - `ImputationStrategy` abstract class (drop, mean, KNN, MICE, flag-as-feature variants)
   - `FeatureEncoder` abstract class (ordinal, one-hot, target, custom transforms)
3. **Models Module** (`src/models/factory.py` and individual model files):
   - `BaseModel` abstract interface with `fit()`, `predict_proba()`, `get_feature_importance()`, `get_params()`, `set_params()`
   - Concrete stubs: `LogisticRegressionModel`, `XGBoostModel`, `LightGBMModel`, `RandomForestModel`
4. **Training Module** (`src/training/engine.py`):
   - `TrainingEngine` class with `run_cv()`, `hyperparameter_search()`, `get_fold_predictions()` methods
5. **Calibration Module** (`src/calibration/calibrator.py`):
   - `CalibratorInterface` abstract class (Platt, isotonic, temperature scaling, none)
   - `CalibratedPredictor` wrapper class
6. **Evaluation Module** (`src/evaluation/metrics.py` and `plots.py`):
   - `Evaluator` class with `compute_auroc()`, `confusion_matrix()`, `calibration_error()` methods
   - Visualization functions (ROC curves, calibration plots, feature importance plots)
7. **Tracking Module** (`src/tracking/logger.py`):
   - `ExperimentTracker` class with `log_run()`, `get_results()`, `rank_by_auroc()`, `export()` methods
8. **Prediction Module** (`src/prediction/predictor.py`):
   - `PredictionEngine` class with `predict_test_set()`, `format_submission()`, `validate_submission()` methods

### Phase 3: Configuration System

1. Create `src/config.py` with dataclasses:
   - `DataConfig` (train/test file paths, random seed, CV fold count)
   - `ImputationConfig` (strategy name, parameters)
   - `EncodingConfig` (per-feature-group encoding choices, drop list)
   - `ModelConfig` (model type, hyperparameters)
   - `TrainingConfig` (CV strategy, class weights, SMOTE, threshold tuning)
   - `CalibrationConfig` (method, parameters)
   - `PipelineConfig` (composition of all above)
2. Add utility function: `load_config_from_yaml()` or `load_from_dict()` for easy configuration switching

### Phase 4: Orchestrator & Main Script

1. Refactor `main.py`:
   - Import all module classes
   - Load `PipelineConfig` (from `config.py` or argument)
   - Instantiate components in sequence: `DataLoader` → `ImputationStrategy` → `FeatureEncoder` → `TrainingEngine` → `CalibratorInterface` → `Evaluator` → `ExperimentTracker` → `PredictionEngine`
   - Pass config and data between components
   - Add error handling and logging stubs
   - Include example run comments showing expected data flow
2. Add entry point: `if __name__ == "__main__":` block with argument parsing (config file, run description, etc.)

### Phase 5: Utility Modules

1. `src/utils/logging.py`: `setup_logging()`, `get_logger()` functions
2. `src/utils/validation.py`: `validate_features()`, `validate_labels()`, `validate_predictions()` functions
3. `src/utils/metrics.py`: Helper functions for computing metrics (AUROC, ECE, Brier score, etc.)
4. `src/utils/plots.py`: Visualization functions (ROC curves, calibration plots, feature importance)
5. `src/utils/helpers.py`: `stratified_split()`, `class_weights()`, `get_feature_groups()` utility functions

### Phase 6: Dependencies & Project Metadata

1. Update `requirements.txt` with complete ML stack:
   - Core: `numpy`, `pandas`, `scikit-learn`
   - Visualization: `matplotlib`, `seaborn`
   - Optional advanced: `xgboost`, `lightgbm`, `optuna` (Bayesian hyperparameter optimization)
   - Development: `jupyter`, `pytest`, `black`, `flake8`
2. Add optional dependencies section with comments (e.g., # For advanced ensemble experiments)
3. Create `requirements-dev.txt` for development-only tools

### Phase 7: Documentation & Examples

1. Add docstrings to all classes and methods (Google style, type hints)
2. Create `docs/architecture.md`: Diagram of module relationships and data flow
3. Add `examples/` folder with stub config files (e.g., `example_config_baseline.yaml`, `example_config_xgboost.yaml`)
4. Update [README.md](../../../README.md) with "Project Structure" section explaining `src/` organization

## Success Criteria

1. ✅ `src/` directory with 9 component packages, each with `__init__.py`
2. ✅ All 9 components have abstract base classes or protocol definitions with method signatures and docstrings
3. ✅ `src/config.py` with dataclasses for all configuration options
4. ✅ `main.py` orchestrates components in correct order with config passing and error handling stubs
5. ✅ `requirements.txt` updated with complete ML dependencies (numpy, scikit-learn, matplotlib, seaborn, optional xgboost/lightgbm)
6. ✅ `src/utils/` module with at least 4 utility files (logging, validation, metrics, plots)
7. ✅ All Python files are importable without errors (no business logic errors, only structural validation)
8. ✅ Type hints on all public methods and class attributes
9. ✅ README.md updated with project structure explanation
10. ✅ Example configuration files provided in `examples/` folder

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Over-abstraction creates complexity | Medium | Keep interfaces simple; use composition over inheritance where possible |
| Component dependencies become tangled | High | Use clear data-passing contracts (dataclasses, TypedDicts); avoid circular imports |
| Configuration system becomes unwieldy | Medium | Start with flat dataclass structure; refactor to hierarchical only if needed |
| Large number of files causes navigation difficulty | Low | Use clear naming convention; add docstring index at top of each package `__init__.py` |
| Stub code unclear about expected behavior | Medium | Comprehensive docstrings with examples; reference [SYSTEM_DESIGN.md](../../SYSTEM_DESIGN.md) in class docstrings |

---

**Last Updated**: January 7, 2026
**Status**: Plan approved and ready for implementation
