# ML Pipeline Scaffolding Task Checklist

## Summary

- **Dependencies**: None (can proceed immediately)
- **Estimated Time**: 4–6 hours (structural work only, no implementation)
- **Output**: 33 Python files + updated requirements.txt + example configs + README update
- **Verification**: All files importable, type checking passes, no runtime errors

---

## Directory & Package Structure

- [x] **Create src/ directory** – Create `/Users/kate/GitHub/Flu Shot/src/` directory
- [x] **Create src/__init__.py** – Root package init with module imports
- [x] **Create src/config.py** – Dataclasses for all configuration options
- [x] **Create data module** – Directory `/Users/kate/GitHub/Flu Shot/src/data/` with `__init__.py`
- [x] **Create preprocessing module** – Directory `/Users/kate/GitHub/Flu Shot/src/preprocessing/` with `__init__.py`
- [x] **Create models module** – Directory `/Users/kate/GitHub/Flu Shot/src/models/` with `__init__.py`
- [x] **Create training module** – Directory `/Users/kate/GitHub/Flu Shot/src/training/` with `__init__.py`
- [x] **Create calibration module** – Directory `/Users/kate/GitHub/Flu Shot/src/calibration/` with `__init__.py`
- [x] **Create evaluation module** – Directory `/Users/kate/GitHub/Flu Shot/src/evaluation/` with `__init__.py`
- [x] **Create tracking module** – Directory `/Users/kate/GitHub/Flu Shot/src/tracking/` with `__init__.py`
- [x] **Create prediction module** – Directory `/Users/kate/GitHub/Flu Shot/src/prediction/` with `__init__.py`
- [x] **Create utils module** – Directory `/Users/kate/GitHub/Flu Shot/src/utils/` with `__init__.py`
- [x] **Create examples directory** – Directory `/Users/kate/GitHub/Flu Shot/examples/` for config templates

## Configuration System

- [x] **Implement DataConfig** – Dataclass for data loading parameters (file paths, CV folds, random seed)
- [x] **Implement ImputationConfig** – Dataclass for imputation strategy selection and parameters
- [x] **Implement EncodingConfig** – Dataclass for feature encoding choices per group
- [x] **Implement ModelConfig** – Dataclass for model type and hyperparameters
- [x] **Implement TrainingConfig** – Dataclass for CV, class weights, SMOTE, threshold tuning options
- [x] **Implement CalibrationConfig** – Dataclass for calibration method and parameters
- [x] **Implement EvaluationConfig** – Dataclass for evaluation settings and metrics
- [x] **Implement TrackingConfig** – Dataclass for experiment tracking and logging
- [x] **Implement PipelineConfig** – Top-level dataclass composing all above + logging setup
- [x] **Add load_config_from_yaml()** – Function to deserialize YAML to PipelineConfig
- [x] **Add load_config_from_dict()** – Function to deserialize dict to PipelineConfig

## Data Module

- [x] **Define DataLoader ABC** – Abstract base class with `load_train()`, `load_test()`, `create_splits()`, `validate()` methods
- [x] **Implement CSVDataLoader stub** – Concrete stub for loading from CSV files (structure only, no file I/O logic)
- [x] **Add docstrings with examples** – Reference [PROBLEM_DESCRIPTION.md](../../docs/PROBLEM_DESCRIPTION.md) for data contracts

## Preprocessing Module

- [x] **Define ImputationStrategy ABC** – Abstract class with `fit()` and `transform()` methods
- [x] **Create imputation subclasses** – Stubs for: DropRowsImputation, DropColumnsImputation, MeanImputation, ModeImputation, KNNImputation, MICEImputation, FlagAsMissingImputation
- [x] **Define FeatureEncoder ABC** – Abstract class with `fit()`, `transform()`, `get_feature_names()` methods
- [x] **Create encoding subclasses** – Stubs for: OrdinalEncoder, OneHotEncoder, TargetEncoder, InteractionEncoder, PolynomialEncoder
- [x] **Add feature group constants** – Dictionary mapping feature group names to feature lists (opinions, behavioral, demographics, etc.)

## Models Module

- [x] **Define BaseModel ABC** – Abstract base class with `fit()`, `predict_proba()`, `get_feature_importance()`, `get_params()`, `set_params()` methods
- [x] **Create LogisticRegressionModel stub** – Wraps sklearn LogisticRegression, implements BaseModel interface
- [x] **Create XGBoostModel stub** – Wraps XGBoost classifier, implements BaseModel interface
- [x] **Create LightGBMModel stub** – Wraps LightGBM classifier, implements BaseModel interface
- [x] **Create RandomForestModel stub** – Wraps sklearn RandomForestClassifier, implements BaseModel interface
- [x] **Implement ModelFactory class** – Registry pattern with `create_model()` method that instantiates models by name

## Training Module

- [x] **Define TrainingEngine class** – Methods: `run_cv()`, `hyperparameter_search()`, `get_fold_predictions()`, `get_best_model()`
- [x] **Add cross-validation logic stub** – Placeholder for stratified k-fold split and fold iteration
- [x] **Add hyperparameter search stub** – Placeholder for grid/random/Bayesian search interface
- [x] **Add class imbalance handling stubs** – Placeholders for class weights, SMOTE, threshold tuning

## Calibration Module

- [x] **Define CalibratorInterface ABC** – Abstract class with `fit()`, `transform()`, `get_calibration_error()` methods
- [x] **Create PlattScalingCalibrator stub** – Implements logistic regression calibration
- [x] **Create IsotonicCalibrator stub** – Implements isotonic regression calibration
- [x] **Create TemperatureScalingCalibrator stub** – Implements temperature scaling
- [x] **Create NoCalibration stub** – Passthrough calibrator (baseline)

## Evaluation Module

- [x] **Define Evaluator class** – Methods: `compute_auroc()`, `confusion_matrix()`, `calibration_error()`, `get_diagnostics()`
- [x] **Implement metric computation stubs** – Placeholders for AUROC, ECE, Brier score, per-vaccine metrics
- [x] **Create ROC curve plotting stub** – In `plots.py`: `plot_roc_curves()` function
- [x] **Create calibration curve plotting stub** – In `plots.py`: `plot_calibration_curve()` function
- [x] **Create feature importance plotting stub** – In `plots.py`: `plot_feature_importance()` function
- [x] **Create prediction histogram stub** – In `plots.py`: `plot_prediction_confidence()` function

## Tracking Module

- [x] **Define ExperimentTracker ABC** – Methods: `log_run()`, `get_results()`, `rank_by_auroc()`, `export()`, `get_run_by_id()`
- [x] **Implement CSVExperimentLogger stub** – Concrete tracker using CSV log file
- [x] **Add run schema definition** – Columns for config, hyperparameters, metrics, timestamps
- [x] **Add filtering/querying methods** – `filter_by_model_type()`, `filter_by_auroc_range()`, etc. (stubs)

## Prediction Module

- [x] **Define PredictionEngine class** – Methods: `predict_test_set()`, `format_submission()`, `validate_submission()`
- [x] **Add test set inference stub** – Apply preprocessing and trained model to test data
- [x] **Add CSV formatting stub** – Transform predictions to submission format (respondent_id, h1n1_vaccine, seasonal_vaccine)
- [x] **Add validation stub** – Check submission format against `submission_format.csv` template

## Utilities Module

- [x] **Create logging.py** – `setup_logging()`, `get_logger()` functions with stub implementations
- [x] **Create validation.py** – `validate_features()`, `validate_labels()`, `validate_predictions()` stubs
- [x] **Create metrics.py** – Helper functions for AUROC, ECE, Brier score, per-vaccine metrics (stubs)
- [x] **Create plots.py** – Matplotlib/seaborn wrapper functions for ROC, calibration, importance plots (stubs)
- [x] **Create helpers.py** – `stratified_split()`, `get_class_weights()`, `get_feature_groups()` utility stubs

## Main Script & Orchestration

- [x] **Refactor main.py** – Orchestrator demonstrating component flow: load → impute → encode → train → calibrate → evaluate → track → predict
- [x] **Add config loading** – Parse config file or use default config in main.py
- [x] **Add error handling stubs** – Try/except blocks with logging placeholders
- [x] **Add data flow comments** – Document expected input/output at each stage
- [x] **Add CLI argument parsing** – Optional: config file path, run description, output directory

## Dependencies & Project Metadata

- [x] **Update requirements.txt** – Add numpy, scikit-learn, matplotlib, seaborn, optional: xgboost, lightgbm, optuna, jupyter
- [x] **Add version pinning** – Specify compatible versions (e.g., scikit-learn>=0.24.0,<2.0.0)
- [x] **Create requirements-dev.txt** – Development tools: pytest, black, flake8, sphinx (optional)
- [x] **Add .gitignore entries** – Ensure __pycache__/, *.pyc, .pytest_cache/, build/, dist/ are ignored

## Documentation & Examples

- [x] **Add architecture.md** – Diagram of module relationships, data flow, component dependencies
- [x] **Create config_baseline.yaml** – Example: logistic regression with simple imputation, one-hot encoding
- [x] **Create config_xgboost.yaml** – Example: XGBoost with KNN imputation, target encoding, class weights
- [x] **Update README.md** – Add "Project Structure" section explaining src/ organization, module purposes
- [x] **Add docstrings to all classes/methods** – Google style with type hints, reference to SYSTEM_DESIGN.md
- [x] **Create SCAFFOLDING.md** – Brief guide to extending/modifying scaffolds (how to add new model, imputation strategy, etc.)

## Validation & Quality Checks

- [x] **Verify import structure** – Ensure all `__init__.py` files export public APIs
- [x] **Check for circular imports** – Use tools like `python -m py_compile` or static analysis
- [x] **Validate type hints** – Run `mypy` or similar on src/ (strict mode optional)
- [x] **Verify main.py runs without error** – Execute with dummy/example config; should fail gracefully on missing implementations
- [x] **Check file count** – Should have 33 Python files + config templates + docs
- [x] **Verify README reflects structure** – User can navigate and understand purpose of each module

## Implementation Order

1. **Directories & packages** (Step 1–3: Create src/, __init__.py files, basic structure)
2. **Configuration system** (Step 4: config.py with all dataclasses)
3. **Component interfaces** (Steps 5–11: Define ABC classes and registry patterns)
4. **Component stubs** (Steps 12–25: Empty implementations of each component)
5. **Utilities** (Step 26: Helper functions)
6. **Main orchestrator** (Step 27: Refactor main.py with component wiring)
7. **Dependencies** (Step 28: Update requirements.txt)
8. **Documentation & examples** (Steps 29–31: README, config examples, architecture docs)
9. **Validation** (Step 32: Check imports, circular dependencies, type hints)
10. **Final review** (Step 33: Verify complete structure matches SYSTEM_DESIGN.md)

## Acceptance Criteria

Each completed task should satisfy:

- **Structural**: File created with correct path, module name, and package membership
- **Interface**: All abstract methods have signatures with type hints and docstrings referencing SYSTEM_DESIGN.md or PROBLEM_DESCRIPTION.md
- **Importable**: `from src.<module> import <class>` works without errors
- **Documented**: Every public class/function has a docstring (minimum: purpose, parameters, return type)
- **Typed**: All method signatures include type hints on parameters and return values
- **Integrated**: Component can be wired into main.py orchestrator without modification

---

## Progress Tracking
- Started: January 7, 2026
- Last Updated: January 7, 2026
- Completion: 92/92 tasks (100%) ✅ COMPLETE

**Status**: In progress
**Last Updated**: January 7, 2026
