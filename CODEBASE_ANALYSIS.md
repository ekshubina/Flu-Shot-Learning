# Flu Shot ML Pipeline - Comprehensive Codebase Analysis

## Executive Summary

This is a **modular, component-based ML pipeline** for multilabel classification (predicting H1N1 and seasonal flu vaccine uptake). The architecture follows a clear 10-stage orchestration pattern: Data Loading → Imputation → Encoding → Training → Calibration → Evaluation → Visualization → Tracking → Prediction → Submission.

**Key Status**: ~90% of the infrastructure is designed and documented. **~70% is stubbed with TODO comments**. The main.py file serves as an orchestration template with clear comments for each stage but implementations are incomplete.

---

## Part 1: Current Implementation Status

### 1.1 What's FULLY IMPLEMENTED ✓

#### Configuration System (✓ Complete)
- **File**: [src/config.py](src/config.py)
- **Dataclasses**: 
  - `DataConfig` - 9 fields for data loading, CV folds, random seed, stratification
  - `ImputationConfig` - 5 fields for imputation strategy selection
  - `EncodingConfig` - 9 fields for feature encoding choices (ordinal, categorical, binary features listed)
  - `ModelConfig` - 6 fields for model type and hyperparameters
  - `TrainingConfig` - 11 fields for CV strategy, SMOTE, early stopping, hyperparameter search
  - `CalibrationConfig` - 3 fields for calibration method selection
  - `EvaluationConfig` - 8 fields for metrics and visualization options
  - `TrackingConfig` - 8 fields for experiment logging
  - `PipelineConfig` - Top-level compositor of all above + helper methods
  
- **Methods**:
  - `PipelineConfig.to_dict()` - Convert config to dictionary
  - `PipelineConfig.from_dict(config_dict)` - Load from dictionary with nested config creation
  - `PipelineConfig.from_yaml(yaml_path)` - Load from YAML file
  - `PipelineConfig.from_json(json_path)` - Load from JSON file
  - `PipelineConfig.to_yaml(yaml_path)` - Save to YAML file
  - `PipelineConfig.to_json(json_path)` - Save to JSON file
  - Module-level `load_config(config_path)` - Auto-detect format (.yaml, .yml, .json)

- **Feature Groups**: Complete FEATURE_GROUPS dict mapping feature names to categories
- **Constants**: ID_COLUMN='respondent_id', TARGETS=['h1n1_vaccine', 'seasonal_vaccine']

#### Main Entry Point & Pipeline Skeleton (✓ Mostly Complete)
- **File**: [main.py](main.py)
- **Functions**:
  - `load_config(config_path)` - Loads config via PipelineConfig.from_yaml() with error handling
  - `run_pipeline(config, run_name)` - **10-stage orchestration template** with clear structure
  - `main()` - Argument parser with --config, --run-name, --verbose, --seed options
  - `setup_logging()` call with log level configuration
  - `seed_all_random_states()` call for reproducibility
  
- **Pipeline Stages** (structure defined, implementation TODOs):
  1. Data Loading - CSVDataLoader instantiation
  2. Imputation - ImputationStrategy instantiation
  3. Feature Encoding - FeatureEncoder instantiation
  4. Model Training (CV) - TrainingEngine.run_cv()
  5. Calibration - create_calibrator() + fit/transform
  6. Evaluation & Metrics - Evaluator.get_diagnostics()
  7. Visualization (optional) - plot_roc_curves, plot_calibration_curve
  8. Experiment Tracking - CSVExperimentLogger.log_run()
  9. Test Prediction - DefaultPredictionEngine.predict_test_set()
  10. Submission Formatting - format_submission(), save_submission()

- **Error Handling**: Try-catch blocks for FileNotFoundError, ValueError, generic Exception
- **Return Value**: Dictionary with run_name, config, stages_completed, errors, success flag

#### Module Interfaces (Abstract Base Classes - ✓ Complete)
- **Data Loader**: [src/data/loader.py](src/data/loader.py)
  - `DataLoader` (ABC) with abstract methods: `load_train()`, `load_test()`, `create_splits()`
  - `DataSplit` dataclass with X_train, y_train, X_val, y_val, X_test, respondent_ids_test
  - `DataValidationResult` dataclass with validation check results
  - `CSVDataLoader` stub (declared but not implemented)

- **Imputation Strategy**: [src/preprocessing/imputation.py](src/preprocessing/imputation.py)
  - `ImputationStrategy` (ABC) with abstract methods: `fit()`, `transform()`, `fit_transform()`
  - Concrete stubs: `DropRowsImputation`, `DropColumnsImputation`
  - Pattern: Fit on training, transform on train/val/test

- **Feature Encoding**: [src/preprocessing/encoding.py](src/preprocessing/encoding.py)
  - `FeatureEncoder` (ABC) with abstract methods: `fit()`, `transform()`, `fit_transform()`, `get_feature_names()`
  - FEATURE_GROUPS dict defining ordinal, categorical, binary features
  - Encoding strategies documented: ordinal, one-hot, target encoding

- **Models**: [src/models/factory.py](src/models/factory.py)
  - `BaseModel` (ABC) with abstract methods: `fit()`, `predict_proba()`, `get_feature_importance()`
  - `predict()` method (hard predictions from probabilities using 0.5 threshold)
  - Model types documented: LogisticRegression, XGBoost, LightGBM, RandomForest
  - `ModelFactory` stub (for creating model instances)

- **Training Engine**: [src/training/engine.py](src/training/engine.py)
  - `FoldResults` dataclass with fold_id, train_indices, val_indices, y_val_true, y_val_proba, metrics, model
  - `CVResults` dataclass with fold_results, mean_metrics, std_metrics, best_fold_id, best_model
  - `TrainingEngine` class with methods: `run_cv()`, `hyperparameter_search()`, `get_fold_predictions()`
  - Supports: stratified k-fold CV, class imbalance (weights, SMOTE), threshold tuning, early stopping, hyperparameter search

- **Evaluation Metrics**: [src/evaluation/metrics.py](src/evaluation/metrics.py)
  - `Evaluator` class with static methods: `compute_auroc()` (✓ IMPLEMENTED), `confusion_matrix()`, etc.
  - `compute_auroc()` implementation:
    - Validates input shapes and value ranges [0,1]
    - Uses `sklearn.metrics.roc_auc_score()`
    - Returns (auroc_h1n1, auroc_seasonal, auroc_mean)
  - Planned metrics: calibration_error, ROC curves

- **Calibration**: [src/calibration/calibrator.py](src/calibration/calibrator.py)
  - `CalibratorInterface` (ABC) with methods: `fit()`, `transform()`, `fit_transform()`, `get_calibration_error()`
  - `create_calibrator()` factory function (stub)
  - Concrete stubs: `NoCalibration`, `PlattScalingCalibrator`, `IsotonicCalibrator`, `TemperatureScaling`

- **Prediction Engine**: [src/prediction/predictor.py](src/prediction/predictor.py)
  - `PredictionEngine` (ABC) with methods: `predict_test_set()`, `format_submission()`, `validate_submission()`, `save_submission()`
  - `DefaultPredictionEngine` stub
  - Submission format validation: respondent_id, h1n1_vaccine, seasonal_vaccine (probabilities [0,1])

- **Experiment Tracking**: [src/tracking/logger.py](src/tracking/logger.py)
  - `RunRecord` dataclass with run_id, timestamp, model_type, config_json, hyperparameters_json, metrics, status
  - `ExperimentTracker` (ABC) interface
  - `CSVExperimentLogger` stub for CSV-based tracking

#### Baseline Configuration (✓ Complete)
- **File**: [examples/config_baseline.yaml](examples/config_baseline.yaml)
- **Specifies**:
  - **Data**: training_set_features.csv, training_set_labels.csv, test_set_features.csv, 5 CV folds
  - **Imputation**: mean strategy with 0.5 drop_threshold
  - **Encoding**: Feature group strategies (ordinal for opinions, onehot for demographics)
  - **Model**: logistic_regression with C=1.0, penalty=l2, class_weight=balanced
  - **Training**: stratified_kfold, 5 folds, no hyperparameter search, balanced class weights, no SMOTE
  - **Calibration**: platt_scaling with 3 calibration folds
  - **Evaluation**: auroc, calibration_error, confusion_matrix, brier_score metrics
  - **Tracking**: CSV logging to experiments_baseline.csv
  - **Prediction**: Output to ./submissions/submission_baseline.csv

---

### 1.2 What's STUBBED (TODO) 🔲

#### Incomplete Implementations

| Module | Class/Function | Status | TODOs |
|--------|----------------|--------|-------|
| `src/data/loader.py` | `CSVDataLoader` | Stub | Implement load_train(), load_test(), create_splits(), validate() |
| `src/preprocessing/imputation.py` | All imputation classes | Stub | Implement fit/transform for all strategies (DropRows, DropColumns, Mean, Mode, KNN, MICE, etc.) |
| `src/preprocessing/encoding.py` | All encoder classes | Stub | Implement fit/transform for ordinal, one-hot, target encoding |
| `src/models/factory.py` | All model implementations | Stub | Implement LogisticRegression, XGBoost, LightGBM, RandomForest models |
| `src/training/engine.py` | `TrainingEngine` | Partial | run_cv(), hyperparameter_search(), get_fold_predictions() are stubs |
| `src/evaluation/metrics.py` | `compute_auroc()` | ✓ DONE | confusion_matrix(), calibration_error() stubs |
| `src/evaluation/plots.py` | All plot functions | Stub | plot_roc_curves(), plot_calibration_curve(), plot_feature_importance() |
| `src/calibration/calibrator.py` | All calibrators | Stub | Implement NoCalibration, PlattScaling, Isotonic, TemperatureScaling |
| `src/prediction/predictor.py` | `DefaultPredictionEngine` | Stub | predict_test_set(), format_submission(), validate_submission(), save_submission() |
| `src/tracking/logger.py` | `CSVExperimentLogger` | Stub | log_run(), filter_by_model_type(), rank_by_auroc(), export() |
| `src/utils/helpers.py` | Helper functions | Stub | stratified_train_test_split(), stratified_k_fold_split(), compute_class_weights() |
| `src/utils/logging.py` | setup_logging(), get_logger() | Stub | Logger configuration |
| `src/utils/validation.py` | validate_features(), validate_labels() | Stub | Data validation functions |

---

## Part 2: Architecture & Data Flow

### 2.1 Pipeline Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    main.py (Entry Point)                            │
│  • CLI argument parsing (--config, --run-name, --verbose, --seed)   │
│  • Configuration loading (YAML → PipelineConfig)                   │
│  • Random seed setting for reproducibility                         │
│  • run_pipeline() orchestration                                    │
└────────────────────────┬────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │   run_pipeline() - 10 Stages  │
         └───────────────┬───────────────┘
                         │
    ┌────────────────────────────────────────────────┐
    │ STAGE 1: Data Loading                          │
    │ • CSVDataLoader.load_train() → X_train, y_train│
    │ • CSVDataLoader.load_test() → X_test           │
    │ • validate_features(X_train)                   │
    │ • validate_labels(y_train)                     │
    │ Output: X_train, y_train, X_test, IDs          │
    └────────────────────┬───────────────────────────┘
                         │
    ┌────────────────────────────────────────────────┐
    │ STAGE 2: Imputation                            │
    │ • Create ImputationStrategy from config        │
    │ • .fit(X_train) → learn params                 │
    │ • X_train_imputed = .transform(X_train)        │
    │ • X_test_imputed = .transform(X_test)          │
    │ Output: Clean X_train_imputed, X_test_imputed  │
    └────────────────────┬───────────────────────────┘
                         │
    ┌────────────────────────────────────────────────┐
    │ STAGE 3: Feature Encoding                      │
    │ • Create FeatureEncoder from config            │
    │ • .fit(X_train_imputed)                        │
    │ • X_train_encoded = .transform(X_train_imputed)│
    │ • X_test_encoded = .transform(X_test_imputed)  │
    │ Output: Encoded features, feature names        │
    └────────────────────┬───────────────────────────┘
                         │
    ┌────────────────────────────────────────────────┐
    │ STAGE 4: Model Training (Cross-Validation)     │
    │ • TrainingEngine.run_cv(X_train_enc, y_train)  │
    │   - Create stratified k-fold splits             │
    │   - For each fold:                             │
    │     • Train model on fold training set         │
    │     • Predict on fold validation set           │
    │     • Compute metrics (AUROC, etc.)            │
    │     • Store FoldResults                        │
    │   - Aggregate metrics across folds             │
    │ • Optional: Hyperparameter search              │
    │ • Optional: Threshold tuning on val set        │
    │ Output: CVResults with fold models + metrics   │
    └────────────────────┬───────────────────────────┘
                         │
    ┌────────────────────────────────────────────────┐
    │ STAGE 5: Probability Calibration               │
    │ • Create calibrator (e.g., PlattScaling)       │
    │ • .fit(cv_predictions, y_train)                │
    │ • calibrated_predictions = .transform(...)     │
    │ Output: Calibrated probability predictions     │
    └────────────────────┬───────────────────────────┘
                         │
    ┌────────────────────────────────────────────────┐
    │ STAGE 6: Evaluation & Metrics                  │
    │ • Evaluator.compute_auroc() - h1n1, seasonal   │
    │ • Evaluator.confusion_matrix()                 │
    │ • Evaluator.calibration_error()                │
    │ Output: Metrics dict (auroc_mean, per-vaccine) │
    └────────────────────┬───────────────────────────┘
                         │
    ┌────────────────────────────────────────────────┐
    │ STAGE 7: Visualization (Optional)              │
    │ • plot_roc_curves(y_train, predictions)        │
    │ • plot_calibration_curve(y_train, predictions) │
    │ • plot_feature_importance(model)               │
    │ Output: PNG/PDF plots in results/              │
    └────────────────────┬───────────────────────────┘
                         │
    ┌────────────────────────────────────────────────┐
    │ STAGE 8: Experiment Tracking                   │
    │ • CSVExperimentLogger.log_run()                │
    │   - run_id, timestamp, model_type              │
    │   - config_json, hyperparameters_json          │
    │   - metrics (auroc_h1n1, auroc_seasonal, etc.) │
    │ Output: experiments_baseline.csv appended      │
    └────────────────────┬───────────────────────────┘
                         │
    ┌────────────────────────────────────────────────┐
    │ STAGE 9: Test Set Prediction                   │
    │ • DefaultPredictionEngine.predict_test_set()   │
    │   - Apply imputation to X_test                 │
    │   - Apply encoding to X_test                   │
    │   - Get ensemble predictions across CV folds   │
    │   (or retrain on full training set)            │
    │ • Returns: y_pred_h1n1, y_pred_seasonal        │
    │ Output: Probability predictions [0, 1]         │
    └────────────────────┬───────────────────────────┘
                         │
    ┌────────────────────────────────────────────────┐
    │ STAGE 10: Submission Formatting                │
    │ • format_submission(test_ids, y_h1n1, y_seas)  │
    │ • validate_submission(submission_df)           │
    │ • save_submission(submission_df, path)         │
    │ Output: submission_baseline.csv                │
    │   Columns: respondent_id, h1n1_vaccine,        │
    │            seasonal_vaccine (probabilities)    │
    └────────────────────┬───────────────────────────┘
                         │
            ┌────────────▼────────────┐
            │  Results Dict Returned  │
            │  • run_name              │
            │  • config                │
            │  • stages_completed      │
            │  • metrics               │
            │  • errors                │
            │  • success flag          │
            └─────────────────────────┘
```

### 2.2 Key Design Patterns

#### 1. **Strategy Pattern (Pluggable Algorithms)**
- **Imputation**: Different strategies (drop, mean, KNN, MICE) implement `ImputationStrategy` interface
- **Encoding**: Different strategies (ordinal, one-hot, target) implement `FeatureEncoder` interface
- **Calibration**: Different calibrators (Platt, isotonic, temperature) implement `CalibratorInterface`
- **Models**: Different models (LogReg, XGBoost, LightGBM) implement `BaseModel` interface

**Benefit**: Easy to swap strategies via config without changing orchestration code

#### 2. **Sklearn-like Fit/Transform Pattern**
All transformers follow:
```python
transformer = Strategy()
transformer.fit(X_train)  # Learn parameters
X_train_transformed = transformer.transform(X_train)
X_test_transformed = transformer.transform(X_test)  # Use learned params
```

**Benefit**: Familiar to sklearn users, prevents train-test leakage

#### 3. **Dataclass-based Configuration**
- Configuration fully specified as nested dataclasses
- YAML/JSON loading via `PipelineConfig.from_yaml()`, `from_json()`, `from_dict()`
- Easy to serialize/deserialize for experiment tracking

#### 4. **Factory Pattern (Models)**
- `ModelFactory` creates model instances from config
- Hides model implementation details
- Supports hyperparameter search over factory-created models

#### 5. **Multilabel Handling**
- Two independent binary classification targets
- Each vaccine prediction computed separately
- Metrics averaged across vaccines: `auroc_mean = (auroc_h1n1 + auroc_seasonal) / 2`

---

## Part 3: Configuration System Deep Dive

### 3.1 Configuration Hierarchy

```
PipelineConfig (Top level)
├── name: str
├── description: str
├── data: DataConfig
│   ├── train_features_path
│   ├── train_labels_path
│   ├── test_features_path
│   ├── submission_format_path
│   ├── n_folds: 5
│   ├── random_seed: 42
│   ├── stratify: True
│   └── val_split: 0.2
├── imputation: ImputationConfig
│   ├── strategy: "mean"  # or drop_rows, drop_columns, knn, mice, flag_as_missing
│   ├── n_neighbors: 5
│   ├── mice_iterations: 10
│   ├── fill_value: 0.0
│   └── drop_threshold: 0.5
├── encoding: EncodingConfig
│   ├── ordinal_features: [...]  # list of feature names
│   ├── categorical_features: [...]
│   ├── binary_features: [...]
│   ├── ordinal_encoding_type: "ordinal"  # or label, as_is
│   ├── categorical_encoding_type: "one_hot"  # or target, ordinal
│   ├── interaction_terms: False
│   ├── polynomial_degree: 0
│   ├── drop_first_onehot: True
│   └── target_encoding_smoothing: 1.0
├── model: ModelConfig
│   ├── model_type: "logistic_regression"  # or xgboost, lightgbm, random_forest, neural_net
│   ├── hyperparameters: {}  # Dict[str, Any]
│   ├── random_seed: 42
│   ├── n_jobs: 1  # -1 for all cores
│   ├── class_weight: "balanced"  # or None, custom dict
│   └── sample_weight: False
├── training: TrainingConfig
│   ├── cv_strategy: "stratified_kfold"  # or kfold, repeated_stratified_kfold
│   ├── test_size: 0.2
│   ├── use_smote: False
│   ├── smote_ratio: 0.5
│   ├── threshold_tuning: False
│   ├── threshold_metric: "auc"
│   ├── early_stopping: False
│   ├── early_stopping_rounds: 10
│   ├── hyperparameter_search: False
│   ├── search_strategy: "grid"  # or random, bayesian
│   └── search_cv_folds: 3
├── calibration: CalibrationConfig
│   ├── method: "none"  # or platt_scaling, isotonic, temperature_scaling
│   ├── calibration_cv_folds: 5
│   └── smooth_calibration: False
├── evaluation: EvaluationConfig
│   ├── metrics: ["auroc", "accuracy", "f1", "precision", "recall"]
│   ├── compute_calibration_error: True
│   ├── plot_roc_curves: True
│   ├── plot_calibration_curves: True
│   ├── plot_feature_importance: True
│   └── output_dir: "results/"
└── tracking: TrackingConfig
    ├── tracker_type: "csv"  # or mlflow, wandb, none
    ├── log_dir: "logs/"
    ├── log_level: "INFO"
    ├── log_to_file: True
    ├── log_frequency: 100
    ├── track_hyperparameters: True
    ├── track_metrics: True
    └── track_data_summary: True
```

### 3.2 Baseline Configuration Specifics

**From [examples/config_baseline.yaml](examples/config_baseline.yaml):**

```yaml
# Baseline = Simple Logistic Regression Pipeline
# Expected AUROC: 0.80-0.82

Data:
  - 5-fold stratified cross-validation
  - 20% test split
  - Random seed 42

Imputation:
  - mean strategy (numeric) + mode (categorical)
  - Drop columns if >50% missing
  - KNN backup with 5 neighbors

Encoding:
  - Opinion features (Likert 1-5): ordinal (preserve order)
  - Behavioral (binary): ordinal (0/1 identity)
  - Medical (mixed): ordinal
  - Demographics: one-hot with drop_first=True
  - Household: ordinal (numeric)
  - Geographic: ordinal or one-hot
  - Concern/Knowledge (ordinal): ordinal
  - Employment: one-hot with drop_first=True

Model:
  - Logistic Regression
  - L2 regularization (penalty: l2)
  - C=1.0 (inverse regularization strength)
  - Balanced class weights
  - max_iter=1000

Training:
  - Stratified k-fold with 5 folds
  - Class weight strategy: balanced
  - No hyperparameter search
  - No SMOTE
  - No threshold tuning

Calibration:
  - Platt Scaling (logistic regression on predictions)
  - 3 calibration folds

Evaluation:
  - Primary metric: AUROC (mean of h1n1 + seasonal)
  - Secondary: calibration_error, confusion_matrix, brier_score
  - Generate ROC curves, calibration plots, feature importance

Tracking:
  - CSV logging to experiments_baseline.csv
  - Track: auroc_h1n1, auroc_seasonal, auroc_mean
  - Track: calibration_error, brier_score per vaccine

Submission:
  - Output: ./submissions/submission_baseline.csv
  - Format: respondent_id, h1n1_vaccine, seasonal_vaccine (probabilities)
```

---

## Part 4: Feature Architecture

### 4.1 Feature Groups (35 Total Features)

#### Group 1: Opinions (5 features) - Ordinal 1-5
```
opinion_h1n1_vacc_effective
opinion_h1n1_risk
opinion_h1n1_sick_from_vacc
opinion_seas_vacc_effective
opinion_seas_risk
```
→ **Encoding**: Ordinal (preserve 1-5 ordering)

#### Group 2: Behavioral (7 features) - Binary 0/1
```
behavioral_antiviral_meds
behavioral_avoidance
behavioral_face_mask
behavioral_large_gatherings
behavioral_outside_home
behavioral_touch_face
behavioral_hand_washing  (note: not in all lists)
```
→ **Encoding**: Identity (already 0/1)

#### Group 3: Medical Recommendations (4 features) - Mixed
```
doctor_recc_h1n1              (binary 0/1)
doctor_recc_seasonal          (binary 0/1)
chronic_med_condition         (binary 0/1)
health_worker                 (binary 0/1)
health_insurance              (binary 0/1)
```
→ **Encoding**: Identity (all 0/1)

#### Group 4: Demographics (8 features) - Categorical
```
age_group                     (categorical: 18-34, 35-49, 50-64, 65+)
education                     (categorical: < 12 yrs, 12 yrs, some college, college+)
race                          (categorical: white, black, hispanic, other)
sex                           (binary: male, female)
income_poverty                (categorical: <=200%, >200%)
marital_status                (categorical: married, single, divorced, etc.)
rent_or_own                   (categorical: rent, own)
employment_status             (categorical: employed, unemployed, not in labor force)
```
→ **Encoding**: One-hot (high cardinality) or target encoding

#### Group 5: Household (2 features) - Numeric
```
household_adults              (integer: 0-3+, top-coded at 3)
household_children            (integer: 0-3+, top-coded at 3)
```
→ **Encoding**: Identity (numeric, already scaled)

#### Group 6: Geographic (2 features) - Categorical
```
hhs_geo_region                (10 regions: 1-10)
census_msa                    (4 categories: metro, non-metro, etc.)
```
→ **Encoding**: Ordinal or one-hot

#### Group 7: Concern & Knowledge (2 features) - Ordinal
```
h1n1_concern                  (ordinal: 1-4, low to high concern)
h1n1_knowledge                (ordinal: 1-4, low to high knowledge)
```
→ **Encoding**: Ordinal (preserve 1-4 ordering)

#### Group 8: Employment (2 features) - Categorical with NaN
```
employment_industry           (categorical with many categories + NaN for non-employed)
employment_occupation         (categorical with many categories + NaN for non-employed)
```
→ **Encoding**: One-hot or target encoding, handle NaN as category

### 4.2 Targets (2 Binary Features)

```
h1n1_vaccine                  (0 or 1) - Did respondent receive H1N1 vaccine?
seasonal_vaccine              (0 or 1) - Did respondent receive seasonal vaccine?
```

**Multilabel Property**: Each respondent can have any combination:
- (0, 0): Neither vaccine
- (0, 1): Seasonal only
- (1, 0): H1N1 only
- (1, 1): Both vaccines

---

## Part 5: Key Classes & Interfaces

### 5.1 Data Loading

```python
# Abstract Interface
class DataLoader(ABC):
    def load_train() -> Tuple[pd.DataFrame, pd.DataFrame]  # X_train, y_train
    def load_test() -> Tuple[pd.DataFrame, pd.Series]      # X_test, respondent_ids
    def create_splits(X_train, y_train) -> List[DataSplit] # k-fold splits

# Data Container
@dataclass
class DataSplit:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_val: pd.DataFrame
    y_val: pd.DataFrame
    X_test: pd.DataFrame
    respondent_ids_test: pd.Series

# Validation Result
@dataclass
class DataValidationResult:
    is_valid: bool
    n_samples: int
    n_features: int
    missing_by_feature: Dict[str, float]
    class_distribution: Dict[str, Dict[str, float]]
    feature_types: Dict[str, str]
    issues: List[str]
    warnings: List[str]
```

### 5.2 Imputation Strategies

```python
# Abstract Interface
class ImputationStrategy(ABC):
    def fit(X: pd.DataFrame) -> ImputationStrategy
    def transform(X: pd.DataFrame) -> pd.DataFrame
    def fit_transform(X: pd.DataFrame) -> pd.DataFrame

# Concrete Strategies (Stubs)
class DropRowsImputation(ImputationStrategy): pass
class DropColumnsImputation(ImputationStrategy): pass
class MeanImputation(ImputationStrategy): pass
class ModeImputation(ImputationStrategy): pass
class KNNImputation(ImputationStrategy): pass
class MICEImputation(ImputationStrategy): pass
class FlagAsMissingImputation(ImputationStrategy): pass
```

### 5.3 Feature Encoders

```python
# Abstract Interface
class FeatureEncoder(ABC):
    def fit(X: pd.DataFrame, y: Optional[pd.Series] = None) -> FeatureEncoder
    def transform(X: pd.DataFrame) -> pd.DataFrame
    def fit_transform(X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame
    def get_feature_names() -> List[str]

# Concrete Strategies (Stubs)
class OrdinalEncoder(FeatureEncoder): pass
class OneHotEncoder(FeatureEncoder): pass
class TargetEncoder(FeatureEncoder): pass
class InteractionEncoder(FeatureEncoder): pass
class PolynomialEncoder(FeatureEncoder): pass
```

### 5.4 Models

```python
# Abstract Interface
class BaseModel(ABC):
    def fit(X: pd.DataFrame, y: pd.Series) -> BaseModel
    def predict_proba(X: pd.DataFrame) -> np.ndarray  # shape (n_samples, 2)
    def predict(X: pd.DataFrame) -> np.ndarray         # shape (n_samples,) 0/1
    def get_feature_importance() -> pd.DataFrame

# Concrete Models (Stubs)
class LogisticRegressionModel(BaseModel): pass
class XGBoostModel(BaseModel): pass
class LightGBMModel(BaseModel): pass
class RandomForestModel(BaseModel): pass
```

### 5.5 Training Engine

```python
# Results Containers
@dataclass
class FoldResults:
    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    y_val_true: np.ndarray
    y_val_proba: np.ndarray
    metrics: Dict[str, float]
    model: Optional[Any] = None

@dataclass
class CVResults:
    fold_results: List[FoldResults]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    best_fold_id: int
    best_model: Optional[Any] = None

# Engine Class
class TrainingEngine:
    def run_cv(X: pd.DataFrame, y: pd.DataFrame) -> CVResults
    def hyperparameter_search(X, y, param_grid, cv_folds) -> Dict[str, Any]
    def get_fold_predictions(X_test, return_std=False) -> np.ndarray
```

### 5.6 Evaluation

```python
class Evaluator:
    @staticmethod
    def compute_auroc(
        y_true_h1n1: np.ndarray,
        y_true_seasonal: np.ndarray,
        y_pred_h1n1: np.ndarray,
        y_pred_seasonal: np.ndarray,
    ) -> Tuple[float, float, float]:  # ✓ IMPLEMENTED
        """Returns (auroc_h1n1, auroc_seasonal, auroc_mean)"""
        # Input validation (shape, value ranges)
        # Compute roc_auc_score() for each vaccine
        # Return tuple

    @staticmethod
    def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Returns TP, FP, TN, FN, sensitivity, specificity, PPV, NPV"""

    @staticmethod
    def calibration_error(...) -> Dict[str, float]:
        """Returns ECE, MCE, Brier score"""

    def get_diagnostics(...) -> Dict[str, Any]:
        """Complete diagnostic report"""
```

### 5.7 Calibration

```python
# Abstract Interface
class CalibratorInterface(ABC):
    def fit(y_true: np.ndarray, y_proba: np.ndarray) -> CalibratorInterface
    def transform(y_proba: np.ndarray) -> np.ndarray
    def fit_transform(y_true, y_proba) -> np.ndarray
    def get_calibration_error(y_true, y_proba_calibrated) -> Dict[str, float]

# Concrete Strategies (Stubs)
class NoCalibration(CalibratorInterface): pass
class PlattScalingCalibrator(CalibratorInterface): pass
class IsotonicCalibrator(CalibratorInterface): pass
class TemperatureScalingCalibrator(CalibratorInterface): pass

# Factory
def create_calibrator(method: str) -> CalibratorInterface:
    """Factory function mapping config.calibration.method → calibrator instance"""
```

### 5.8 Prediction

```python
class PredictionEngine(ABC):
    @abstractmethod
    def predict_test_set(
        X_test: pd.DataFrame,
        model: object,
        preprocessing_pipeline: Optional[object] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:  # (y_pred_h1n1, y_pred_seasonal)
        """Generate probability predictions on test set"""

    @staticmethod
    def format_submission(
        respondent_ids: np.ndarray,
        y_pred_h1n1: np.ndarray,
        y_pred_seasonal: np.ndarray,
    ) -> pd.DataFrame:
        """Create submission DataFrame with columns: [respondent_id, h1n1_vaccine, seasonal_vaccine]"""

    @staticmethod
    def validate_submission(
        submission_df: pd.DataFrame,
        submission_template_path: Optional[str] = None,
    ) -> bool:
        """Validate format: columns, types, ranges [0,1], no NaN/inf"""

    @staticmethod
    def save_submission(
        submission_df: pd.DataFrame,
        output_path: str,
        validate_before_save: bool = True,
    ) -> None:
        """Save to CSV"""
```

### 5.9 Experiment Tracking

```python
@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    model_type: str
    config_json: str
    hyperparameters_json: str
    auroc_h1n1: Optional[float] = None
    auroc_seasonal: Optional[float] = None
    auroc_mean: Optional[float] = None
    h1n1_sensitivity: Optional[float] = None
    h1n1_specificity: Optional[float] = None
    h1n1_ppv: Optional[float] = None
    seasonal_sensitivity: Optional[float] = None
    seasonal_specificity: Optional[float] = None
    seasonal_ppv: Optional[float] = None
    h1n1_ece: Optional[float] = None
    seasonal_ece: Optional[float] = None
    notes: Optional[str] = None
    status: str = 'in_progress'

class ExperimentTracker(ABC):
    @abstractmethod
    def log_run(run_id, model_type, config, hyperparameters, metrics) -> None
    @abstractmethod
    def filter_by_model_type(model_type) -> List[RunRecord]
    @abstractmethod
    def rank_by_auroc(limit=10) -> List[RunRecord]
    @abstractmethod
    def export() -> pd.DataFrame

class CSVExperimentLogger(ExperimentTracker):
    """Logs to CSV file (stub)"""
```

---

## Part 6: Missing Pieces & Implementation Requirements

### 6.1 Critical Gaps (Must Implement)

1. **Data Loader** - Core dependency for everything
   - `CSVDataLoader.load_train()` - Load and merge features + labels
   - `CSVDataLoader.load_test()` - Load test features
   - `CSVDataLoader.create_splits()` - Stratified k-fold respecting both targets

2. **Imputation Strategies** - At least "mean" for baseline
   - Implement fit/transform for MeanImputation
   - Pattern: Learn column means in fit(), apply in transform()

3. **Feature Encoders** - At least ordinal + one-hot for baseline
   - OrdinalEncoder: Preserve feature order
   - OneHotEncoder: Binary indicators with drop_first option
   - TargetEncoder: For high-cardinality categoricals

4. **Models** - LogisticRegression for baseline
   - Wrap sklearn.linear_model.LogisticRegression
   - Implement fit(), predict_proba(), get_feature_importance()

5. **Training Engine** - run_cv() orchestration
   - Create StratifiedKFold splits
   - Train model on each fold
   - Compute metrics per fold
   - Return CVResults with aggregated stats

6. **Prediction Engine** - Test set inference
   - Apply learned imputation/encoding to test set
   - Get predictions from trained model(s)
   - Format submission CSV

7. **Calibrators** - At least NoCalibration, Platt for baseline
   - NoCalibration: Passthrough
   - PlattScalingCalibrator: Fit logistic regression on probabilities

8. **Experiment Logger** - CSV tracking
   - Append RunRecord to CSV file
   - Query/filter runs

### 6.2 Optional Enhancements (Nice to Have)

- Hyperparameter search (grid, random, Bayesian)
- SMOTE for class imbalance
- Threshold tuning
- Early stopping
- Advanced imputation (KNN, MICE)
- Target encoding for categoricals
- Polynomial/interaction features
- Neural network models
- Visualization (ROC curves, calibration plots)
- MLflow or Weights & Biases integration

---

## Part 7: Multilabel Specifics

### 7.1 Multilabel Problem Setup

**Two independent binary targets**:
- `h1n1_vaccine`: 0 or 1
- `seasonal_vaccine`: 0 or 1

**NOT multiclass** (vaccine type 0/1/2) - **separate predictions per vaccine**

### 7.2 Stratification Challenge

Standard StratifiedKFold doesn't work on 2D targets. Solution:
- Combine labels into 4 categories: (0,0), (0,1), (1,0), (1,1)
- Use StratifiedKFold on combined labels
- Ensures each fold has all 4 label combinations

```python
# Pseudo-code
y_combined = y[:, 0].astype(str) + y[:, 1].astype(str)  # "00", "01", "10", "11"
skf = StratifiedKFold(n_splits=5)
for train_idx, val_idx in skf.split(X, y_combined):
    # train/val split respects both vaccine distributions
```

### 7.3 Metrics Aggregation

**Primary evaluation metric**: Mean AUC
$$\text{AUROC}_{\text{mean}} = \frac{\text{AUC}_{H1N1} + \text{AUC}_{\text{seasonal}}}{2}$$

**Per-vaccine metrics** (for analysis):
- $\text{AUC}_{H1N1}$ - for H1N1 vaccine only
- $\text{AUC}_{\text{seasonal}}$ - for seasonal vaccine only

### 7.4 Training Strategy Options

**Option A: Two Independent Models** (Simple)
- Train model 1 on X → h1n1_vaccine
- Train model 2 on X → seasonal_vaccine
- Predictions are independent

**Option B: Single Multilabel Model** (Flexible)
- Train one model that outputs 2 probabilities
- Each output layer trained on separate targets
- Allows shared representation learning

**Current design supports Option A** (two independent binary classifiers)

---

## Part 8: Expected Usage

### 8.1 Running the Pipeline

```bash
# Load baseline config
python main.py --config examples/config_baseline.yaml --run-name "baseline_001" --verbose

# With custom seed for reproducibility
python main.py --config examples/config_baseline.yaml --seed 123

# Use default config (all defaults)
python main.py --verbose

# Run XGBoost config
python main.py --config examples/config_xgboost.yaml --run-name "xgboost_exp_001"
```

### 8.2 Argument Parsing

```
--config PATH              Path to YAML config file (optional)
--run-name NAME            Name for this run (default: default_run)
--verbose                  Enable DEBUG logging (default: INFO)
--seed INT                 Random seed (default: 42)
```

### 8.3 Output Artifacts

```
logs/
  baseline_pipeline.log          # Execution log
  
results/
  roc_curves.png                 # Visualizations
  calibration_curve.png
  feature_importance.png
  
submissions/
  submission_baseline.csv        # Test predictions
                                 # Columns: respondent_id, h1n1_vaccine, seasonal_vaccine

experiments_baseline.csv          # Tracking log
  # Columns: run_id, timestamp, model_type, auroc_h1n1, auroc_seasonal, auroc_mean, ...
```

---

## Part 9: Testing Strategy

### 9.1 Unit Test Targets

| Module | Test Cases |
|--------|-----------|
| `config.py` | Load YAML/JSON, dataclass validation, from_dict/to_dict roundtrip |
| `data/loader.py` | Load CSVs, validate shapes, create stratified splits |
| `imputation.py` | Mean/mode imputation, fit/transform consistency, handle NaN |
| `encoding.py` | Ordinal/one-hot encoding, fit/transform consistency, feature names |
| `models.py` | Model fit, predict_proba shape/ranges, feature importance |
| `training.py` | CV splits, metrics computation, fold aggregation |
| `evaluation.py` | AUROC computation (✓ partially), confusion matrix |
| `calibration.py` | Calibrator fit/transform, probability ranges |
| `prediction.py` | Submission formatting, validation, save/load |
| `tracking.py` | Log run, filter, rank, export |

### 9.2 Integration Tests

- Full pipeline execution with small sample data
- Config load → data load → preprocessing → training → evaluation
- Submission generation and validation

---

## Part 10: Summary Table

### Implementation Status by Module

| Module | File | Status | Key Classes | Completeness |
|--------|------|--------|------------|--------------|
| Config | `src/config.py` | ✓ Complete | DataConfig, PipelineConfig, from_yaml, from_json | 100% |
| Data Loading | `src/data/loader.py` | 🔲 Stub | DataLoader (ABC), CSVDataLoader | 20% |
| Imputation | `src/preprocessing/imputation.py` | 🔲 Stub | ImputationStrategy (ABC), 8+ concrete stubs | 10% |
| Encoding | `src/preprocessing/encoding.py` | 🔲 Stub | FeatureEncoder (ABC), FEATURE_GROUPS dict | 20% |
| Models | `src/models/factory.py` | 🔲 Stub | BaseModel (ABC), ModelFactory, 4 model stubs | 15% |
| Training | `src/training/engine.py` | 🔲 Stub | TrainingEngine, FoldResults, CVResults | 20% |
| Evaluation | `src/evaluation/metrics.py` | ◐ Partial | Evaluator, compute_auroc (✓) | 30% |
| Evaluation Plots | `src/evaluation/plots.py` | 🔲 Stub | Plot functions | 0% |
| Calibration | `src/calibration/calibrator.py` | 🔲 Stub | CalibratorInterface, 4 calibrator stubs | 15% |
| Prediction | `src/prediction/predictor.py` | 🔲 Stub | PredictionEngine, format/validate submission | 20% |
| Tracking | `src/tracking/logger.py` | 🔲 Stub | ExperimentTracker, CSVExperimentLogger, RunRecord | 15% |
| Utils | `src/utils/` | 🔲 Stub | Helpers, logging, validation | 10% |
| Main | `main.py` | ◐ Partial | load_config (✓), run_pipeline (skeleton), main (✓) | 60% |
| Baseline Config | `examples/config_baseline.yaml` | ✓ Complete | Full config for baseline | 100% |

**Overall Completion**: ~30% (infrastructure 100%, implementations 0%)

---

## Conclusion

The Flu Shot ML pipeline is **well-architected with comprehensive documentation and clear stub templates**. The next step is implementing the stubbed classes following the sklearn fit/transform pattern. The configuration and orchestration layers are production-ready; execution requires filling in the strategy implementations.

The baseline config specifies a simple logistic regression + platt scaling approach with mean imputation and one-hot encoding—perfect for validating the pipeline before trying advanced techniques.
