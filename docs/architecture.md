# ML Pipeline Architecture

This document describes the modular architecture of the flu shot prediction ML pipeline.

## System Overview

The pipeline follows a component-based architecture with clear separation of concerns. Each component has a well-defined interface (Abstract Base Class) and one or more concrete implementations.

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LOADING LAYER                        │
│  src.data.CSVDataLoader → [train_features.csv, train_labels.csv]│
│                                    ↓                              │
├─────────────────────────────────────────────────────────────────┤
│                    PREPROCESSING LAYER                           │
│  Imputation          →  Encoding                                │
│  (7 strategies)          (5 strategies)                          │
│  DropRows            → Ordinal/OneHot/Target                    │
│  DropColumns         → Interaction/Polynomial                    │
│  Mean, Mode, KNN,    →                                           │
│  MICE, FlagAsMissing →                                           │
│                                    ↓                              │
├─────────────────────────────────────────────────────────────────┤
│                     MODEL TRAINING LAYER                         │
│  ModelFactory.create_model()                                    │
│  ├─ LogisticRegressionModel                                    │
│  ├─ RandomForestModel                                          │
│  ├─ XGBoostModel                                               │
│  └─ LightGBMModel                                              │
│         ↓                                                        │
│  TrainingEngine.run_cv()                                       │
│  ├─ Stratified K-Fold splits                                  │
│  ├─ Train on each fold                                        │
│  ├─ Hyperparameter search (grid/random/Bayesian)              │
│  └─ Class imbalance handling (weights/SMOTE/threshold)        │
│                                    ↓                              │
├─────────────────────────────────────────────────────────────────┤
│                   CALIBRATION LAYER                              │
│  create_calibrator(method)                                      │
│  ├─ NoCalibration (baseline)                                   │
│  ├─ PlattScalingCalibrator                                    │
│  ├─ IsotonicCalibrator                                        │
│  └─ TemperatureScalingCalibrator                              │
│                                    ↓                              │
├─────────────────────────────────────────────────────────────────┤
│                   EVALUATION LAYER                               │
│  Evaluator class:                                               │
│  ├─ compute_auroc() → ROC AUC for each vaccine                │
│  ├─ confusion_matrix() → TP, FP, TN, FN                        │
│  ├─ calibration_error() → ECE, MCE, Brier score               │
│  └─ get_diagnostics() → All metrics combined                   │
│                                    ↓                              │
│  Visualization (evaluation/plots.py):                           │
│  ├─ plot_roc_curves() → Dual ROC curves                        │
│  ├─ plot_calibration_curve() → Calibration plot               │
│  ├─ plot_feature_importance() → Feature importance chart       │
│  └─ plot_prediction_confidence() → Prediction distribution    │
│                                    ↓                              │
├─────────────────────────────────────────────────────────────────┤
│                   EXPERIMENT TRACKING                            │
│  ExperimentTracker (ABC)                                        │
│  └─ CSVExperimentLogger (implementation)                        │
│     ├─ log_run() → Store run results                           │
│     ├─ get_results() → Retrieve all runs                       │
│     ├─ rank_by_auroc() → Sort by performance                   │
│     ├─ filter_by_model_type() → Query runs                     │
│     └─ export() → Return as DataFrame                          │
│                                    ↓                              │
├─────────────────────────────────────────────────────────────────┤
│                  TEST SET PREDICTION                             │
│  PredictionEngine:                                              │
│  ├─ predict_test_set() → Apply preprocessing + model           │
│  ├─ format_submission() → respondent_id, h1n1, seasonal       │
│  ├─ validate_submission() → Check format                       │
│  └─ save_submission() → Write to CSV                           │
│                                    ↓                              │
│                   SUBMISSION CSV OUTPUT                          │
│           submission.csv (respondent_id, h1n1_vaccine,         │
│                          seasonal_vaccine)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Component Dependency Graph

```
┌──────────────────┐
│ src.config       │  ← All components read from PipelineConfig
└─────────┬────────┘
          │
    ┌─────┴──────────────────────────────────────────────┐
    │                                                      │
    ↓                                                      ↓
┌─────────────┐                              ┌──────────────────────┐
│ src.data    │                              │ src.preprocessing    │
│ CSVDataLoader│─────────────────┐          │ (Imputation +        │
└─────────────┘                  │          │  Encoding)           │
                                  │          └──────────────────────┘
                                  ↓                    ↓
                          ┌──────────────────────────┐
                          │ X_train (n, m)           │
                          │ y_train (n, 2)           │
                          │ X_test (n_test, m)       │
                          └──────────────────────────┘
                                  ↓
                          ┌──────────────────────────┐
                          │ src.models.ModelFactory  │
                          │ src.training.            │
                          │ TrainingEngine           │
                          └──────────────────────────┘
                                  ↓
                          ┌──────────────────────────┐
                          │ cv_predictions (fold ×   │
                          │ samples × vaccines)      │
                          └──────────────────────────┘
                                  ↓
                          ┌──────────────────────────┐
                          │ src.calibration.         │
                          │ CalibratorInterface      │
                          └──────────────────────────┘
                                  ↓
                          ┌──────────────────────────┐
                          │ calibrated_predictions   │
                          │ (samples × vaccines)     │
                          └──────────────────────────┘
           ┌────────────────────┬────────────────────┐
           ↓                    ↓                    ↓
    ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐
    │ src.evaluation │  │ src.tracking.  │  │ src.prediction  │
    │ (Metrics +     │  │ CSVExperiment  │  │ PredictionEngine│
    │  Plots)        │  │ Logger         │  │                 │
    └────────────────┘  └────────────────┘  └─────────────────┘
           ↓                    ↓                    ↓
      [Plots &            [experiments.csv]   [submission.csv]
       Metrics]
```

## Data Contracts

### Training Data
```
X_train: DataFrame
  Shape: (n_samples, 35)
  Columns: [age_group, opinion_h1n1_risk, doctor_recc_h1n1, ...]
  Types: Numeric (int/float), Categorical (object)
  Missing: Can have NaN values

y_train: DataFrame
  Shape: (n_samples, 2)
  Columns: [h1n1_vaccine, seasonal_vaccine]
  Values: 0 or 1 (binary)
  Missing: None expected
```

### Predictions
```
Predictions: ndarray or DataFrame
  Shape: (n_samples, 2)
  Values: Float in [0.0, 1.0] (probabilities)
  Interpretation: 
    - Column 0: P(received H1N1 vaccine)
    - Column 1: P(received seasonal vaccine)
```

### Evaluation Metric
```
Metric: ROC AUC (Area Under the Receiver Operating Characteristic Curve)
Computed as: mean(AUC_h1n1, AUC_seasonal)
Range: [0.0, 1.0]
Better: Higher values
Interpretation: Probability that model ranks random positive higher than random negative
```

## Module Descriptions

### 1. src/data/ - Data Loading
- **Interface**: `DataLoader` (ABC)
- **Implementations**:
  - `CSVDataLoader`: Loads from CSV files
- **Responsibilities**:
  - Load training features and labels
  - Load test features
  - Create cross-validation splits
  - Validate data structure and types

### 2. src/preprocessing/ - Feature Preprocessing
- **Imputation Strategies**:
  - `DropRowsImputation`: Remove rows with missing values
  - `DropColumnsImputation`: Remove columns with >threshold missing
  - `MeanImputation`: Fill numeric with mean
  - `ModeImputation`: Fill with mode
  - `KNNImputation`: K-nearest neighbors imputation
  - `MICEImputation`: Multivariate imputation by chained equations
  - `FlagAsMissingImputation`: Create boolean flag feature for missing

- **Encoding Strategies**:
  - `OrdinalEncoder`: Convert categorical to integers
  - `OneHotEncoder`: Create dummy variables
  - `TargetEncoder`: Target mean encoding
  - `InteractionEncoder`: Create interaction features
  - `PolynomialEncoder`: Create polynomial features

### 3. src/models/ - ML Models
- **Factory Pattern**: `ModelFactory.create_model(type, params)`
- **Implementations**:
  - `LogisticRegressionModel`: Baseline linear model
  - `RandomForestModel`: Ensemble tree model
  - `XGBoostModel`: Gradient boosting model
  - `LightGBMModel`: Light gradient boosting model

### 4. src/training/ - Model Training
- **Class**: `TrainingEngine`
- **Responsibilities**:
  - Cross-validation setup (stratified k-fold)
  - Model training on each fold
  - Hyperparameter search (grid/random/Bayesian)
  - Threshold tuning for class imbalance
  - Aggregate and report fold results

### 5. src/calibration/ - Probability Calibration
- **Interface**: `CalibratorInterface` (ABC)
- **Implementations**:
  - `NoCalibration`: Passthrough (baseline)
  - `PlattScalingCalibrator`: Logistic regression fit
  - `IsotonicCalibrator`: Isotonic regression fit
  - `TemperatureScalingCalibrator`: Single temperature parameter

### 6. src/evaluation/ - Model Evaluation
- **Metrics Computation**:
  - ROC AUC for each vaccine
  - Confusion matrix (TP, FP, TN, FN)
  - Calibration error (ECE, MCE, Brier score)
  - Sensitivity, specificity, F1 score
  
- **Visualization**:
  - ROC curves for both vaccines
  - Calibration plots
  - Feature importance charts
  - Prediction confidence distributions

### 7. src/tracking/ - Experiment Tracking
- **Interface**: `ExperimentTracker` (ABC)
- **Implementation**: `CSVExperimentLogger`
- **Stored Info**:
  - Run ID and timestamp
  - Configuration (all pipeline settings)
  - Hyperparameters
  - Metrics (AUROC, calibration, sensitivity/specificity)
  - Model type and version

### 8. src/prediction/ - Test Set Prediction
- **Class**: `PredictionEngine`
- **Responsibilities**:
  - Apply same preprocessing to test data
  - Generate predictions using trained model
  - Apply calibration
  - Format as submission CSV
  - Validate submission format

### 9. src/utils/ - Utilities
- **Modules**:
  - `logging.py`: Centralized logging configuration
  - `validation.py`: Data validation helpers
  - `metrics.py`: Helper metric computations
  - `plots.py`: Matplotlib/seaborn wrappers
  - `helpers.py`: General utilities (stratified splits, class weights, feature groups)

## Configuration Flow

```
PipelineConfig (src/config.py)
├── DataConfig: File paths, CV folds, random seed
├── ImputationConfig: Strategy name, parameters
├── EncodingConfig: Per-feature-group encoding choices
├── ModelConfig: Model type, hyperparameters
├── TrainingConfig: CV strategy, search method, class weight handling
├── CalibrationConfig: Calibration method, parameters
├── EvaluationConfig: Evaluation metrics, thresholds
├── TrackingConfig: Log file path, run metadata
├── PredictionConfig: Output directory, format options
└── LoggingConfig: Log level, file path
```

## Execution Flow (main.py)

```python
# 1. Load config (from YAML or default)
config = load_config(config_path)

# 2. Data loading and validation
loader = CSVDataLoader(config.data)
X_train, y_train = loader.load_train()
X_test = loader.load_test()

# 3. Imputation
imputer = create_imputation_strategy(config.imputation)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 4. Feature encoding
encoder = create_feature_encoder(config.encoding)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# 5. Model training (cross-validation)
engine = TrainingEngine(config.training)
best_model, cv_predictions = engine.run_cv(X_train, y_train)

# 6. Probability calibration
calibrator = create_calibrator(config.calibration)
calibrator.fit(cv_predictions, y_train)
calibrated_preds = calibrator.transform(cv_predictions)

# 7. Evaluation
evaluator = Evaluator()
metrics = evaluator.get_diagnostics(y_train, calibrated_preds)

# 8. Experiment tracking
tracker = CSVExperimentLogger(config.tracking.log_path)
tracker.log_run(run_id, config.model.type, config, metrics)

# 9. Test predictions
test_preds = best_model.predict_proba(X_test)
test_preds = calibrator.transform(test_preds)

# 10. Format submission
predictor = DefaultPredictionEngine()
submission = predictor.format_submission(test_respondent_ids, test_preds)
predictor.save_submission(submission)
```

## Extension Points

### Adding a New Model
1. Create `src/models/my_model.py`
2. Inherit from `BaseModel`
3. Implement: `fit()`, `predict_proba()`, `get_feature_importance()`, `get_params()`, `set_params()`
4. Register in `ModelFactory.create_model()`

### Adding a New Imputation Strategy
1. Create class in `src/preprocessing/imputation.py`
2. Inherit from `ImputationStrategy`
3. Implement: `fit()`, `transform()`
4. Update configuration to reference it

### Adding a New Encoding
1. Create class in `src/preprocessing/encoding.py`
2. Inherit from `FeatureEncoder`
3. Implement: `fit()`, `transform()`, `get_feature_names()`

### Adding a New Metric
1. Create function in `src/utils/metrics.py`
2. Add to `Evaluator.get_diagnostics()`
3. Update tracking schema to include new metric

## File Statistics

- **Total Python Files**: 33
- **Core Modules**: 9 (data, preprocessing, models, training, calibration, evaluation, tracking, prediction, utils)
- **Stub Functions**: ~50 (ready for implementation)
- **Lines of Code**: ~3,500 (structure + docstrings)
- **Configuration Files**: PipelineConfig with 9 sub-configs
- **Dependencies**: 11 required, 12 development

---

**Last Updated**: January 7, 2026  
**Architecture Version**: 1.0  
**Status**: Implementation-ready
