## Project Overview

This is a machine learning competition project from [DrivenData](https://www.drivendata.org/competitions/66/flu-shot-learning/) focused on **multilabel classification** to predict H1N1 and seasonal flu vaccine uptake based on survey data from the National 2009 H1N1 Flu Survey.

### Key Problem Characteristics

- **Multilabel, not multiclass**: Each respondent can receive neither, one, or both vaccines independently
- **Two target variables**: `h1n1_vaccine` and `seasonal_vaccine` (both binary: 0 or 1)
- **Evaluation metric**: ROC AUC (mean of scores for each vaccine) - predictions must be **probabilities (0.0-1.0), not binary labels**
- **Feature set**: 35 features covering demographics, opinions, health behaviors, medical recommendations, and health status
- **Data source**: CDC's National 2009 H1N1 Flu Survey (survey responses, not clinical data)

## Architecture & Data Flow

This is a **modular, component-based pipeline** following a 10-stage orchestration pattern:
`Data Loading → Imputation → Encoding → Training → Calibration → Evaluation → Visualization → Tracking → Prediction → Submission`

**Key architectural patterns**:
- **Strategy Pattern**: Abstract base classes (ABC) define interfaces for swappable components (e.g., `ImputationStrategy`, `CalibratorInterface`)
- **Factory Pattern**: `ModelFactory.create_model()` instantiates models by string name
- **Configuration System**: YAML-driven pipeline configuration using dataclasses in [src/config.py](src/config.py)
- **Experiment Tracking**: CSV-based logging to `experiments_*.csv` files for comparing runs

### Component Structure

```
src/
├── config.py                    # PipelineConfig dataclass (loads from YAML)
├── data/loader.py               # CSVDataLoader for train/test data
├── preprocessing/
│   ├── imputation.py            # 11 imputation strategies (TypeBased, KNN, MICE, etc.)
│   └── encoding.py              # FeatureEncoder (ordinal/onehot/interaction/polynomial)
├── models/factory.py            # ModelFactory + model wrappers (LR, XGBoost, RF, LightGBM)
├── training/engine.py           # TrainingEngine (stratified CV, SMOTE, threshold tuning)
├── calibration/calibrator.py   # 4 calibrators (Platt, Isotonic, Temperature, None)
├── evaluation/
│   ├── metrics.py               # Evaluator class (AUROC, ECE, Brier, confusion matrix)
│   └── plots.py                 # Visualization (ROC curves, calibration plots)
├── tracking/logger.py           # CSVExperimentLogger for run tracking
└── prediction/predictor.py     # DefaultPredictionEngine (test set inference + submission)
```

**Entry point**: [main.py](main.py) - orchestrates all stages via `run_pipeline(config, run_name)`

### Development Setup

This project uses a Python virtual environment (`.venv`) to isolate dependencies. Always activate the virtual environment before working:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Experiments

The pipeline is **fully configured via YAML files**. All experiments are run from the command line:

```bash
# Run a baseline experiment
python main.py --config examples/config_baseline.yaml

# Run with a custom experiment name
python main.py --config examples/config_xgboost.yaml --run-name "my_experiment"

# Run with verbose logging
python main.py --config examples/config_type_based_lr_enhanced.yaml --verbose
```

**Available configs** in [examples/](examples/):
- `config_baseline.yaml` — Logistic Regression with mean imputation (AUROC ~0.84)
- `config_type_based_basic.yaml` — Type-based imputation with LR (AUROC ~0.844)
- `config_type_based_lr_enhanced.yaml` — LR + interactions + polynomials (AUROC ~0.85-0.86)
- `config_type_based_boosting.yaml` — XGBoost with Bayesian hyperparameter search (AUROC ~0.86-0.88)
- `config_type_based_knn.yaml` — KNN imputation variant
- `config_mice_typebased.yaml` — MICE imputation variant

**Outputs**:
- `experiments_*.csv` — Experiment logs with AUROC, ECE, Brier scores
- `submissions/submission_*.csv` — Test set predictions for competition upload
- `logs/*_pipeline.log` — Detailed execution logs
- `results/` — ROC curves, calibration plots (if visualization enabled)

### Feature Categories

Organize feature engineering and EDA around these logical groupings:

1. **Opinions** (5 features): `opinion_h1n1_*`, `opinion_seas_*` - ordinal scales (1-5)
2. **Behavioral** (7 features): `behavioral_*` - binary preventive actions
3. **Medical** (4 features): `doctor_recc_*`, `chronic_med_condition`, `health_worker`, `health_insurance`
4. **Demographics** (8 features): `age_group`, `education`, `race`, `sex`, `income_poverty`, `marital_status`, `rent_or_own`, `employment_status`
5. **Household** (2 features): `household_adults`, `household_children` (top-coded to 3)
6. **Geographic** (2 features): `hhs_geo_region`, `census_msa`
7. **Concern & Knowledge** (2 features): `h1n1_concern`, `h1n1_knowledge` (ordinal scales)
8. **Employment** (2 features): `employment_industry`, `employment_occupation` (categorical with NaN)

### Imputation Strategies

The pipeline supports **11 imputation strategies** (all in [src/preprocessing/imputation.py](src/preprocessing/imputation.py)):

1. **TypeBasedImputation** ⭐ (recommended): Applies strategy-per-feature-type (mean for numeric/ordinal, mode for categorical)
2. **MeanImputation**: Fill missing numeric values with column means
3. **ModeImputation**: Fill missing categorical values with column modes
4. **KNNImputation**: Use k-nearest neighbors to impute based on similar samples
5. **MICEImputation**: Multiple Imputation by Chained Equations (iterative imputation)
6. **DropRowsImputation**: Remove rows with any missing values
7. **DropColumnsImputation**: Remove columns with >threshold% missing
8. **FlagAsMissingImputation**: Create binary indicator columns for missing values
9. **OrdinalStringKNNImputation**: KNN for ordinal categorical features
10. **CategoricalKNNImputation**: KNN for nominal categorical features

**Pattern**: All imputation classes inherit from `ImputationStrategy` ABC with `fit()` and `transform()` methods (scikit-learn pattern).

### Encoding Strategies

[src/preprocessing/encoding.py](src/preprocessing/encoding.py) provides `FeatureEncoder` with configurable encoding per feature group:

- **Ordinal encoding**: Preserve order for Likert scales (opinions, concern, knowledge)
- **One-hot encoding**: For categorical demographics/geography
- **Target encoding**: Encode categories by target mean (optional)
- **Interaction terms**: Create cross-products of specified feature pairs (e.g., `doctor_recc_h1n1 × h1n1_concern`)
- **Polynomial features**: Generate degree-2 or degree-3 polynomial features
- **Missing indicators**: Add binary flags for originally-missing values

**Configuration example** (from YAML):
```yaml
encoding:
  strategies:
    opinion:
      type: ordinal
    demographic:
      type: onehot
      parameters:
        drop_first: true
  interaction_terms:
    - ["doctor_recc_h1n1", "h1n1_concern"]
    - ["opinion_h1n1_vacc_effective", "h1n1_knowledge"]
  polynomial_degree: 2
```

## Critical Implementation Details

### Configuration-Driven Architecture

All pipeline behavior is controlled by YAML config files that map to dataclasses in [src/config.py](src/config.py):

- **PipelineConfig**: Top-level config composing all sub-configs
- **DataConfig**: Paths, CV folds, random seed, stratification
- **ImputationConfig**: Strategy selection + parameters
- **EncodingConfig**: Per-feature-group encoding types + advanced features
- **ModelConfig**: Model type, hyperparameters, class weights
- **TrainingConfig**: CV strategy, SMOTE, early stopping, hyperparameter search
- **CalibrationConfig**: Calibration method (none, platt, isotonic, temperature)
- **EvaluationConfig**: Metrics, visualization, output paths
- **TrackingConfig**: Experiment logging settings

**Loading configs**:
```python
from src.config import PipelineConfig
config = PipelineConfig.from_yaml("examples/config_baseline.yaml")
```

**Feature type constants** (used by TypeBasedImputation):
- `ORDINAL_COLUMNS`: Opinion scales, concern, knowledge, age, education, income
- `NOMINAL_COLUMNS`: Race, sex, employment, geography
- `BINARY_NUMERIC_COLUMNS`: Behavioral flags, health status, household counts

### Model Factory Pattern

[src/models/factory.py](src/models/factory.py) provides `ModelFactory.create_model(model_type, hyperparameters)` supporting:
- `logistic_regression`: Fast baseline, interpretable, natural probabilities
- `xgboost`: Gradient boosting, best performance (~0.86-0.88 AUROC)
- `lightgbm`: Fast boosting, memory efficient
- `random_forest`: Ensemble baseline

**All models** wrap scikit-learn/XGBoost APIs with unified interface:
- `fit(X_train, y_train, X_val=None, y_val=None)`: Train with optional validation
- `predict_proba(X)`: Return probabilities [0, 1]
- `get_feature_importance()`: Extract importance scores (if available)

### Training Engine

[src/training/engine.py](src/training/engine.py) provides `TrainingEngine` handling:

1. **Stratified K-Fold CV**: Preserves distribution of both target variables across folds
2. **Class imbalance handling**:
   - SMOTE oversampling (synthetic minority samples)
   - Class weights (`balanced`, `balanced_subsample`)
   - Threshold tuning (find optimal probability cutoff for AUROC)
3. **Hyperparameter search**:
   - Grid search (exhaustive, for small spaces)
   - Random search (sampling-based, for large spaces)
   - Bayesian optimization (Optuna, for complex spaces - XGBoost)
4. **Per-vaccine training**: Train separate models or single multilabel model

**Key method**: `TrainingEngine.run_cv(X, y, model, config)` returns CV scores, best params, fold predictions

### Calibration Methods

[src/calibration/calibrator.py](src/calibration/calibrator.py) implements 4 calibrators (all inherit from `CalibratorInterface`):

1. **NoCalibration**: Identity function (use raw model probabilities)
2. **PlattScalingCalibrator**: Logistic regression on validation scores (standard for LR/SVM)
3. **IsotonicCalibrator**: Non-parametric monotonic mapping (good for tree models)
4. **TemperatureScalingCalibrator**: Single-parameter scaling (common for neural nets/boosting)

**Usage pattern**:
```python
from src.calibration.calibrator import create_calibrator
calibrator = create_calibrator("platt")
calibrator.fit(val_probs, val_labels)
test_probs_calibrated = calibrator.calibrate(test_probs)
```

**Why calibration matters**: ROC AUC rewards well-ranked predictions, but competition also values calibration (Brier score). Calibration improves probability reliability without changing ranking.

### Experiment Tracking

[src/tracking/logger.py](src/tracking/logger.py) provides `CSVExperimentLogger` for systematic experiment tracking:

**Tracked metrics per run**:
- Run metadata: timestamp, model_type, config_json, hyperparameters
- AUROC scores: h1n1, seasonal, mean (competition metric)
- Per-vaccine metrics: sensitivity, specificity, PPV
- Calibration quality: ECE (Expected Calibration Error), Brier score
- Run status: completed, failed, in_progress

**Output files**: `experiments_{run_name}.csv` - one row per experiment, enables easy comparison via pandas

**Usage**:
```python
from src.tracking.logger import CSVExperimentLogger
logger = CSVExperimentLogger("experiments_baseline.csv")
logger.log_run(run_id, timestamp, model_type, auroc_h1n1, auroc_seasonal, ...)
results_df = logger.get_results()  # Query all runs
```

### Test Set Prediction Pipeline

[src/prediction/predictor.py](src/prediction/predictor.py) provides `DefaultPredictionEngine`:

1. **Preprocessing alignment**: Apply same imputation + encoding fitted on training data
2. **Model inference**: Generate probabilities via `model.predict_proba(X_test)`
3. **Calibration**: Apply fitted calibrator to test probabilities
4. **Submission formatting**: Create CSV with columns `[respondent_id, h1n1_vaccine, seasonal_vaccine]`
5. **Validation**: Check format, probability ranges [0.0, 1.0], no missing values

**Critical**: Test set transformations must use fitted preprocessors (no re-fitting on test data to avoid leakage)

### Multilabel Problem Setup

Since this is multilabel (not multiclass):
- Train **two separate models** (one for each vaccine) OR one model that outputs two probabilities
- Do NOT assume probabilities sum to 1 for each row
- Use `sklearn.metrics.roc_auc_score(..., average="macro")` to compute mean AUC
- Each vaccine prediction is independent

**Current implementation**: Trains two independent binary classifiers (one per vaccine) in [src/training/engine.py](src/training/engine.py)

### Component Implementation Pattern

All swappable components follow the same pattern (based on Strategy/ABC pattern from Python's `abc` module):

**Example: Adding a new imputation strategy**
```python
# In src/preprocessing/imputation.py
from abc import ABC, abstractmethod

class ImputationStrategy(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "ImputationStrategy":
        """Learn imputation parameters from training data"""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned imputation to new data"""
        pass

# New implementation
class MyCustomImputation(ImputationStrategy):
    def fit(self, X):
        self.fitted = True
        return self
    
    def transform(self, X):
        # Your logic here
        return X_imputed

# Register in factory (src/preprocessing/__init__.py)
STRATEGY_MAP = {
    'my_custom': MyCustomImputation,
    # ... other strategies
}
```

**Same pattern applies to**:
- Models: `BaseModel` ABC in [src/models/factory.py](src/models/factory.py)
- Calibrators: `CalibratorInterface` ABC in [src/calibration/calibrator.py](src/calibration/calibrator.py)
- Encoders: `FeatureEncoder` class in [src/preprocessing/encoding.py](src/preprocessing/encoding.py)

### Debugging & Diagnostics

**View experiment results**:
```bash
# Compare all experiments for a run type
cat experiments_baseline.csv | column -t -s,

# Or use pandas in Python/IPython
import pandas as pd
df = pd.read_csv("experiments_baseline.csv")
df.sort_values("auroc_mean", ascending=False)
```

**Check logs for errors**:
```bash
# Pipeline execution logs (created by main.py)
tail -f logs/baseline_pipeline.log

# Grep for errors across all logs
grep -i error logs/*.log
```

**Validate submission format**:
```python
# Check submission CSV before upload
import pandas as pd
sub = pd.read_csv("submissions/submission_baseline.csv")
assert list(sub.columns) == ["respondent_id", "h1n1_vaccine", "seasonal_vaccine"]
assert sub.shape[0] == 26708  # Test set size
assert sub["h1n1_vaccine"].between(0, 1).all()
assert sub["seasonal_vaccine"].between(0, 1).all()
```

### Project-Specific Conventions

**Naming conventions**:
- Config files: `config_{strategy}_{model}.yaml` (e.g., `config_type_based_lr_enhanced.yaml`)
- Experiment CSVs: `experiments_{strategy}.csv` (e.g., `experiments_baseline.csv`)
- Submission files: `submission_{strategy}.csv` (e.g., `submission_type_based_boosting.csv`)
- Log files: `{run_name}_pipeline.log`

**Feature type classification** (critical for type-based imputation):
```python
# In src/config.py
ORDINAL_COLUMNS = [
    "h1n1_concern", "h1n1_knowledge",  # Ordinal scales 1-4
    "opinion_h1n1_*", "opinion_seas_*",  # Likert 1-5
    "age_group", "education", "income_poverty"  # Natural order
]

NOMINAL_COLUMNS = [
    "race", "sex", "marital_status", "employment_*",
    "hhs_geo_region", "census_msa"  # No natural order
]

BINARY_NUMERIC_COLUMNS = [
    "behavioral_*",  # 7 binary preventive actions
    "doctor_recc_*", "chronic_med_condition", "health_worker",
    "household_adults", "household_children"  # Numeric 0-3
]
```

**Cross-validation approach**:
- Always use **stratified splits** to preserve both vaccine label distributions
- Standard: 5-fold CV (configurable via `data.cv_folds` in YAML)
- Each fold trains on 4/5 of data, validates on 1/5
- Final model: retrain on all training data with best hyperparameters

**Class imbalance**: H1N1 vaccine is less common (~21%) than seasonal (~47%)
- Use `SMOTE=true` in config for synthetic oversampling
- Or `class_weight_strategy: balanced` for weighted loss
- Or tune decision threshold post-hoc

## Data Sets

The `./data/` directory contains the competition datasets:

- **training_set_features.csv**: Features for training samples (respondent_id + 35 features)
- **training_set_labels.csv**: Target labels for training samples (respondent_id, h1n1_vaccine, seasonal_vaccine)
- **test_set_features.csv**: Features for test samples (respondent_id + 35 features)
- **submission_format.csv**: Template for submission format (respondent_id, h1n1_vaccine, seasonal_vaccine)

### Data Confidentiality Rule

⚠️ **CRITICAL**: The competition data is proprietary and derived from CDC's National 2009 H1N1 Flu Survey.

**Constraints**:
- Data files must **NOT be committed to version control** (already excluded in `.gitignore`)
- Data must **NOT be exposed publicly** or shared outside the team/organization
- Data must **NOT be uploaded** to public repositories, public Jupyter notebooks, Discord, Slack, or any shared storage
- Use data exclusively for this competition project
- Downloaded data should be stored locally with restricted file access

These restrictions are part of the DrivenData competition's terms of service and data use agreement.

## Key Files & Patterns

- **[docs/PROBLEM_DESCRIPTION.md](docs/PROBLEM_DESCRIPTION.md)**: Authoritative source for feature definitions and submission format
- **[docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)**: Complete architecture overview with component responsibilities
- **[docs/TWO_PIPELINE_QUICKSTART.md](docs/TWO_PIPELINE_QUICKSTART.md)**: Quick reference for running LR vs XGBoost comparison
- **[README.md](README.md)**: Competition context and resources
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Summary of implemented pipelines and results

**Testing utilities**:
- **test_knn_neighbors.py**: Hyperparameter optimization script for KNN imputation
- Run standalone: `python test_knn_neighbors.py` (tests k=3,5,7,9,11 neighbors)

## Development Notes

- Focus on probability calibration (ROC AUC rewards well-calibrated estimates, not just rankings)
- Doctor recommendation (`doctor_recc_h1n1`, `doctor_recc_seasonal`) is likely a strong predictor - explore its correlation with targets
- Handle class imbalance if one vaccine is significantly more/less adopted than the other
- Missing data in employment features may indicate unemployment or non-applicability
