# Baseline Pipeline Execution Context & References

## Key Files

### Existing Code to Modify/Complete

| File | Current State | Implementation Needed |
|------|---------------|----------------------|
| [main.py](main.py) | Empty, needs orchestrator | Implement 10-stage pipeline orchestrator: load config → data → preprocess → train → calibrate → evaluate → track → predict → submit |
| [src/config.py](src/config.py) | Dataclass definitions exist, YAML loading stubbed | Complete YAML/JSON parsing, validation logic, ensure all config sections map to dataclasses (DataConfig, PreprocessingConfig, ModelConfig, EvaluationConfig, PipelineConfig) |
| [src/data/loader.py](src/data/loader.py) | DataLoader interface defined, CSVDataLoader is TODO | Implement CSVDataLoader to read train features/labels and test features from CSV files, return DataFrames with proper indexing |
| [src/preprocessing/imputation.py](src/preprocessing/imputation.py) | Interfaces defined, all strategies are TODO | Implement MeanImputer, ModeImputer, KNNImputer, MICEImputer; fit on training data only, transform all splits |
| [src/preprocessing/encoding.py](src/preprocessing/encoding.py) | Interfaces defined, all encoders are TODO | Implement OrdinalEncoder, OneHotEncoder, TargetEncoder; handle multiple feature groups, respect `drop_first=true`, use `handle_unknown=ignore` for test compatibility |
| [src/models/factory.py](src/models/factory.py) | Factory interface defined, LogisticRegressionModel is TODO | Implement LogisticRegressionModel with sklearn.linear_model.LogisticRegression; output probabilities not binary labels |
| [src/training/engine.py](src/training/engine.py) | TrainingEngine interface defined, run_cv() is TODO | Implement stratified K-fold loop with `combined_label = h1n1 + 2*seasonal` stratification; fit preprocessing per fold; train models; collect out-of-fold predictions |
| [src/calibration/calibrator.py](src/calibration/calibrator.py) | Calibrator interfaces defined, PlattScaler is TODO | Implement Platt scaling: fit logistic regression on (validation_probs, actual_labels); apply to test predictions |
| [src/evaluation/metrics.py](src/evaluation/metrics.py) | Partial implementation | Complete AUROC calculation per vaccine, mean AUROC, calibration error (ECE), Brier score; handle multilabel case properly |
| [src/evaluation/plots.py](src/evaluation/plots.py) | Plot interface defined, implementations are TODO | Implement roc_curves(), calibration_curve(), confidence_distribution() plots; use matplotlib/seaborn; save to output_dir |
| [src/tracking/logger.py](src/tracking/logger.py) | Logger interface defined, CSV logging is TODO | Implement CSVExperimentLogger to append metrics to CSV file at intervals; include timestamp, config identifier, all metrics |
| [src/prediction/predictor.py](src/prediction/predictor.py) | Predictor interface defined, implementation is TODO | Implement test set prediction: apply preprocessing fitted on full training data, generate probabilities from both models |

### New Files to Create

None required; all necessary module structure exists.

### Reference Implementations & Patterns

| File | Relevance |
|------|-----------|
| [examples/config_baseline.yaml](examples/config_baseline.yaml) | Baseline configuration that drives all pipeline behavior; defines imputation (mean), encoding (ordinal + onehot per feature group), logistic regression hyperparameters (C=1.0, penalty=l2, class_weight=balanced), Platt scaling calibration, evaluation metrics, and output paths |
| [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | High-level system architecture and design patterns; reference for module responsibilities |
| [docs/PROBLEM_DESCRIPTION.md](docs/PROBLEM_DESCRIPTION.md) | Feature definitions, target variable details, submission format specification, evaluation metric definition (mean ROC AUC) |
| [data/submission_format.csv](data/submission_format.csv) | Template for submission output; defines required format: respondent_id, h1n1_vaccine, seasonal_vaccine |

## Architecture Decisions

### Decision 1: Two Independent Models vs. Single Multilabel Model

- **Context**: Flu shot problem has two independent binary targets (h1n1_vaccine, seasonal_vaccine); each person can receive neither, one, or both vaccines
- **Decision**: Train two independent binary classifiers (one for each vaccine) rather than a single multilabel model
- **Rationale**: Simpler implementation, better interpretability, easier to debug, allows different hyperparameters per vaccine if needed, multilabel libraries add unnecessary complexity for independent targets
- **Alternatives Considered**: 
  - Single multilabel model (scikit-multioutput or specialized libraries): overkill for independent targets, harder to calibrate per-vaccine probabilities
  - Single model with two outputs: could work but would require custom loss functions to treat targets as truly independent

### Decision 2: Stratified K-Fold with Combined Label Stratification

- **Context**: Standard StratifiedKFold assumes single target, but we need balanced folds for BOTH targets
- **Decision**: Create synthetic stratification column: `combined_label = h1n1_vaccine + 2*seasonal_vaccine` (creates 4 classes: 0=neither, 1=seasonal only, 2=h1n1 only, 3=both)
- **Rationale**: StratifiedKFold on this combined label ensures each fold has proportional representation of all 4 combinations, balancing both vaccines simultaneously
- **Alternatives Considered**:
  - Separate stratification per vaccine: would require custom fold logic, harder to implement
  - Random KFold: could lead to severe class imbalance in some folds
  - Nested stratification: more complex without clear benefit

### Decision 3: Preprocessing Fit Only on Training Fold

- **Context**: Must prevent data leakage while handling imputation and encoding
- **Decision**: For each CV fold, fit imputation and encoding statistics ONLY on the training fold; apply fitted transformations to validation and test folds
- **Rationale**: Prevents leakage of validation/test information into preprocessing parameters; matches real-world scenario where test set statistics are unknown
- **Alternatives Considered**:
  - Fit on entire training set before CV: introduces leakage, inflates metrics
  - Fit per-instance: not possible for imputation and encoding
  - Use preprocessing within sklearn Pipeline: can be done but requires careful nested pipeline construction

### Decision 4: Platt Scaling for Probability Calibration

- **Context**: Logistic regression outputs uncalibrated probabilities; ROC AUC rewards calibration
- **Decision**: Use Platt scaling (fit logistic regression on validation fold predictions vs. actual labels) to calibrate probabilities
- **Rationale**: Simple, interpretable, works well with logistic regression, prevents overfitting with nested CV, minimal computational overhead
- **Alternatives Considered**:
  - Isotonic regression: more flexible but requires more data, risks overfitting
  - Temperature scaling: effective but requires tuning temperature parameter
  - No calibration: leaves probability estimates uncalibrated; ROC AUC may not reward well-calibrated models

### Decision 5: Config-Driven Pipeline with No Hardcoding

- **Context**: Multiple experiments with different hyperparameters and strategies should be easy to run
- **Decision**: All pipeline behavior defined in YAML config file; main.py is generic orchestrator with no model-specific logic
- **Rationale**: Enables rapid experimentation, reproducibility, easy sharing of configurations, separation of concerns
- **Alternatives Considered**:
  - Hardcoded parameters in code: inflexible, requires code changes for each experiment
  - Command-line arguments for everything: verbose, hard to manage many parameters
  - Hybrid approach (some config, some CLI): confusing, error-prone

## Dependencies

### Internal Dependencies

- **[src/config.py](src/config.py)**: Parses baseline config into structured dataclasses; used by main.py to configure all downstream components
- **[src/data/loader.py](src/data/loader.py)**: Loads training and test data; output is raw DataFrames fed to preprocessing
- **[src/preprocessing/imputation.py](src/preprocessing/imputation.py)**: First preprocessing stage; fits on training fold, transforms all folds
- **[src/preprocessing/encoding.py](src/preprocessing/encoding.py)**: Second preprocessing stage; follows imputation; fits on training fold, transforms all folds
- **[src/models/factory.py](src/models/factory.py)**: Creates model instances from config; used by training engine
- **[src/training/engine.py](src/training/engine.py)**: Orchestrates cross-validation loop; calls preprocessing, model training, prediction generation
- **[src/calibration/calibrator.py](src/calibration/calibrator.py)**: Calibrates validation predictions using Platt scaling; applied to test predictions
- **[src/evaluation/metrics.py](src/evaluation/metrics.py)**: Computes AUROC, calibration error, Brier score on validation predictions
- **[src/evaluation/plots.py](src/evaluation/plots.py)**: Generates ROC curves and calibration plots from validation predictions and metrics
- **[src/tracking/logger.py](src/tracking/logger.py)**: Logs final metrics to CSV for experiment tracking
- **[src/prediction/predictor.py](src/prediction/predictor.py)**: Generates test set predictions after training; applies preprocessing fitted on full training data
- **[src/utils/helpers.py](src/utils/helpers.py)**: Utility functions for creating output directories, stratification logic, etc.

### External Dependencies

- **pandas**: Data loading, manipulation, imputation; already in requirements
- **scikit-learn**: Logistic regression, stratified k-fold, ROC AUC, calibration, imputation (KNN), encoding; already in requirements
- **PyYAML**: Config file parsing; likely needed, verify in requirements
- **matplotlib/seaborn**: Visualization; likely needed, verify in requirements
- **numpy**: Numeric operations; already in requirements

## Related Documentation

- [examples/config_baseline.yaml](examples/config_baseline.yaml): Baseline configuration file that drives the entire pipeline; defines imputation strategy, feature encoding per group, model hyperparameters, calibration method, evaluation metrics, output paths
- [docs/PROBLEM_DESCRIPTION.md](docs/PROBLEM_DESCRIPTION.md): Complete feature definitions, target variable descriptions, evaluation metric (mean ROC AUC), submission format requirements
- [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md): Architectural overview of the pipeline; module responsibilities and interactions
- [docs/SCAFFOLDING.md](docs/SCAFFOLDING.md): Guidance on feature engineering and EDA patterns for this problem
- [data/submission_format.csv](data/submission_format.csv): Template showing required submission format (respondent_id, h1n1_vaccine, seasonal_vaccine)

## Open Questions

1. **Error Handling for Test Set Edge Cases** — If test set contains categorical values not seen during training, how should encoding handle them? Decision: Use `handle_unknown=ignore` for one-hot encoding, or pre-fill with training mode.

2. **Imputation Method for Missing Employment Features** — Employment features have significant NaN values; should they be dropped if >50% missing, or imputed with a "missing" category flag? Decision: Config specifies `drop_threshold: 0.5`, implement accordingly.

3. **Calibration on Validation vs. Separate Holdout Set** — Should Platt scaling fit on CV validation fold predictions or a separate calibration holdout? Decision: Per config `calibration_folds: 3`, use nested CV; alternatively fit on validation fold.

4. **Submission File Path and Naming** — Where should final submission CSV be written? Decision: `output_dir: ./submissions/` and `output_filename: submission_baseline.csv` from config.

5. **Logging and Debugging Output** — How verbose should logging be during pipeline execution? Decision: Config specifies `logging.level: INFO`; include timing for each stage.

