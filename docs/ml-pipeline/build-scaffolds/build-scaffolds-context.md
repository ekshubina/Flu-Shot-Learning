# ML Pipeline Scaffolding Context & References

## Key Files

### Existing Files to Reference

| File | Purpose | Relevance |
|------|---------|-----------|
| [SYSTEM_DESIGN.md](../../SYSTEM_DESIGN.md) | Architecture blueprint for 9-component ML pipeline | Primary reference for scaffolding structure |
| [PROBLEM_DESCRIPTION.md](../../PROBLEM_DESCRIPTION.md) | Feature definitions, submission format, evaluation metric | Defines data contracts and output requirements |
| [CONTEXT_REPORT.md](../../CONTEXT_REPORT.md) | Comprehensive analysis of dataset, features, missing data | Informs imputation/encoding strategy design |
| [main.py](../../../main.py) | Current entry point (empty) | Will become orchestrator after scaffolding |
| [requirements.txt](../../../requirements.txt) | Current dependencies (only pandas) | Needs expansion with ML stack |
| [README.md](../../../README.md) | Project overview | Will be updated with architecture section |

### New Files to Create

| File Path | Purpose | Type |
|-----------|---------|------|
| [src/__init__.py](../../../src/__init__.py) | Package root | Init file |
| [src/config.py](../../../src/config.py) | Configuration dataclasses | Config system |
| [src/data/__init__.py](../../../src/data/__init__.py) | Data module exports | Init file |
| [src/data/loader.py](../../../src/data/loader.py) | DataLoader interface and implementations | Abstract interface |
| [src/preprocessing/__init__.py](../../../src/preprocessing/__init__.py) | Preprocessing module exports | Init file |
| [src/preprocessing/imputation.py](../../../src/preprocessing/imputation.py) | Imputation strategy interfaces | Abstract interface |
| [src/preprocessing/encoding.py](../../../src/preprocessing/encoding.py) | Feature encoding interfaces | Abstract interface |
| [src/models/__init__.py](../../../src/models/__init__.py) | Models module exports | Init file |
| [src/models/factory.py](../../../src/models/factory.py) | Model factory and base classes | Factory pattern |
| [src/models/logistic_regression.py](../../../src/models/logistic_regression.py) | LogisticRegressionModel stub | Model stub |
| [src/models/xgboost.py](../../../src/models/xgboost.py) | XGBoostModel stub | Model stub |
| [src/models/lightgbm.py](../../../src/models/lightgbm.py) | LightGBMModel stub | Model stub |
| [src/models/random_forest.py](../../../src/models/random_forest.py) | RandomForestModel stub | Model stub |
| [src/training/__init__.py](../../../src/training/__init__.py) | Training module exports | Init file |
| [src/training/engine.py](../../../src/training/engine.py) | TrainingEngine interface | Abstract interface |
| [src/calibration/__init__.py](../../../src/calibration/__init__.py) | Calibration module exports | Init file |
| [src/calibration/calibrator.py](../../../src/calibration/calibrator.py) | CalibratorInterface and implementations | Abstract interface |
| [src/evaluation/__init__.py](../../../src/evaluation/__init__.py) | Evaluation module exports | Init file |
| [src/evaluation/metrics.py](../../../src/evaluation/metrics.py) | Evaluator class and metric functions | Evaluation logic |
| [src/evaluation/plots.py](../../../src/evaluation/plots.py) | Visualization functions | Plotting stubs |
| [src/tracking/__init__.py](../../../src/tracking/__init__.py) | Tracking module exports | Init file |
| [src/tracking/logger.py](../../../src/tracking/logger.py) | ExperimentTracker interface | Abstract interface |
| [src/prediction/__init__.py](../../../src/prediction/__init__.py) | Prediction module exports | Init file |
| [src/prediction/predictor.py](../../../src/prediction/predictor.py) | PredictionEngine interface | Abstract interface |
| [src/utils/__init__.py](../../../src/utils/__init__.py) | Utils module exports | Init file |
| [src/utils/logging.py](../../../src/utils/logging.py) | Logging setup helpers | Utility functions |
| [src/utils/validation.py](../../../src/utils/validation.py) | Data validation helpers | Utility functions |
| [src/utils/metrics.py](../../../src/utils/metrics.py) | Metric computation helpers | Utility functions |
| [src/utils/plots.py](../../../src/utils/plots.py) | Visualization helpers | Utility functions |
| [src/utils/helpers.py](../../../src/utils/helpers.py) | General utility functions | Utility functions |
| [examples/config_baseline.yaml](../../../examples/config_baseline.yaml) | Example baseline configuration | Configuration template |
| [examples/config_xgboost.yaml](../../../examples/config_xgboost.yaml) | Example XGBoost configuration | Configuration template |

## Architecture Decisions

### Decision 1: Modular Package Structure (src/)

- **Context**: System design specifies 9 distinct components; need clear organizational boundaries to avoid monolithic code
- **Decision**: Create `src/` folder with 9 subpackages (data, preprocessing, models, training, calibration, evaluation, tracking, prediction, utils) + config.py at top level
- **Rationale**: Separates concerns, enables parallel development, makes component swapping easy, aligns exactly with SYSTEM_DESIGN.md
- **Alternatives Considered**:
  - Flat structure (all modules in root): Simpler to navigate but harder to scale; doesn't reflect architectural intent
  - Nested hierarchy (deeper nesting): Better organization but adds navigation overhead; 2 levels (src/ + component) is optimal

### Decision 2: Abstract Base Classes for Interfaces

- **Context**: Multiple imputation, encoding, model, and calibration strategies must be interchangeable; need clear contracts
- **Decision**: Use Python ABC (Abstract Base Classes) with abstract methods and properties; type hints throughout
- **Rationale**: Forces clear interface definition; IDEs can validate implementations against contracts; facilitates testing with mocks
- **Alternatives Considered**:
  - Protocol classes (typing.Protocol): More Pythonic but less enforceable; harder to catch implementation mistakes
  - Duck typing: Most flexible but error-prone; easy to introduce incompatibilities

### Decision 3: Configuration as Dataclasses

- **Context**: Pipeline has 50+ configurable parameters across 8+ stages; need systematic, type-safe way to pass config
- **Decision**: Use Python dataclasses (`dataclasses` module) with nested structure mirroring pipeline stages; provide `load_from_dict()` and `load_from_yaml()` methods
- **Rationale**: Type hints, IDE auto-completion, easy serialization, clear validation, mirrors SYSTEM_DESIGN.md structure
- **Alternatives Considered**:
  - Dictionary-based config: Simpler but error-prone (typos in keys not caught); harder to document
  - YAML only: Flexible but decoupling config structure from Python types; separate schema validation needed
  - Pydantic models: More robust but adds external dependency; overkill for initial scaffolding

### Decision 4: Single Orchestrator (main.py) vs. Pipeline Library

- **Context**: Need entry point that demonstrates full workflow; future implementations may want flexible pipelines
- **Decision**: Keep main.py as simple orchestrator (import → instantiate → run sequence); components are independent and reusable, not bound to main.py
- **Rationale**: Allows both simple end-to-end runs and component-level testing; avoids vendor lock-in to a specific pipeline orchestration framework
- **Alternatives Considered**:
  - Pipeline object (Pipeline class managing component order): More OOP but adds complexity; harder to debug
  - Separate pipeline library (like Kubeflow): Overkill for current scope; can add later if needed

### Decision 5: Utilities Organization

- **Context**: Helper functions for logging, validation, metrics, plotting scattered across components; need centralized location
- **Decision**: Consolidate into `src/utils/` with 5 files (logging, validation, metrics, plots, helpers)
- **Rationale**: Easy discovery, reduces duplication, keeps component files focused on core responsibility
- **Alternatives Considered**:
  - One monolithic utils.py: Harder to navigate as utilities grow
  - Inline in each component: Duplicates code; harder to maintain; less reusable

## Dependencies

### Internal Dependencies (Modules)

- **src/main.py**: Imports all components and orchestrates execution
- **src/config.py**: Imported by all components for type hints and configuration loading
- **src/data/loader.py**: Imported by main.py; provides X_train, y_train, X_val, y_val, X_test
- **src/preprocessing/**: Imported by main.py; transforms data between stages
- **src/models/**: Imported by training engine; instantiated with config
- **src/training/**: Imported by main.py; consumes preprocessed data
- **src/calibration/**: Imported by main.py; post-processes predictions
- **src/evaluation/**: Imported by main.py; computes metrics on validation/test sets
- **src/tracking/**: Imported by main.py; logs all configuration and results
- **src/prediction/**: Imported by main.py; generates submission CSV
- **src/utils/**: Imported across all components as helper functions

### External Dependencies

Core ML stack (required):
- `numpy>=1.20.0`: Numerical computing
- `pandas>=1.3.0`: Data manipulation and CSV I/O
- `scikit-learn>=0.24.0`: Models, preprocessing, metrics, model selection

Visualization:
- `matplotlib>=3.3.0`: Plotting library (ROC curves, calibration plots)
- `seaborn>=0.11.0`: Statistical visualization (optional but recommended for better defaults)

Optional advanced:
- `xgboost>=1.5.0`: Gradient boosting models (recommended for strong baseline)
- `lightgbm>=3.3.0`: LightGBM models (recommended as faster alternative to XGBoost)
- `optuna>=2.0.0`: Bayesian hyperparameter optimization (for Phase 3: Model Exploration)

Development:
- `jupyter>=1.0.0`: Interactive notebooks for EDA (optional)
- `pytest>=6.0.0`: Unit testing framework (optional, for test scaffolds)
- `black>=21.0.0`: Code formatter (optional, for development)
- `flake8>=3.9.0`: Linter (optional, for development)

## Related Documentation

- [SYSTEM_DESIGN.md](../../SYSTEM_DESIGN.md): High-level architecture with component descriptions, data flow, and implementation roadmap
- [PROBLEM_DESCRIPTION.md](../../PROBLEM_DESCRIPTION.md): Feature definitions (35 features across 9 categories), multilabel setup, evaluation metric (ROC AUC)
- [CONTEXT_REPORT.md](../../CONTEXT_REPORT.md): Dataset analysis, missing data patterns, class imbalance ratios, feature cardinality
- [.github/copilot-instructions.md](../../../.github/copilot-instructions.md): Project-level guidelines and best practices

## Open Questions

1. **Configuration loading at runtime**: Should main.py accept a YAML config file path as CLI argument, or load from a default location? (Decision affects flexibility vs. simplicity)

2. **Model persistence format**: Should trained models be pickled, joblib, or another serialization format? (Affects reproducibility and deployment)

3. **Test scaffolds scope**: Should unit test stubs be included in scaffolds, or added only when implementation begins? (Decision affects scope of this phase)

4. **External experiment tracking**: Integrate MLflow/W&B now or defer to Phase 4? (Early integration provides structure; deferral keeps scope minimal)

5. **Feature group constants**: Should feature group definitions (opinion_features, behavioral_features, etc.) be in config.py or a separate constants file? (Affects config coupling to problem domain)

6. **Validation strictness**: How much should data validation happen at load time vs. at component boundaries? (Tight validation catches bugs early; loose validation allows more flexibility)

---

**Last Updated**: January 7, 2026
