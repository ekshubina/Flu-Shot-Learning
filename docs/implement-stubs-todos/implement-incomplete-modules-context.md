# Implement Incomplete Modules Context & References

## Key Files

### Existing Code to Modify

| File | Purpose | Changes Needed |
|------|---------|----------------|
| [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py) | Contains 11 imputation strategy classes; base class + implementations | Implement `DropRowsImputation.fit()`, `.transform()`, `DropColumnsImputation.fit()`, `.transform()`, complete `MeanImputation.transform()`, `ModeImputation.transform()` |
| [src/data/loader.py](../../../../src/data/loader.py) | Data loading and splitting; CSVDataLoader class | Verify `load_train()` and `load_test()` completeness; implement `validate()` method |
| [src/models/factory.py](../../../../src/models/factory.py) | Model wrappers and factory; 4 model classes | Implement `XGBoostModel` (5 methods), `LightGBMModel` (5 methods), `RandomForestModel` (5 methods) |

### Reference Implementations

| File | Relevance |
|------|-----------|
| [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L220) - TypeBasedImputation | Complete working imputation strategy showing the full `fit()`/`transform()` pattern |
| [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L260) - MeanImputation.fit() | Already-implemented `fit()` method showing how to store learned parameters in `fit_params` |
| [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L300) - ModeImputation.fit() | Already-implemented `fit()` method for categorical data handling |
| [src/data/loader.py](../../../../src/data/loader.py#L330) - load_train() | Already-implemented training data loading with caching and validation |
| [src/models/factory.py](../../../../src/models/factory.py#L160) - LogisticRegressionModel | **COMPLETE reference implementation** showing full pattern for all 5 methods (fit, predict_proba, get_feature_importance, get_params, set_params) |

## Architecture Decisions

### Decision 1: Imputation Strategy Pattern
- **Context**: Need consistency across 11 different imputation strategies
- **Decision**: Follow scikit-learn's `fit()`/`transform()` pattern with stored `fit_params` dict
- **Rationale**: Enables pipeline orchestration, allows test set transformation with training-learned parameters, familiar to users
- **Alternatives Considered**: Custom fit methods per strategy; storing parameters as instance variables (chose fit_params for consistency)

### Decision 2: Data Validation Placement
- **Context**: Need to validate data before training begins
- **Decision**: Implement `validate()` as method in `CSVDataLoader` class returning `DataValidationResult` dataclass
- **Rationale**: Centralized location, can be called before pipeline starts, returns structured result for logging
- **Alternatives Considered**: Separate validation module; inline checks in main.py (chose class method for modularity)

### Decision 3: Row/Column Dropping Strategies
- **Context**: Need to handle two different deletion scenarios
- **Decision**: Separate `DropRowsImputation` (removes samples with any NaN) from `DropColumnsImputation` (removes features exceeding threshold)
- **Rationale**: Different use cases; row dropping is drastic (loses data samples), column dropping is selective (loses features)
- **Alternatives Considered**: Single strategy with configurable behavior (chose separate classes for clarity)

### Decision 4: Model Wrapper Implementation
- **Context**: Need to support XGBoost, LightGBM, and RandomForest alongside LogisticRegression
- **Decision**: Create wrapper classes for each model type following the BaseModel ABC and LogisticRegressionModel pattern
- **Rationale**: Consistent interface across all models; enables easy swapping in configs; supports hyperparameter optimization
- **Alternatives Considered**: Direct integration of sklearn models without wrappers (chose wrappers for flexibility and feature consistency)

## Dependencies

### Internal Dependencies
- `ImputationStrategy`: Abstract base class that all imputation strategies inherit from
- `DataValidationResult`: Dataclass defined in [src/data/loader.py](../../../../src/data/loader.py) to hold validation results
- `pd.DataFrame`, `np.ndarray`: Core data structures for all implementations
- `sklearn.model_selection.StratifiedKFold`: Used in `create_splits()` for multilabel stratification

### External Dependencies
- `pandas` (pd): DataFrame operations for imputation and data loading
- `numpy` (np): Numerical operations and NaN handling
- `scikit-learn` (sklearn): Required by project; LogisticRegression, RandomForestClassifier
- `xgboost` (xgb): Optional but required for XGBoostModel (already in requirements.txt)
- `lightgbm` (lgb): Optional but required for LightGBMModel (already in requirements.txt)

## Related Documentation

- [docs/PROBLEM_DESCRIPTION.md](../../../../docs/PROBLEM_DESCRIPTION.md): Feature definitions and data schema
- [docs/SYSTEM_DESIGN.md](../../../../docs/SYSTEM_DESIGN.md): Architecture overview and component responsibilities
- [docs/CONTEXT_REPORT.md](../../../../docs/CONTEXT_REPORT.md): Analysis of missing data patterns
- [docs/IMPUTATION_ANALYSIS.md](../../../../docs/IMPUTATION_ANALYSIS.md): Deep dive into imputation strategies
- [examples/config_baseline.yaml](../../../../examples/config_baseline.yaml): Example config using imputation
- [main.py](../../../../main.py#L1): Pipeline orchestrator that calls these methods

## Open Questions

1. **Load methods status**: `load_train()` and `load_test()` appear to have implementation beyond the TODOs in docstrings. Should we verify these are truly complete by checking for any remaining `pass` statements or `raise NotImplementedError()`?

2. **Validation integration**: Should `validate()` be called automatically at pipeline start, or only when explicitly requested by user? (Currently no auto-call in main.py)

3. **Data loss warnings**: For `DropRowsImputation` and `DropColumnsImputation`, should we add warnings when >50% of data would be lost, or silently allow it?

4. **Missing data logging**: Should imputation strategies log statistics (e.g., "Dropped 250 rows with NaN", "Dropped 3 columns exceeding 70% missing") to pipeline logs, or just return silently?

5. **XGBoost/LightGBM imports**: Should implementations try to import xgboost/lightgbm at init time and fail fast, or try to import at fit() time? (Lazy loading vs eager loading)

## Implementation Patterns to Follow

### Pattern 1: Imputation Strategy fit()/transform()
```python
def fit(self, X: pd.DataFrame) -> "StrategyName":
    if X.empty:
        raise ValueError("Training data X cannot be empty")
    # Learn parameters from training data
    self.fit_params['key'] = computed_value
    self.feature_names = list(X.columns)
    self.fitted = True
    return self

def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    if not self.fitted:
        raise ValueError("Strategy must be fit before transform")
    X_transformed = X.copy()
    # Apply learned parameters to transform data
    return X_transformed
```

### Pattern 2: Model Implementation (from LogisticRegressionModel)
```python
class MyModel(BaseModel):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Extract hyperparams from config['hyperparameters']
        # Create underlying model instance
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyModel":
        self.feature_names = list(X.columns)
        self._model.fit(X, y)
        self.fitted = True
        self.feature_importances_ = <extract_importance>
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted...")
        return self._model.predict_proba(X)
        
    def get_feature_importance(self) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Model must be fitted...")
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.feature_importances_
        })
        return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {all param names and values}
        
    def set_params(self, **params) -> "MyModel":
        # Update self.param_name for each param
        # Recreate self._model with updated params
        self.fitted = False
        return self
```

### Pattern 3: Validation Result Collection
```python
issues = []
warnings = []
# Check conditions
if condition_fails:
    issues.append("Critical issue description")
if warning_condition:
    warnings.append("Non-critical warning description")
# Return structured result
return DataValidationResult(
    is_valid=len(issues) == 0,
    # ... other fields
    issues=issues,
    warnings=warnings,
)
```

## Critical Implementation Details

- **DropRowsImputation**: Must preserve column order and dtypes after removing rows
- **DropColumnsImputation**: Must store dropped column names for potential reporting
- **MeanImputation.transform()**: Must only fill columns that were in training data
- **ModeImputation.transform()**: Must handle case where mode is NaN (column is entirely NaN)
- **validate()**: Must not modify input data; purely analytical
- **XGBoostModel/LightGBMModel**: Must handle optional imports gracefully (try/except); XGBoost requires `use_label_encoder=False` in __init__
- **RandomForestModel**: Must extract feature_importances_ correctly from sklearn model
- **All models**: Must handle the case where features have different dtypes (some might be categorical); sklearn models expect numeric input
- **get_params()/set_params()**: Must maintain the pattern where set_params resets fitted flag and returns self for chaining

