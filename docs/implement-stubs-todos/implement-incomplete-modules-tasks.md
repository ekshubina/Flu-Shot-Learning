# Implement Incomplete Modules Task Checklist

## Summary
- **Total incomplete stubs**: 28 methods across 3 files
- **Estimated effort**: ~700 lines of implementation code
- **Blocking dependencies**: None (existing pipeline works with current implementations)
- **Testing approach**: Verify existing configs still work + add new configs using drop strategies and new models

---

## Phase 1: Imputation Strategy Implementation

### DropRowsImputation
- [x] **Implement fit() method** - Analyze missing data pattern, store which columns have NaN in fit_params
  - Add check: raise ValueError if X is empty
  - Compute columns with any missing values
  - Store column names in `fit_params['columns_with_missing']`
  - Set `self.fitted = True` and return self
  - [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L155)

- [x] **Implement transform() method** - Remove rows containing any NaN values
  - Add check: raise ValueError if not fitted
  - Use `X.dropna()` to remove rows with any missing values
  - Return cleaned DataFrame with same columns
  - [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L167)

### DropColumnsImputation
- [x] **Implement fit() method** - Identify columns exceeding missing threshold
  - Add check: raise ValueError if X is empty
  - Validate `drop_threshold` is between 0.0 and 1.0
  - Compute % missing per column: `(col.isna().sum() / len(col)) * 100`
  - Identify columns where % missing > `drop_threshold`
  - Store list in `fit_params['columns_to_drop']`
  - Set `self.fitted = True` and return self
  - [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L218)

- [x] **Implement transform() method** - Drop marked columns
  - Add check: raise ValueError if not fitted
  - Get columns to drop from `fit_params['columns_to_drop']`
  - Drop those columns from X (only if they exist in X)
  - Return DataFrame with fewer columns
  - [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L228)

### MeanImputation
- [x] **Implement transform() method** - Fill NaN with learned means
  - Add check: raise ValueError if not fitted
  - Create copy of X to avoid modifying input
  - For each column in `fit_params['column_means']`, fill NaN with mean value
  - Return imputed DataFrame
  - Note: fit() method is already complete
  - [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L285)

### ModeImputation
- [x] **Implement transform() method** - Fill NaN with learned modes
  - Add check: raise ValueError if not fitted
  - Create copy of X to avoid modifying input
  - For each column in `fit_params['column_modes']`, fill NaN with mode value
  - Return imputed DataFrame
  - Note: fit() method is already complete
  - [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L320)

---

## Phase 2: Data Validation Implementation

### Data Loading Verification
- [x] **Review load_train() method** - Confirm implementation is complete
  - Check that code loads training_set_features.csv
  - Check that code loads training_set_labels.csv
  - Check that features and labels are merged on respondent_id
  - Check that data is cached in `self._X_train` and `self._y_train`
  - Check that method returns tuple (X_train, y_train)
  - Verify no remaining `pass` or `NotImplementedError` statements
  - [src/data/loader.py](../../../../src/data/loader.py#L315)

- [x] **Review load_test() method** - Confirm implementation is complete
  - Check that code loads test_set_features.csv
  - Check that respondent_id is extracted as separate Series
  - Check that data is cached in `self._X_test`
  - Check that method returns tuple (X_test, respondent_ids)
  - Verify no remaining `pass` or `NotImplementedError` statements
  - [src/data/loader.py](../../../../src/data/loader.py#L368)

### Data Validation Implementation
- [x] **Implement validate() method** - Comprehensive data integrity checks
  - Load training and test data via load_train() and load_test()
  - Wrap in try/except to catch FileNotFoundError and return issues list
  - Check feature counts: expect 35 features + respondent_id = 36 columns
  - Check target columns exist: h1n1_vaccine, seasonal_vaccine
  - Check respondent_id uniqueness in X_train, y_train, and X_test
  - Check train features and labels have same number of rows
  - Check target values are binary (0 or 1, NaN allowed)
  - Compute % missing per feature and store in `missing_by_feature` dict
  - Flag warnings for columns with >50% missing
  - Compute class distribution for each target
  - Infer feature types (categorical/ordinal/numeric) based on dtype and cardinality
  - Collect critical issues and non-critical warnings
  - Return DataValidationResult with all computed fields
  - [src/data/loader.py](../../../../src/data/loader.py#L490)

---

## Phase 3: Model Implementation

### XGBoostModel
- [x] **Implement __init__() method** - Initialize XGBoost model with hyperparameters
  - Extract hyperparameters from config (n_estimators, max_depth, learning_rate, etc.)
  - Set default values for any missing hyperparameters
  - Create xgboost.XGBClassifier instance with parameters
  - Handle ImportError if xgboost not installed
  - [src/models/factory.py](../../../../src/models/factory.py#L300)

- [x] **Implement fit() method** - Train XGBoost model
  - Store feature names from X.columns
  - Call self._model.fit(X, y)
  - Set self.fitted = True
  - Extract and store feature_importances_ from trained model
  - Return self for method chaining
  - [src/models/factory.py](../../../../src/models/factory.py#L315)

- [x] **Implement predict_proba() method** - Return probability predictions
  - Check if model is fitted; raise ValueError if not
  - Call self._model.predict_proba(X)
  - Return (n_samples, 2) array with probabilities
  - [src/models/factory.py](../../../../src/models/factory.py#L328)

- [x] **Implement get_feature_importance() method** - Extract importance scores
  - Check if model is fitted; raise ValueError if not
  - Create DataFrame with feature names and feature_importances_
  - Sort by importance descending
  - Return sorted DataFrame
  - [src/models/factory.py](../../../../src/models/factory.py#L339)

- [x] **Implement get_params() and set_params() methods** - Hyperparameter management
  - get_params(): Return dict of all hyperparameters
  - set_params(): Update hyperparameters, recreate XGBClassifier, reset fitted flag
  - Return self from set_params for chaining
  - [src/models/factory.py](../../../../src/models/factory.py#L352)

### LightGBMModel
- [x] **Implement __init__() method** - Initialize LightGBM model with hyperparameters
  - Extract hyperparameters from config (n_estimators, max_depth, num_leaves, etc.)
  - Set default values for any missing hyperparameters
  - Create lightgbm.LGBMClassifier instance with parameters
  - Handle ImportError if lightgbm not installed
  - [src/models/factory.py](../../../../src/models/factory.py#L370)

- [x] **Implement fit() method** - Train LightGBM model
  - Store feature names from X.columns
  - Call self._model.fit(X, y)
  - Set self.fitted = True
  - Extract and store feature_importances_ from trained model
  - Return self for method chaining
  - [src/models/factory.py](../../../../src/models/factory.py#L385)

- [x] **Implement predict_proba() method** - Return probability predictions
  - Check if model is fitted; raise ValueError if not
  - Call self._model.predict_proba(X)
  - Return (n_samples, 2) array with probabilities
  - [src/models/factory.py](../../../../src/models/factory.py#L396)

- [x] **Implement get_feature_importance() method** - Extract importance scores
  - Check if model is fitted; raise ValueError if not
  - Create DataFrame with feature names and feature_importances_
  - Sort by importance descending
  - Return sorted DataFrame
  - [src/models/factory.py](../../../../src/models/factory.py#L407)

- [x] **Implement get_params() and set_params() methods** - Hyperparameter management
  - get_params(): Return dict of all hyperparameters
  - set_params(): Update hyperparameters, recreate LGBMClassifier, reset fitted flag
  - Return self from set_params for chaining
  - [src/models/factory.py](../../../../src/models/factory.py#L420)

### RandomForestModel
- [x] **Implement __init__() method** - Initialize Random Forest model with hyperparameters
  - Extract hyperparameters from config (n_estimators, max_depth, min_samples_split, etc.)
  - Set default values for any missing hyperparameters
  - Create sklearn.ensemble.RandomForestClassifier instance with parameters
  - [src/models/factory.py](../../../../src/models/factory.py#L440)

- [x] **Implement fit() method** - Train Random Forest model
  - Store feature names from X.columns
  - Call self._model.fit(X, y)
  - Set self.fitted = True
  - Extract and store feature_importances_ from trained model
  - Return self for method chaining
  - [src/models/factory.py](../../../../src/models/factory.py#L455)

- [x] **Implement predict_proba() method** - Return probability predictions
  - Check if model is fitted; raise ValueError if not
  - Call self._model.predict_proba(X)
  - Return (n_samples, 2) array with probabilities
  - [src/models/factory.py](../../../../src/models/factory.py#L466)

- [x] **Implement get_feature_importance() method** - Extract importance scores
  - Check if model is fitted; raise ValueError if not
  - Create DataFrame with feature names and feature_importances_
  - Sort by importance descending
  - Return sorted DataFrame
  - [src/models/factory.py](../../../../src/models/factory.py#L477)

- [x] **Implement get_params() and set_params() methods** - Hyperparameter management
  - get_params(): Return dict of all hyperparameters
  - set_params(): Update hyperparameters, recreate RandomForestClassifier, reset fitted flag
  - Return self from set_params for chaining
  - [src/models/factory.py](../../../../src/models/factory.py#L490)

---

## Phase 4: Testing & Verification

### Integration Testing
- [ ] **Test baseline config** - Run existing baseline pipeline
  - Execute: `python main.py --config examples/config_baseline.yaml`
  - Verify: pipeline completes without errors
  - Verify: experiments_baseline.csv contains valid metrics
  - [examples/config_baseline.yaml](../../../../examples/config_baseline.yaml)

- [ ] **Test type-based config** - Run existing type-based imputation
  - Execute: `python main.py --config examples/config_type_based_basic.yaml`
  - Verify: pipeline completes without errors
  - Verify: experiments_type_based_basic.csv contains valid metrics
  - [examples/config_type_based_basic.yaml](../../../../examples/config_type_based_basic.yaml)

- [ ] **Test XGBoost config** - Run existing XGBoost config
  - Execute: `python main.py --config examples/config_type_based_boosting.yaml`
  - Verify: pipeline completes without errors
  - Verify: XGBoost model trains and generates valid predictions
  - Verify: experiments_type_based_boosting.csv contains valid AUROC scores
  - [examples/config_type_based_boosting.yaml](../../../../examples/config_type_based_boosting.yaml)

### New Model Implementation Testing
- [ ] **Test XGBoostModel directly** - Unit test new model implementation
  - Create simple X_train, y_train, X_test data
  - Fit model: `model.fit(X_train, y_train)`
  - Get predictions: `proba = model.predict_proba(X_test)`
  - Verify shape: `proba.shape == (n_test, 2)`
  - Verify range: `0 <= proba <= 1`
  - Get importance: `importance_df = model.get_feature_importance()`
  - Test params: `params = model.get_params()` and `model.set_params(**params)`

- [ ] **Test LightGBMModel directly** - Unit test new model implementation
  - Create simple X_train, y_train, X_test data
  - Fit model: `model.fit(X_train, y_train)`
  - Get predictions: `proba = model.predict_proba(X_test)`
  - Verify shape: `proba.shape == (n_test, 2)`
  - Verify range: `0 <= proba <= 1`
  - Get importance: `importance_df = model.get_feature_importance()`
  - Test params: `params = model.get_params()` and `model.set_params(**params)`

- [ ] **Test RandomForestModel directly** - Unit test new model implementation
  - Create simple X_train, y_train, X_test data
  - Fit model: `model.fit(X_train, y_train)`
  - Get predictions: `proba = model.predict_proba(X_test)`
  - Verify shape: `proba.shape == (n_test, 2)`
  - Verify range: `0 <= proba <= 1`
  - Get importance: `importance_df = model.get_feature_importance()`
  - Test params: `params = model.get_params()` and `model.set_params(**params)`

### Implementation Verification
- [ ] **Remove all TODO comments** - Ensure docstrings are updated to reflect actual implementations
  - Search for "TODO: Implement" in imputation.py
  - Replace with descriptions of actual implementation
  - Search for "TODO:" in loader.py
  - Replace with descriptions of actual implementation

- [ ] **Verify no pass-only stubs remain** - Confirm all methods have real code
  - Check DropRowsImputation: no `pass` statements in fit/transform
  - Check DropColumnsImputation: no `pass` statements in fit/transform
  - Check MeanImputation.transform(): has actual fill logic
  - Check ModeImputation.transform(): has actual fill logic
  - Check CSVDataLoader.validate(): has actual validation logic

- [ ] **Test with new config** - Create config using drop strategies (optional)
  - Create `config_drop_rows_test.yaml` using DropRowsImputation
  - Create `config_drop_columns_test.yaml` using DropColumnsImputation
  - Run both to verify strategies work end-to-end
  - Verify experiments_*.csv outputs are valid

---

## Phase 4: Documentation

- [ ] **Update docstrings** - Replace TODO comments with actual implementation descriptions
  - For each completed stub, update the docstring with actual behavior
  - Ensure examples are accurate and match implementation
  - Add any warnings about data loss or edge cases

- [ ] **Update IMPLEMENTATION_SUMMARY.md** - Document completed work
  - Add section: "Phase N: Completed Imputation and Validation Implementations"
  - List all 13 completed stubs
  - Note: No breaking changes to existing pipeline
  - Document any new validation checks available

---

## Phase 5: Documentation

- [ ] **Update docstrings** - Replace TODO comments with actual implementation descriptions
  - For each completed stub, update the docstring with actual behavior
  - Ensure examples are accurate and match implementation
  - Add any warnings about data loss or edge cases
  - Update imputation.py docstrings
  - Update loader.py docstrings
  - Update factory.py docstrings

- [ ] **Update IMPLEMENTATION_SUMMARY.md** - Document completed work
  - Add section: "Phase X: Completed Imputation, Validation, and Model Implementations"
  - List all 28 completed stubs by file and class
  - Note: No breaking changes to existing pipeline
  - Document any new validation checks available
  - Document availability of XGBoost, LightGBM, RandomForest models

---

## Implementation Order

**Recommended sequence for efficient execution:**

1. **Imputation Phase 1** (4 methods, ~50 lines)
   - DropRowsImputation.fit() and .transform()
   - DropColumnsImputation.fit() and .transform()
   - Simplest implementations; good warm-up

2. **Imputation Phase 2** (2 methods, ~15 lines)
   - MeanImputation.transform()
   - ModeImputation.transform()
   - Very straightforward; loop and fillna

3. **Data Loading Phase** (1 method, ~80 lines)
   - CSVDataLoader.validate()
   - Most complex; multiple checks
   - Do after imputation to maintain momentum

4. **Model Phase - LogisticRegression review** (0 methods, ~5 mins)
   - Review LogisticRegressionModel as reference pattern
   - Confirm all 5 methods are fully implemented
   - Use as template for other models

5. **Model Phase - XGBoostModel** (5 methods, ~120 lines)
   - Start with models since you have LogisticRegression pattern
   - XGBoost first since config_type_based_boosting.yaml uses it

6. **Model Phase - LightGBMModel** (5 methods, ~120 lines)
   - Similar pattern to XGBoost
   - Build on lessons from XGBoost implementation

7. **Model Phase - RandomForestModel** (5 methods, ~100 lines)
   - Similar pattern to other models
   - Simplest since sklearn is already available (no optional imports)

8. **Testing and cleanup** (Phase 4-5, ~45-60 mins)
   - Run existing configs to verify nothing broke
   - Test new models with simple datasets
   - Remove TODO comments and update docstrings
   - Update IMPLEMENTATION_SUMMARY.md

---

## Acceptance Criteria

Each task should be:
- **Testable**: Can run `python main.py --config examples/config_baseline.yaml` and verify no new errors
- **Atomic**: Each method implementation is independent and can be reviewed separately
- **Specific**: Implementation matches docstring specifications exactly
- **Actionable**: Clear line numbers and file locations provided for each task

