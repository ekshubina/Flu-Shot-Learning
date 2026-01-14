# Implement TODOs and Pass-Only Stubs

## Overview

This project has 28 incomplete method stubs across 3 core modules that block full utilization of the pipeline. The imputation strategies module contains 4 classes with empty implementations (`DropRowsImputation`, `DropColumnsImputation`, and partial stubs in transformation methods), the data loader module has validation logic marked with TODOs, and the models factory contains 3 incomplete model wrapper classes (`XGBoostModel`, `LightGBMModel`, `RandomForestModel` - each with 5 stub methods). These are foundational pipeline components that the training engine, preprocessing system, and model orchestration depend on. Completing these stubs will enable:

- Validation of data integrity before pipeline execution
- Use of row-dropping and column-dropping imputation strategies in experiments
- Full visibility into missing data patterns and class distributions
- Gradient boosting and random forest model support (XGBoost, LightGBM, RandomForest)
- More robust error handling and data quality assurance
- Complete model flexibility for hyperparameter optimization experiments

## Goals

1. Complete all imputation strategy implementations (`DropRowsImputation`, `DropColumnsImputation`) following the existing pattern in `MeanImputation` and `ModeImputation`
2. Finalize transformation methods in `MeanImputation` and `ModeImputation` to fill NaN values correctly
3. Implement data validation method in `CSVDataLoader.validate()` to check file integrity, class distributions, and missing value patterns
4. Implement three gradient boosting/tree model wrapper classes (`XGBoostModel`, `LightGBMModel`, `RandomForestModel`) following the `LogisticRegressionModel` pattern
5. Ensure all implementations maintain consistency with the scikit-learn `fit()`/`transform()` pattern used throughout the codebase
6. Verify existing implementations that appear complete (`load_train()`, `load_test()`, `LogisticRegressionModel`) are truly finished

## Non-Goals

1. Refactor existing imputation strategies or change their API
2. Optimize performance of imputation methods beyond current design
3. Add new imputation strategies beyond those already stubbed
4. Implement early stopping or advanced XGBoost/LightGBM features (basic functionality only)
5. Create comprehensive test suite (beyond verifying integration with existing pipeline)

## Implementation Steps

### Phase 1: Complete Imputation Strategy Stubs

#### Step 1.1: Implement DropRowsImputation class
- **File**: [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L140)
- Implement `fit()`: Analyze which columns have missing values, store column names in `fit_params['columns_with_missing']`
- Implement `transform()`: Remove rows with any NaN values, return cleaned DataFrame
- Add logging to report how many rows were dropped and from which columns

#### Step 1.2: Implement DropColumnsImputation class
- **File**: [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L180)
- Implement `fit()`: Compute percentage missing per column, identify columns exceeding `drop_threshold`, store column list in `fit_params['columns_to_drop']`
- Implement `transform()`: Drop marked columns from input DataFrame, return reduced DataFrame
- Validate threshold parameter (0.0-1.0 range)

#### Step 1.3: Complete MeanImputation.transform()
- **File**: [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L285)
- Finalize filling NaN values with learned column means
- Already has `fit()` implemented; just need to fill in `transform()` method body
- Use the pattern: iterate over `fit_params['column_means']` and fill NaN in each column

#### Step 1.4: Complete ModeImputation.transform()
- **File**: [src/preprocessing/imputation.py](../../../../src/preprocessing/imputation.py#L315)
- Finalize filling NaN values with learned column modes
- Already has `fit()` implemented; just need to fill in `transform()` method body
- Use the pattern: iterate over `fit_params['column_modes']` and fill NaN in each column

### Phase 2: Implement Data Validation

#### Step 2.1: Review and verify load_train() and load_test()
- **File**: [src/data/loader.py](../../../../src/data/loader.py#L315)
- Review the implementation to confirm all TODOs are addressed
- Verify caching mechanism works correctly
- Check error handling for missing files and columns

#### Step 2.2: Implement CSVDataLoader.validate()
- **File**: [src/data/loader.py](../../../../src/data/loader.py#L490)
- Load training and test data (using `load_train()` and `load_test()`)
- Check file accessibility and raise `FileNotFoundError` if missing
- Verify column counts (35 features + respondent_id = 36 total)
- Verify target columns exist (h1n1_vaccine, seasonal_vaccine)
- Check respondent_id uniqueness in each dataset
- Verify train features and labels have matching lengths
- Verify target values are binary (0 or 1, excluding NaN)
- Compute missing value percentages by column
- Compute class distribution for each target
- Infer feature types (categorical, ordinal, numeric)
- Return `DataValidationResult` with all collected information

### Phase 3: Implement Model Wrapper Classes

#### Step 3.1: Implement XGBoostModel class
- **File**: [src/models/factory.py](../../../../src/models/factory.py#L300)
- **Requirements**: xgboost package must be installed (already in requirements.txt)
- Implement `__init__()`: Extract hyperparameters, create xgboost.XGBClassifier instance
- Implement `fit()`: Store feature names, train XGBClassifier, set fitted flag
- Implement `predict_proba()`: Return probability predictions from trained model
- Implement `get_feature_importance()`: Extract feature_importances_, create sorted DataFrame
- Implement `get_params()` and `set_params()`: Follow LogisticRegressionModel pattern

#### Step 3.2: Implement LightGBMModel class
- **File**: [src/models/factory.py](../../../../src/models/factory.py#L370)
- **Requirements**: lightgbm package must be installed (already in requirements.txt)
- Implement `__init__()`: Extract hyperparameters, create lgb.LGBMClassifier instance
- Implement `fit()`: Store feature names, train LGBMClassifier, set fitted flag
- Implement `predict_proba()`: Return probability predictions from trained model
- Implement `get_feature_importance()`: Extract feature_importances_, create sorted DataFrame
- Implement `get_params()` and `set_params()`: Follow LogisticRegressionModel pattern

#### Step 3.3: Implement RandomForestModel class
- **File**: [src/models/factory.py](../../../../src/models/factory.py#L440)
- **Requirements**: sklearn RandomForestClassifier (already imported)
- Implement `__init__()`: Extract hyperparameters, create RandomForestClassifier instance
- Implement `fit()`: Store feature names, train RandomForestClassifier, set fitted flag
- Implement `predict_proba()`: Return probability predictions from trained model
- Implement `get_feature_importance()`: Extract feature_importances_, create sorted DataFrame
- Implement `get_params()` and `set_params()`: Follow LogisticRegressionModel pattern

### Phase 4: Verification and Testing

#### Step 4.1: Functional verification against existing configs
- Run `python main.py --config examples/config_baseline.yaml` to verify baseline still works
- Run `python main.py --config examples/config_type_based_basic.yaml` to verify type-based imputation
- Run `python main.py --config examples/config_type_based_boosting.yaml` to verify XGBoost works
- Verify that `DropRowsImputation` and `DropColumnsImputation` can be used in new configs without errors

#### Step 4.2: Test new model implementations
- Create test configs using XGBoost, LightGBM, and RandomForest models
- Verify each model trains successfully and generates valid predictions
- Verify feature importance extraction works for all models
- Verify hyperparameter getting/setting works correctly

#### Step 4.3: Validate implementation completeness
- Check that all TODO comments are resolved or replaced with actual code
- Verify that all abstract methods are no longer just `pass` stubs
- Ensure error messages are clear and helpful for debugging

## Success Criteria

1. All 28 incomplete stubs have functional implementations (not just `pass` statements)
2. `DropRowsImputation.fit()` and `.transform()` correctly remove rows with NaN values
3. `DropColumnsImputation.fit()` and `.transform()` correctly identify and remove columns exceeding threshold
4. `MeanImputation.transform()` and `ModeImputation.transform()` correctly fill NaN values
5. `CSVDataLoader.validate()` returns valid `DataValidationResult` with accurate statistics
6. `XGBoostModel`, `LightGBMModel`, and `RandomForestModel` train successfully and return probability predictions
7. All model implementations follow the scikit-learn `fit()`/`predict_proba()` pattern consistently
8. Feature importance extraction works for all models (tree-based models via feature_importances_)
9. Hyperparameter getting/setting (`get_params()`, `set_params()`) works correctly for all models
10. Existing configs (`config_baseline.yaml`, `config_type_based_basic.yaml`, `config_type_based_boosting.yaml`) still run successfully without modification
11. No new errors or warnings are introduced in the pipeline

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes to existing pipeline due to implementation changes | High | Review existing configs and test runs before/after implementation; verify no API changes to public methods |
| Imputation strategies drop too much data (rows/columns) making models invalid | Medium | Add validation checks in `fit()` to warn when >50% of data would be lost; document thresholds clearly |
| Missing data validation logic creates silent failures | Medium | Add comprehensive error messages; test with intentionally corrupt data files |
| XGBoost/LightGBM not installed causing import errors | Medium | Add try/except for optional imports; fail gracefully with helpful error message if library missing |
| Model implementations have subtle sklearn compatibility issues | Medium | Follow LogisticRegressionModel pattern exactly; test get_params/set_params round-trip |
| Feature importance values sum to zero causing division errors | Low | Add safety checks in DataFrame creation; handle edge case where all importances are zero |

