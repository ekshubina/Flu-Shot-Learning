# Type-Based Imputation Task Checklist

**Last Updated**: January 12, 2026  
**Dependencies**: None (can start immediately)

## Summary

Implement `TypeBasedImputation` class and configuration to support different imputation strategies for ordinal, nominal, and binary-numeric feature groups.

## Phase 1: Configuration & Constants

- [ ] **Define column type constants** in [src/config.py](../../src/config.py) - Add `ORDINAL_COLUMNS`, `NOMINAL_COLUMNS`, `BINARY_NUMERIC_COLUMNS` lists with feature names from dataset spec

- [ ] **Create TypeBasedImputationConfig dataclass** in [src/config.py](../../src/config.py) - Add fields for `type`, `ordinal_strategy`, `nominal_strategy`, `ordinal_params`, `nominal_params`; implement `__post_init__()` validation

- [ ] **Add configuration validation** in [src/config.py](../../src/config.py) - Reject invalid combinations (e.g., `nominal_strategy='knn'`); provide clear error messages

## Phase 2: Core Implementation

- [ ] **Implement TypeBasedImputation class** in [src/preprocessing/imputation.py](../../src/preprocessing/imputation.py) - Create class inheriting from `ImputationStrategy` with fit/transform methods that delegate to composed strategies

- [ ] **Implement fit() method** - Create strategy instances for each type, fit each on respective columns

- [ ] **Implement transform() method** - Apply each strategy to its columns in sequence, build combined imputed DataFrame

- [ ] **Add error handling** - Handle missing columns, all-missing columns, unfitted state

- [ ] **Update create_imputation_strategy() factory** in [src/preprocessing/imputation.py](../../src/preprocessing/imputation.py) - Add branch for `type_based` strategy type; instantiate TypeBasedImputationConfig and TypeBasedImputation

## Phase 3: Configuration Examples

- [ ] **Create config_type_based_basic.yaml** in [examples/](../../examples/) - Example with ordinal_strategy='mean', nominal_strategy='mode'

- [ ] **Create config_type_based_knn.yaml** in [examples/](../../examples/) - Example with ordinal_strategy='knn', nominal_strategy='mode', including KNN parameters

- [ ] **Optional: Update existing configs** - Update baseline/knn/xgboost configs to use type-based imputation, or keep single-strategy for backward compatibility

## Phase 4: Testing

- [ ] **Test TypeBasedImputation instantiation** - Verify class can be created and initialized with column lists and strategy names

- [ ] **Test fit() behavior** - Fit on sample training data; verify all strategy instances are created and fitted

- [ ] **Test transform() behavior** - Transform test data; verify correct strategies applied to correct columns; verify output shape and dtype preservation

- [ ] **Test no data leakage** - Verify fit parameters (e.g., mean, mode) computed only from training data, not test data

- [ ] **Test invalid strategy combinations** - Verify TypeBasedImputationConfig rejects invalid combinations with clear error message

- [ ] **Test MICE handling** - If both strategies are 'mice', verify single MICE instance is used (or document limitation)

- [ ] **Test missing columns** - Verify graceful handling if column lists reference non-existent columns

- [ ] **Test edge cases** - All-missing columns, empty column lists, no missing values in data

## Phase 5: Documentation & Integration

- [ ] **Update SYSTEM_DESIGN.md** - Add section explaining type-based imputation architecture, decision rationale, and usage examples

- [ ] **Add docstrings** - Comprehensive docstrings for TypeBasedImputation class and methods (parameters, returns, raises, examples)

- [ ] **Update PROBLEM_DESCRIPTION.md** if needed - Reference type-based imputation for users choosing strategies

- [ ] **Add integration test** - End-to-end test with config loading, training engine integration, verifying type-based imputation works in full pipeline

## Implementation Order

1. Phase 1: Configuration & Constants (foundation for all downstream work)
2. Phase 2: Core Implementation (main feature)
3. Phase 3: Configuration Examples (user guidance)
4. Phase 4: Testing (validation and regression prevention)
5. Phase 5: Documentation & Integration (completeness and discoverability)

## Acceptance Criteria

Each task should be:
- **Testable**: Clear success criteria that can be verified (e.g., "TypeBasedImputation exists and inherits from ImputationStrategy")
- **Atomic**: Can be completed independently (though Phase 1 should complete before Phase 2)
- **Specific**: Focused on a single deliverable (e.g., not "implement everything")
- **Actionable**: Has clear implementation steps (file paths, method names, etc.)

### Definition of Done for Feature

✅ `TypeBasedImputation` class exists and implements full `ImputationStrategy` interface  
✅ Configuration accepts `type: 'type_based'` with `ordinal_strategy` and `nominal_strategy`  
✅ Invalid strategy combinations raise validation errors at config load time  
✅ Unit tests cover core functionality (fit, transform, data leakage, validation)  
✅ Example YAML configs demonstrate usage patterns  
✅ Column type classification verified against problem definition  
✅ Documentation updated with architecture explanation and usage guide  
✅ Backward compatibility maintained: single-strategy configs continue to work  

