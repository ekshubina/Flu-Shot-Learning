# Type-Based Imputation Implementation Plan

## Overview

Implement type-based imputation that applies different missing value strategies to ordinal, nominal, and binary-numeric features independently. This enables flexible strategy combinations: ordinal columns can use mean/median/knn while nominal columns use mode, with full parameter control via YAML configuration. The implementation leverages existing strategy classes (MeanImputation, ModeImputation, KNNImputation) as composable components, avoiding code duplication.

## Goals

1. Support applying different imputation strategies to different feature types (ordinal, nominal, binary-numeric)
2. Enable flexible strategy combinations via YAML: `ordinal_strategy: 'mean'`, `nominal_strategy: 'mode'`, etc.
3. Allow per-strategy parameter customization (e.g., `ordinal_params: {n_neighbors: 5}`)
4. Maintain data leakage prevention: fit on training data only, apply to test/validation splits
5. Reuse existing ImputationStrategy classes without modification
6. Add validation to reject invalid strategy combinations (e.g., KNN for nominal columns)

## Non-Goals

- Implement new imputation algorithms (use existing strategies)
- Support MICE with per-type customization (MICE is monolithic, imputes jointly)
- Auto-detect feature types at runtime (use fixed column lists from problem definition)
- Break backward compatibility with single-strategy ImputationConfig

## Implementation Steps

### Phase 1: Configuration & Constants

1. Add column type constants to [src/config.py](src/config.py):
   - `ORDINAL_COLUMNS`: h1n1_concern, h1n1_knowledge, opinion_* (6), age_group, education, income_poverty (8 total)
   - `NOMINAL_COLUMNS`: race, sex, marital_status, rent_or_own, employment_status, hhs_geo_region, census_msa, employment_industry, employment_occupation (9 total)
   - `BINARY_NUMERIC_COLUMNS`: behavioral_*, doctor_recc_*, chronic_med_condition, health_worker, health_insurance, household_adults, household_children (11 total)

2. Create `TypeBasedImputationConfig` dataclass in [src/config.py](src/config.py):
   - Fields: `type: 'type_based'`, `ordinal_strategy: str`, `nominal_strategy: str`, `ordinal_params: Dict[str, Any]`, `nominal_params: Dict[str, Any]`
   - Validation: Reject `nominal_strategy='knn'` or any unsupported combinations
   - Document strategy options: ordinal/binary → 'mean', 'median', 'knn', 'mice'; nominal → 'mode', 'mice'

### Phase 2: Implementation

3. Implement `TypeBasedImputation` class in [src/preprocessing/imputation.py](src/preprocessing/imputation.py):
   - Constructor takes column lists and strategy names
   - `fit(X)`: Create strategy instances (`MeanImputation(ORDINAL_COLUMNS)`, `ModeImputation(NOMINAL_COLUMNS)`, etc.) and fit each
   - `transform(X)`: Apply each strategy to its designated columns in sequence, building final imputed DataFrame
   - Handle edge cases: columns with all missing, mismatched column sets between train and test

4. Update imputation factory function in [src/preprocessing/imputation.py](src/preprocessing/imputation.py):
   - Modify `create_imputation_strategy()` to detect `type_based` strategy type
   - Instantiate `TypeBasedImputationConfig` from YAML
   - Create and return `TypeBasedImputation` instance with appropriate column lists and strategy names

### Phase 3: Configuration & Examples

5. Create example YAML configs demonstrating type-based imputation in [examples/](examples/):
   - `config_type_based_basic.yaml`: ordinal_strategy='mean', nominal_strategy='mode' (default)
   - `config_type_based_knn.yaml`: ordinal_strategy='knn', nominal_strategy='mode' (advanced)
   - Update existing baseline/knn/xgboost configs to use type-based if desired, or keep single-strategy for backward compatibility

### Phase 4: Testing & Documentation

6. Add unit tests in `tests/preprocessing/test_imputation.py`:
   - Test TypeBasedImputation applies correct strategies to correct columns
   - Test fit/transform produces expected output shapes and no data leakage
   - Test validation rejects invalid strategy combinations
   - Test parameter passing to underlying strategies

7. Update documentation:
   - Add section to [docs/SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) explaining type-based imputation architecture
   - Document column type classification and strategy selection guide

## Success Criteria

1. `TypeBasedImputation` class exists and implements `ImputationStrategy` interface
2. Configuration loading accepts `type: 'type_based'` with ordinal/nominal strategy specifications
3. Invalid strategy combinations raise clear validation errors at config load time
4. Unit tests demonstrate correct strategies applied to correct column groups
5. Example configs show type-based usage patterns
6. No data leakage: fit parameters computed only on training data
7. Backward compatibility maintained: single-strategy configs still work

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Incomplete column classification | Data leakage or incorrect imputation | Validate column lists against actual feature names; add tests |
| Strategy ordering affects results | Different imputation quality depending on application order | Document strategy application order; test with different orderings |
| Parameter mismatch to strategy | Config validation errors at runtime | Validate parameter keys against strategy signature at config load time |
| MICE behavior with mixed types | Unclear if MICE should be per-type or joint | Document design decision: MICE is monolithic, applied to all columns jointly |
| Column presence mismatch | TypeBasedImputation references columns not in data | Add defensive checks in fit/transform for missing columns |

