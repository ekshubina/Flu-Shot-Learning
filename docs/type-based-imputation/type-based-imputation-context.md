# Type-Based Imputation Context & References

## Key Files

### Files to Modify

| File | Purpose | Changes Needed |
|------|---------|----------------|
| [src/config.py](../../src/config.py) | Configuration dataclasses | Add `ORDINAL_COLUMNS`, `NOMINAL_COLUMNS`, `BINARY_NUMERIC_COLUMNS` constants; create `TypeBasedImputationConfig` dataclass with validation |
| [src/preprocessing/imputation.py](../../src/preprocessing/imputation.py) | Imputation strategies | Implement `TypeBasedImputation` class; update `create_imputation_strategy()` factory function to handle 'type_based' |
| [examples/config_baseline.yaml](../../examples/config_baseline.yaml) | Example configurations | Add or update to demonstrate type-based imputation (optional; backward compatibility allows keeping single-strategy) |

### New Files to Create

| File | Purpose |
|------|---------|
| [examples/config_type_based_basic.yaml](../../examples/config_type_based_basic.yaml) | Example: ordinal=mean, nominal=mode |
| [examples/config_type_based_knn.yaml](../../examples/config_type_based_knn.yaml) | Example: ordinal=knn, nominal=mode |

### Reference Implementation Files

| File | Relevance |
|------|-----------|
| [src/preprocessing/imputation.py](../../src/preprocessing/imputation.py) - Lines 233-292 (MeanImputation) | Pattern for numeric imputation; will be composed by TypeBasedImputation |
| [src/preprocessing/imputation.py](../../src/preprocessing/imputation.py) - Lines 295-356 (ModeImputation) | Pattern for categorical imputation; will be composed by TypeBasedImputation |
| [src/preprocessing/imputation.py](../../src/preprocessing/imputation.py) - Lines 359-447 (KNNImputation) | Pattern for KNN imputation; will be composed by TypeBasedImputation |
| [src/config.py](../../src/config.py) - Lines 48-80 (ImputationConfig) | Existing single-strategy config; TypeBasedImputationConfig parallels this |
| [src/training/engine.py](../../src/training/engine.py) - preprocessing pipeline integration | Shows how imputation is called in the training workflow |

## Feature Type Classification

Based on problem definition and ordinal/nominal semantics:

### Ordinal Features (8 columns)
Features with natural ordering that benefit from ordinal encoding and numeric imputation:

**Concern & Knowledge (2):**
- `h1n1_concern`: 0 (Not at all) → 1 → 2 → 3 (Very)
- `h1n1_knowledge`: 0 (No knowledge) → 1 (A little) → 2 (A lot)

**Opinions (6):** All Likert scale 1-5
- `opinion_h1n1_vacc_effective`
- `opinion_h1n1_risk`
- `opinion_h1n1_sick_from_vacc`
- `opinion_seas_vacc_effective`
- `opinion_seas_risk`
- `opinion_seas_sick_from_vacc`

*Imputation strategies*: mean, median, knn (numeric distance is meaningful due to ordering)

### Nominal Features (9 columns)
Categorical features with no natural ordering; only mode imputation appropriate:

- `race` (categorical)
- `sex` (categorical)
- `marital_status` (categorical)
- `rent_or_own` (categorical)
- `employment_status` (categorical)
- `hhs_geo_region` (ordinal region codes but treated nominally in imputation)
- `census_msa` (nominal)
- `employment_industry` (categorical; 49% missing)
- `employment_occupation` (categorical; 50% missing)

*Imputation strategies*: mode only (KNN inappropriate for unordered categories)

### Binary-Numeric Features (11 columns)
Binary 0/1 features treated as numeric/ordinal for imputation purposes:

**Behavioral (6):**
- `behavioral_antiviral_meds`
- `behavioral_avoidance`
- `behavioral_face_mask`
- `behavioral_large_gatherings`
- `behavioral_outside_home`
- `behavioral_touch_face`

**Medical Recommendations & Status (4):**
- `doctor_recc_h1n1` (8% missing)
- `doctor_recc_seasonal` (8% missing)
- `chronic_med_condition`
- `health_worker`

**Other (1):**
- `health_insurance` (46% missing)

**Household (2):**
- `household_adults` (numeric count, top-coded to 3)
- `household_children` (numeric count, top-coded to 3)

*Imputation strategies*: mean, median, knn (same as ordinal; 0/1 values support numeric distance)

## Architecture Decisions

### Decision 1: Fixed Column Lists vs. Runtime Type Detection
- **Context**: Need to map features to imputation strategies
- **Decision**: Use fixed column lists (`ORDINAL_COLUMNS`, `NOMINAL_COLUMNS`, `BINARY_NUMERIC_COLUMNS`) defined in config
- **Rationale**: Dataset is constant for this project; no need for runtime detection complexity. Matches problem definition.
- **Alternatives Considered**: Auto-detect types from dtype (numeric vs. object); but this conflates storage type with imputation semantics

### Decision 2: Composition vs. Monolithic Implementation
- **Context**: Need to apply different strategies to different columns
- **Decision**: Compose existing strategy classes (`MeanImputation`, `ModeImputation`, etc.) in `TypeBasedImputation`
- **Rationale**: Avoids code duplication; reuses tested implementations; strategy pattern is extensible
- **Alternatives Considered**: Single monolithic class implementing all logic; but harder to maintain and test

### Decision 3: MICE Multi-Type Handling
- **Context**: MICE (Multivariate Imputation by Chained Equations) imputes all columns jointly
- **Decision**: If user specifies `ordinal_strategy='mice'` and `nominal_strategy='mice'`, use single MICE instance on all columns
- **Rationale**: MICE is designed for joint imputation across all feature types; running it per-type defeats the purpose
- **Alternatives Considered**: Allow per-type MICE; but would require monolithic design
- **Note**: Document clearly that mixing MICE with other strategies (e.g., ordinal='mice', nominal='mode') is currently unsupported; user should choose either full-MICE or type-specific non-MICE strategies

### Decision 4: Configuration Validation Timing
- **Context**: Reject invalid combinations like `nominal_strategy='knn'`
- **Decision**: Validate in `TypeBasedImputationConfig.__post_init__()` at config load time
- **Rationale**: Fail fast with clear error message; prevents confusing runtime errors
- **Alternatives Considered**: Validate in TypeBasedImputation.fit(); but delays error detection

## Dependencies

### Internal Dependencies
- `MeanImputation`, `ModeImputation`, `KNNImputation`: Composed by `TypeBasedImputation` to perform actual imputation
- `ImputationStrategy`: Abstract base class that `TypeBasedImputation` implements
- `ImputationConfig`: Dataclass pattern; `TypeBasedImputationConfig` follows same pattern
- `create_imputation_strategy()`: Factory function; will route 'type_based' to new class

### External Dependencies
- `pandas`: DataFrame operations, already used throughout
- `sklearn.impute.KNNImputer`: Used by KNNImputation strategy

## Related Documentation

- [docs/PROBLEM_DESCRIPTION.md](../PROBLEM_DESCRIPTION.md): Feature definitions and missing data patterns
- [docs/SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md): Overall architecture; imputation fits into preprocessing pipeline
- [docs/CONTEXT_REPORT.md](../CONTEXT_REPORT.md): Analysis of missing value distributions (informative missing data in employment features)

## Open Questions

1. **Employment features missing handling**: Columns `employment_industry` and `employment_occupation` have ~50% missing. Should `TypeBasedImputation` support a `flag_as_missing` strategy for nominal columns to preserve missingness informativeness?
   - Current decision: Use `mode` imputation; future enhancement could add flagging

2. **Parameter validation**: How strict should validation be? Should we validate that ordinal_params keys (e.g., 'n_neighbors') are valid for the chosen strategy?
   - Recommendation: Validate at config load time; raise clear error if mismatch

3. **Mixed MICE**: Should allowing `ordinal_strategy='mice', nominal_strategy='mode'` be supported, or only allow MICE for all types or none?
   - Current decision: Unsupported; document that MICE is all-or-nothing

