# Baseline Imputation Analysis & Improvement Recommendations

**Date**: January 12, 2026  
**Current Baseline AUROC**: 0.8441 (H1N1: 0.8356, Seasonal: 0.8525)  
**Configuration**: `mean` imputation strategy

---

## Executive Summary

The baseline uses **simple mean imputation for all numeric columns and mode imputation for non-numeric columns**. While this achieves respectable AUROC (~0.84), there are **significant problems** with this approach given the dataset's characteristics:

1. **Categorical features imputed with mean**: Employment, demographic, and geographic categorical features contain NaN but are treated as numeric
2. **High-missingness features**: 8 features have >40% missing data (employment_industry/occupation, doctor recommendations, health_insurance) but are handled generically
3. **Information loss**: Mean/mode imputation discards the fact that values were missing—which may itself be predictive
4. **Feature-type agnostic**: No distinction between ordinal features (opinions, age_group) that should preserve order vs. nominal (employment, race) that shouldn't
5. **Binary features degraded**: doctor_recc_h1n1/seasonal (strong predictors) lose critical information when imputed to mean values (~0.46)

---

## Problem 1: Categorical Features Treated as Numeric

### Current Behavior
The baseline's `MeanImputation` only fills numeric columns:
```python
self.fit_params['column_means'] = X.select_dtypes(include=[np.number]).mean()
```

This leaves **categorical columns with NaN unchanged**, then they're later one-hot encoded with NaN as a silent missing category.

### Affected Features & Impact

| Feature | Type | Missing % | Current Issue |
|---------|------|-----------|---------------|
| `employment_industry` | Categorical | 49.9% | NaN becomes its own category |
| `employment_occupation` | Categorical | 50.4% | NaN becomes its own category |
| `race` | Categorical | 0% | OK |
| `sex` | Categorical | 0% | OK |
| `education` | Categorical | 5.3% | NaN becomes its own category |
| `employment_status` | Categorical | 5.5% | NaN becomes its own category |
| `rent_or_own` | Categorical | 7.6% | NaN becomes its own category |
| `income_poverty` | Categorical | 16.6% | NaN becomes its own category |
| `hhs_geo_region` | Categorical | 0% | OK |
| `census_msa` | Categorical | 0% | OK |
| `marital_status` | Categorical | 5.3% | NaN becomes its own category |

### Problems
- **Silent NaN encoding**: One-hot encoder treats NaN as a separate category without explicit acknowledgment
- **Wasted feature space**: Creates `employment_industry_NaN` and `employment_occupation_NaN` features
- **Loss of domain knowledge**: For employment features (~50% missing), the missingness likely indicates unemployment, which should be explicitly modeled
- **Inconsistent with domain**: Employment missingness is **systematic** (MCAR—Missing Completely At Random is false; it's likely MAR—Missing At Random, conditional on employment status)

---

## Problem 2: High-Missingness Features Lose Critical Information

### Doctor Recommendations (8% missing, but very predictive)
```
doctor_recc_h1n1: 2,160 rows missing (8.1%)
doctor_recc_seasonal: 2,160 rows missing (8.1%)
```

**Current approach**: Mean imputation fills with ~0.46 (the average rate of recommendations)

**Problem**: 
- These are **strong predictors** of vaccine uptake (people who get doctor recommendations are more likely to vaccinate)
- Mean imputation destroys the signal: imputing 0.46 for missing values dilutes the strong positive correlation
- Better: Create an indicator feature `missing_doctor_recc` to capture that information was unavailable

### Health Insurance (46% missing)
```
health_insurance: 12,274 rows missing (45.96%)
```

**Current approach**: Mean imputation (~0.80)

**Problem**:
- 46% missing likely indicates: (a) non-response bias, (b) uninsured respondents, or (c) structural issue in survey
- Mean imputation assumes missing = 0.80, which is unrealistic
- Better: Model missing as a separate indicator; treat it as information that may correlate with vaccine hesitancy or healthcare access issues

### Employment Features (~50% missing)
```
employment_industry: 13,330 rows missing (49.9%)
employment_occupation: 13,470 rows missing (50.4%)
```

**Current approach**: One-hot encoded with NaN as a category

**Problem**:
- 50% missing ~= "not employed" (unemployment = 3.5% in 2009; missing ~= unemployment + non-response + not in labor force)
- NaN encoding creates 21 + 23 = 44 binary features from only ~9,400 employed respondents
- Severe feature sparsity: most employment features are nearly all-zeros, containing little signal
- Better: Replace with single binary `is_employed` feature, then apply employment features only if employed

---

## Problem 3: No Distinction Between Feature Types

The baseline applies mean/mode imputation uniformly, but features have different semantic types:

### Ordinal Features (Should Preserve Order)
```
opinion_h1n1_vacc_effective, opinion_h1n1_risk, opinion_h1n1_sick_from_vacc (1-5 scales)
opinion_seas_vacc_effective, opinion_seas_risk, opinion_seas_sick_from_vacc (1-5 scales)
h1n1_concern, h1n1_knowledge (ordinal scales)
age_group, education, income_poverty (categorical ordinal)
```

**Current**: Mean imputation fills with ~2.5-3.0 (middle of scale)

**Problem**: Mean imputation doesn't respect ordinal structure for strings like 'age_group'. After one-hot encoding, loses ordinal relationships.

**Better**: 
- Use ordinal encoding for ordinal features (preserves relationships: 18-34 < 35-44 < 45-54, etc.)
- Apply KNN imputation on ordinal-encoded features to impute with values near neighbors
- Maintains the semantic order in the feature space

### Nominal/Categorical Features (No Order)
```
race, sex, marital_status, rent_or_own, employment_status, hhs_geo_region, census_msa, employment_industry, employment_occupation
```

**Current**: One-hot with NaN as a category

**Better**: 
- Apply mode imputation on categorical columns
- OR use `FlagAsMissingImputation` to create `missing_*` indicators
- For high-missingness features (employment, health_insurance), consider domain-aware imputation:
  - employment_industry/occupation missing → impute with "Unknown/Not Employed"
  - health_insurance missing → create separate category or flag

### Binary Features (0/1)
```
behavioral_*, doctor_recc_*, chronic_med_condition, child_under_6_months, health_worker, health_insurance, household_adults, household_children
```

**Current**: Mean imputation fills with proportions (0.0-1.0)

**Problem**: Mean imputation with values like 0.46 is reasonable for binary (it's the probability), but:
- Loses information that value was missing (which might correlate with non-response bias)
- For doctor recommendations and health insurance, missingness is informative

**Better**: 
- For sparse missing (<5%): simple mean or median imputation is OK
- For high missing (>40%): use `FlagAsMissingImputation` to preserve missingness signal

---

## Problem 4: Information Loss from Mean/Mode Imputation

Mean/mode imputation is a **first-order** technique that assumes MCAR (Missing Completely At Random). It has known limitations:

1. **Reduces variance**: Imputing with column means artificially compresses the distribution
2. **Creates false certainty**: All missing values filled with the same value → reduces model uncertainty
3. **No multivariate relationships**: Doesn't consider correlations between features
4. **Loses MAR information**: When missingness depends on other features (e.g., employment missing → unemployed), mean imputation ignores this

### Example: Doctor Recommendations
- **Before imputation**: Some respondents have doctor recommendation = 0/1, others are NaN
- **After mean imputation**: All NaN → 0.46 (proportion of "yes" responses)
- **Signal destroyed**: Model sees artificial certainty where uncertainty existed; loses information that some respondents' doctor status wasn't assessed

---

## Current Baseline Results Analysis

| Metric | H1N1 | Seasonal | Mean |
|--------|------|----------|------|
| AUROC | 0.8356 | 0.8525 | 0.8441 |
| CV Std | ±0.0037 | ±0.0055 | — |
| ECE (Calibration) | 0.0090 | 0.0135 | — |

**Interpretation**:
- Baseline achieves **competition baseline** (80-82% expected range)
- But AUROC is **above** expected, suggesting room for improvement hasn't been fully explored
- Calibration is reasonable (ECE < 2%), suggesting probability estimates are mostly well-calibrated despite mean imputation's coarseness

**Why baseline still works well**:
- Logistic regression is robust; mean imputation's variance reduction doesn't hurt badly
- Doctor recommendations are strong enough signal that even mean imputation captures their effect
- Most features have minimal missing data (<2%), so mean imputation's impact is limited

---

## Recommended Improvements

### Quick Wins (Low-Risk, Easy to Implement)

#### 1. **Type-Based Imputation** (Ordinal vs. Nominal)
Apply different strategies to different feature types:

```yaml
imputation:
  type: 'type_based'
  ordinal_strategy: 'mean'  # For opinions, concern, age_group, education, income
  nominal_strategy: 'mode'  # For race, sex, marital_status, employment_status, rent_or_own
  ordinal_params: {}
  nominal_params: {}
```

**Expected impact**: +0.5-1.0% AUROC (Seasonal more than H1N1, since seasonal has more balanced classes)

**Reasoning**: 
- Mode imputation for nominal features is more semantic (a category, not an average)
- Preserves the categorical structure during one-hot encoding
- Prevents artificial NaN categories

---

#### 2. **Flag-as-Missing for High-Missingness Features**
Preserve missingness as a signal for features with >5% missing:

```python
from src.preprocessing.imputation import FlagAsMissingImputation

# For doctor_recc_*, health_insurance, employment_*, income_poverty
imputer = FlagAsMissingImputation(base_strategy='mode')
```

**Creates indicator columns**:
- `missing_doctor_recc_h1n1`, `missing_doctor_recc_seasonal`
- `missing_health_insurance`
- `missing_income_poverty`
- `missing_education`
- etc.

**Expected impact**: +1-2% AUROC
- Doctor recommendation missingness likely indicates "not recommended" (0), creating signal
- Health insurance missingness correlates with uninsured/low-SES, which relates to vaccine hesitancy
- Income missingness indicates non-response bias, which may correlate with vaccine behavior

---

#### 3. **Domain-Aware Employment Handling**
Replace employment industry/occupation with a single binary feature:

```python
# In preprocessing:
# If employment_industry or employment_occupation is NaN → is_employed = 0
# Otherwise → is_employed = 1
```

**Why**:
- Employment feature sparsity (50% missing) makes the 44 one-hot features nearly useless
- Missingness is systematic (indicates unemployment or non-applicability)
- Single binary captures the signal more efficiently

**Expected impact**: +0.5-1.0% AUROC (less direct signal, but cleaner features)

---

### Medium Complexity (Moderate Risk, Better Performance)

#### 4. **KNN Imputation for Ordinal Features**
Use KNN instead of mean for ordinal and binary features:

```yaml
imputation:
  type: 'type_based'
  ordinal_strategy: 'knn'
  nominal_strategy: 'mode'
  ordinal_params:
    n_neighbors: 5
    weights: 'distance'
  nominal_params: {}
```

**Why KNN is better**:
- Finds similar respondents (based on non-missing features) and uses their values
- Respects multivariate relationships (e.g., older respondents' opinions similar to older neighbors)
- Preserves variance better than mean
- Captures local structure in feature space

**Expected impact**: +1-2% AUROC
- Especially beneficial for opinion features and doctor recommendations
- KNN handles the systematic missingness in doctor_recc better than mean

---

#### 5. **MICE Imputation for Full Multivariate Imputation**
Multivariate Imputation by Chained Equations (MICE):

```yaml
imputation:
  type: 'mice'
  parameters:
    max_iter: 10
    random_state: 42
```

**Why MICE**:
- Iteratively models relationships between features
- Handles MAR (Missing At Random) patterns well
- Creates multiple imputed datasets and combines predictions
- Most sophisticated approach; gold standard in statistics

**Expected impact**: +2-3% AUROC
- Best for capturing complex relationships
- Computationally more expensive (~10x slower)
- Risk: can overfit if patterns are spurious

---

### Implementation Priority

**Phase 1 (Quick Wins)** - Implement immediately:
1. Type-based imputation (ordinal mean + nominal mode)
2. Flag-as-missing for doctor_recc and health_insurance

**Expected cumulative gain**: +1.5-3.0% AUROC → **0.86-0.87**

**Phase 2 (Medium Effort)**:
1. Domain-aware employment handling (replace with is_employed)
2. KNN imputation for ordinal features

**Expected cumulative gain**: +2-4% AUROC → **0.86-0.88**

**Phase 3 (Advanced)**:
1. MICE imputation (if Phase 2 doesn't converge on improvements)
2. Interaction terms between missingness flags and original features

**Expected cumulative gain**: +3-5% AUROC → **0.87-0.89**

---

## Implementation Roadmap

### Already Implemented Infrastructure
✅ Type-based imputation (config and logic in place)
✅ Flag-as-missing imputation (scaffolding exists)
✅ KNN imputation (sklearn wrapper ready)
✅ MICE imputation (scaffolding exists)

### Files to Modify

1. **[examples/config_type_based_basic.yaml](examples/config_type_based_basic.yaml)** - Already exists; use as template
2. **[src/preprocessing/imputation.py](src/preprocessing/imputation.py)** - Review `TypeBasedImputation` and `FlagAsMissingImputation`
3. **[main.py](main.py)** - Switch config to use type-based instead of simple mean

### Quick Configuration Change
Switch from `config_baseline.yaml` to `config_type_based_basic.yaml`:

```bash
python main.py --config examples/config_type_based_basic.yaml --seed 42
```

---

## Expected Results by Strategy

| Strategy | Ordinal | Nominal | Doctor Recc | Health Ins | Expected Δ AUROC |
|----------|---------|---------|-------------|-----------|------------------|
| Baseline (Mean) | Mean | Mode | Mean → 0.46 | Mean → 0.80 | 0.0% (reference) |
| Type-Based | Mean | Mode | Mean → 0.46 | Mean → 0.80 | +0.5-1.0% |
| + Flag Missing | Mean | Mode | **Flag + Mean** | **Flag + Mean** | +1.5-2.5% |
| + KNN | **KNN** | Mode | **KNN + Flag** | **KNN + Flag** | +2.5-3.5% |
| + MICE | **MICE** | **MICE** | **MICE + Flag** | **MICE + Flag** | +3.5-5.0% |

---

## Summary: Key Takeaways

### Problems with Baseline Mean Imputation
1. **No type awareness**: Treats all features the same (ordinal, nominal, binary)
2. **Categorical NaN leakage**: Categorical columns silently encoded with NaN as a category
3. **Information loss**: Discards the fact that values were missing (which may be predictive)
4. **High-missingness features**: doctor_recc (8%), health_insurance (46%), employment (50%) lose critical signal
5. **Variance reduction**: Mean imputation artificially reduces variance and feature variability

### Path Forward
- **Quick win**: Switch to type-based imputation (ordinal mean + nominal mode)
- **Medium effort**: Add flag-as-missing for >5% missing features
- **Advanced**: Implement KNN or MICE for multivariate imputation
- **Realistic target**: 0.87-0.88 AUROC (3-4% improvement) with thoughtful imputation strategy
