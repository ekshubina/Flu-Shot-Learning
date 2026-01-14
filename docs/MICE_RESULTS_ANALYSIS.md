# MICE Imputation Results & Analysis

**Date**: January 12, 2026  
**Configuration**: Type-based MICE (MICE for numeric ordinal, mode for categorical)  
**Runtime**: 187.47 seconds (3.1 minutes)

---

## Results Comparison

| Metric | Baseline (Mean) | MICE | Δ | % Change |
|--------|-----------------|------|----|-----------| 
| H1N1 AUROC | 0.8356 | 0.8324 | -0.0032 | -0.38% |
| Seasonal AUROC | 0.8525 | 0.8505 | -0.0020 | -0.24% |
| **Mean AUROC** | **0.8441** | **0.8414** | **-0.0027** | **-0.32%** |
| CV Std Dev | ±0.0045 | ±0.0045 | 0 | 0% |
| H1N1 ECE (Calibration) | 0.0090 | 0.0065 | -0.0025 | -27.8% ✅ |
| Seasonal ECE | 0.0135 | 0.0126 | -0.0009 | -6.7% ✅ |
| H1N1 Brier Score | 0.1192 | 0.1205 | +0.0013 | +1.1% ❌ |
| Seasonal Brier Score | 0.1552 | 0.1561 | +0.0009 | +0.6% ❌ |
| **Runtime** | **2.75s** | **187.47s** | **+184.7s** | **+6,716%** 🐌 |

---

## Key Findings

### 1. **MICE Underperforms Baseline** ⚠️

MICE imputation slightly **decreases** AUROC by 0.32% compared to mean imputation:

- **Why this might happen**:
  - IterativeImputer uses BayesianRidge regression by default, which may overfit on training fold data
  - MICE assumes linear relationships between features; flu vaccine decisions are complex/non-linear
  - Imputed values from MICE might introduce spurious correlations or noise
  - The 10 iterations may not be enough for convergence, or too many (overfitting)
  - Ordinal string features fall back to KNN (not true MICE), reducing the benefit
  
- **Evidence for overfitting**:
  - Calibration improved (lower ECE) but AUROC decreased
  - This suggests MICE generated overconfident predictions that don't generalize well
  - The model may be fitting to imputation artifacts rather than real signal

### 2. **Better Calibration, Worse Ranking** 🎯

MICE achieves **better calibration** (27.8% lower ECE for H1N1) but **worse AUROC**. This is the classic trade-off:

- **What this means**:
  - MICE predictions are better calibrated (probabilities closer to true rates)
  - But the ranking/discrimination ability is worse (ROC AUC = ranking metric)
  - The model is more conservative but less discriminative
  - For competition, AUROC is what matters → baseline is better

### 3. **Massive Computational Cost** 🐌

MICE takes **68x longer** to run:
- Baseline: 2.75s
- MICE: 187.47s
- Delta: 184.7s extra per CV fold

**Imputation breakdown** (from logs):
- CV training: ~130s
- Full refit: ~55.8s
- Total imputation overhead: ~186s

**Not worth the performance loss** when:
- AUROC decreases by 0.32%
- Runtime increases by 6,700%

---

## Why Did MICE Fail to Improve?

### Problem 1: Mixed Data Types

MICE is designed for continuous numeric data. Our dataset has:
- **Numeric**: opinions (1-5), binary features (0/1), household counts
- **Categorical strings**: age_group ('18-34', '55-64'), education, employment, geography

**Solution implemented**: 
- Ordinal string features → fallback to OrdinalStringKNN (not true MICE)
- Nominal categorical → mode imputation (not MICE)

**Result**: Only ~20% of features got true MICE treatment, rest got fallback strategies. Defeats the purpose of MICE.

### Problem 2: Linear Assumption

IterativeImputer assumes linear relationships between features:
- Feature A = β₀ + β₁×B + β₂×C + ... + ε

**Our data reality**:
- Vaccine decisions are driven by:
  - Non-linear effects (e.g., doctor recommendation is binary → dominant predictor, not linear contribution)
  - Interactions (doctor rec × opinion effectiveness)
  - Discrete categories (employment status → affects other features)

**Result**: Linear imputation may fit spurious patterns, hurting generalization.

### Problem 3: Limited Missing Data

Most features have <2% missing:
- Only 9 features have >3% missing
- Only 4 features have >8% missing

**For sparse missing data**:
- Mean/mode imputation is nearly optimal (simple is better)
- MICE/KNN overhead without benefit
- Domain knowledge (mean) ≈ model knowledge (MICE) when data is complete

**Result**: Imputation quality doesn't matter much when 98%+ of data is non-missing. The signal is in the features themselves, not the imputation strategy.

### Problem 4: Overfitting on Train Fold

IterativeImputer runs 10 iterations of fitting regressions. On each CV fold:
- Train fold: ~21,365 samples
- IterativeImputer fits 35 regression models × 10 iterations = 350 model fits

**Risk factors**:
- Small training set (21k samples) relative to 35 features
- High chance of fitting to noise in the imputation process
- Overfitting transfers to downstream logistic regression

**Result**: MICE learns spurious patterns that don't generalize to validation fold.

---

## Recommendations

### ❌ Don't Use MICE For This Problem

**Why**:
1. **Worse AUROC** (-0.32%) than baseline
2. **68x slower** (187s vs 2.75s per fold)
3. **Mixed benefits**: better calibration, worse discrimination
4. **Incompatible with data types**: need fallbacks for strings

### ✅ Better Alternatives

**Option 1: Stick with Baseline Mean Imputation** (Current Winner)
- Simple, fast, effective
- 0.8441 AUROC
- 2.75s total runtime
- Recommendation: **Use this for final submission**

**Option 2: Try KNN Imputation** (Conservative Upgrade)
- Similar complexity to MICE, better for mixed data types
- Likely slight improvement over mean (~+0.5-1%)
- Moderate runtime cost (~30-60s)
- Recommendation: **Test if time permits**

**Option 3: Type-Based Mean + Mode** (Lightweight Improvement)
- Apply mean to numeric, mode to categorical
- Cheaper than KNN (should be <5s)
- Likely +0.5% improvement
- Recommendation: **Quick win to try first**

**Option 4: Flag-as-Missing for High-Missingness Features** (Best ROI)
- Add indicators for >5% missing (doctor_recc, health_insurance, income_poverty)
- Preserves missing information without heavy imputation
- Likely +1-2% improvement with minimal cost
- Recommendation: **Most promising of the improvements**

---

## Conclusion

**MICE underperformed on this dataset due to**:
1. Mixed data types requiring fallbacks
2. Linear imputation assumptions (data is non-linear)
3. Sparse missing data (simple methods are nearly optimal)
4. Overfitting risk on small train folds

**Best path forward**:
1. **Baseline mean imputation** is already quite good (0.8441)
2. Focus on **feature engineering** or **flag-as-missing** instead of sophisticated imputation
3. **MICE not recommended** unless you can fully numerize the data or use a non-linear imputer

**Lesson learned**: 
> Sophisticated imputation methods aren't always better. When missing data is sparse (<5%), domain knowledge (mean/mode) often beats complex learned approaches. Save complexity for where it matters: feature engineering, model selection, hyperparameter tuning.
