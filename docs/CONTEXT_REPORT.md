# Flu Shot Learning Competition - Comprehensive Context Report

## Executive Summary

This is a **multilabel binary classification** competition from DrivenData aimed at predicting H1N1 and seasonal flu vaccine uptake. The project is structured as a clean template with minimal implementation—just empty files and raw data. The goal is to build a system that generates probability predictions (0.0-1.0) for each vaccine independently, evaluated using ROC AUC.

---

## 1. CURRENT PROJECT STATE

### Implemented Components
- ✅ **Project structure**: Organized with main.py, requirements.txt, docs/, and data/
- ✅ **Documentation**: Complete problem description and README with resources
- ✅ **Data files**: All training, test, and label files present and accessible
- ✅ **Virtual environment**: Python `.venv` pre-configured

### Empty/Incomplete Components
- ❌ **main.py**: Completely empty (0 lines) - this is where all analysis, feature engineering, model training, and evaluation will happen
- ❌ **requirements.txt**: Only contains `pandas` as a bare dependency (scikit-learn, numpy, etc. need to be added)
- ⚠️ **No models trained**: No pre-trained models or baselines exist yet
- ⚠️ **No feature engineering**: No transformations, encoding, or preprocessing implemented

### Development Context
- Competition **end date**: July 30, 2026 (11:59 p.m. UTC)
- **Status**: Ongoing with 8,563+ participants
- **Repository**: Currently private template (not tracking data files per competition rules)
- **Virtual environment**: Activated and functional

---

## 2. PROBLEM DEFINITION

### Problem Type
**Multilabel Binary Classification** (NOT multiclass)
- Each respondent is independent for each vaccine
- A person can receive neither, one, or both vaccines
- Probabilities for the two vaccines do NOT need to sum to 1 on each row

### Target Variables
Two separate binary targets:
1. **`h1n1_vaccine`**: Did respondent receive H1N1 vaccine? (0 = No, 1 = Yes)
2. **`seasonal_vaccine`**: Did respondent receive seasonal flu vaccine? (0 = No, 1 = Yes)

### Evaluation Metric
- **ROC AUC** (Receiver Operating Characteristic - Area Under Curve)
- Calculated separately for each vaccine, then averaged: `mean([AUC_h1n1, AUC_seasonal])`
- Predictions MUST be **probabilities** (float values from 0.0 to 1.0), not binary labels
- ROC AUC rewards well-calibrated probability estimates, not just correct rankings

### Submission Format
Three columns in CSV format:
```csv
respondent_id,h1n1_vaccine,seasonal_vaccine
26707,0.45,0.62
26708,0.38,0.71
...
```
Each value is a probability between 0.0 and 1.0.

---

## 3. DATASET CHARACTERISTICS

### Data Sources
- **Primary**: CDC's National 2009 H1N1 Flu Survey
- **Source institution**: U.S. Department of Health and Human Services (DHHS), National Center for Health Statistics (NCHS)
- **Data confidentiality**: Proprietary CDC data - NOT to be shared, committed, or exposed publicly

### Dataset Sizes
| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| Training Features | 26,707 | 36 (1 ID + 35 features) | Features for model training |
| Training Labels | 26,707 | 3 (ID + 2 targets) | Ground truth vaccine status |
| Test Features | 26,708 | 36 (1 ID + 35 features) | Features for final predictions |

### Target Distribution (Training Data)
| Metric | H1N1 | Seasonal |
|--------|------|----------|
| Received vaccine (1) | 5,674 (21.2%) | 12,435 (46.5%) |
| Did not receive (0) | 21,033 (78.8%) | 14,272 (53.5%) |
| **Class imbalance ratio** | ~3.7:1 | ~1.15:1 |

**Multilabel breakdown**:
- Both vaccines: 4,697 (17.6%)
- Neither vaccine: 13,295 (49.8%)
- Only H1N1: 977 (3.7%)
- Only seasonal: 7,738 (29.0%)

**Key insight**: H1N1 is significantly rarer (21.2%), indicating strong class imbalance. Seasonal flu adoption is more balanced.

---

## 4. FEATURE OVERVIEW

### Feature Count and Types
- **Total features**: 35 (excluding respondent_id)
- **Data types**: 21 float (numeric/binary), 14 object (categorical/string)

### Feature Categories and Missing Data Patterns

#### A. Opinion Features (6 features)
Ordinal scales representing respondent beliefs about vaccine effectiveness, risk, and side effects:
- `opinion_h1n1_vacc_effective` → Scale: 1-5 (Not effective to Very effective) | Missing: 391 (1.5%)
- `opinion_h1n1_risk` → Scale: 1-5 (Very Low to Very High risk) | Missing: 388 (1.5%)
- `opinion_h1n1_sick_from_vacc` → Scale: 1-5 (Not worried to Very worried) | Missing: 395 (1.5%)
- `opinion_seas_vacc_effective` → Scale: 1-5 | Missing: 462 (1.7%)
- `opinion_seas_risk` → Scale: 1-5 | Missing: 514 (1.9%)
- `opinion_seas_sick_from_vacc` → Scale: 1-5 | Missing: 537 (2.0%)

**Note**: These are ordinal and should preserve order during encoding.

#### B. Behavioral Features (7 features)
Binary indicators of health-protective behaviors:
- `behavioral_antiviral_meds`, `behavioral_avoidance`, `behavioral_face_mask`
- `behavioral_wash_hands`, `behavioral_large_gatherings`, `behavioral_outside_home`
- `behavioral_touch_face`

Missing rates: 0.1% - 0.8% (minimal missing data)

#### C. Doctor Recommendations (2 features)
**Strong predictors** - whether physician recommended each vaccine:
- `doctor_recc_h1n1` | Missing: 2,160 (8.1%)
- `doctor_recc_seasonal` | Missing: 2,160 (8.1%)

**Insight**: These likely have the highest correlation with vaccine uptake. Exact missing pattern (same 2,160 rows) suggests systematic missingness.

#### D. Health Status Features (4 features)
- `chronic_med_condition` → Binary | Missing: 971 (3.6%)
- `child_under_6_months` → Binary contact status | Missing: 820 (3.1%)
- `health_worker` → Binary occupation status | Missing: 804 (3.0%)
- `health_insurance` → Binary coverage status | Missing: 12,274 (45.96%)

**Alert**: `health_insurance` has 46% missing—high missingness may require imputation or special handling.

#### E. Concern & Knowledge (2 features)
Ordinal scales specific to H1N1:
- `h1n1_concern` → Scale: 0-3 (Not at all to Very concerned) | Missing: 92 (0.3%)
- `h1n1_knowledge` → Scale: 0-2 (No knowledge to A lot) | Missing: 116 (0.4%)

#### F. Demographic Features (8 features)
Categorical variables representing respondent background:

| Feature | Unique Values | Missing % | Values |
|---------|---------------|-----------|--------|
| `age_group` | 5 | 0 | 18-34, 35-44, 45-54, 55-64, 65+ Years |
| `education` | 4 | 5.3% | <12 Years, 12 Years, Some College, College Graduate |
| `race` | 4 | 0 | White, Black, Hispanic, Other or Multiple |
| `sex` | 2 | 0 | Female, Male |
| `income_poverty` | 3 | 16.6% | Below Poverty, ≤$75K Above Poverty, >$75K |
| `marital_status` | 2 | 5.3% | Married, Not Married |
| `rent_or_own` | 2 | 7.6% | Own, Rent |
| `employment_status` | 3 | 5.5% | Employed, Unemployed, Not in Labor Force |

**Note**: `income_poverty` has 16.6% missing—substantial but manageable.

#### G. Household Features (2 features)
Count data (top-coded to maximum of 3):
- `household_adults` | Missing: 249 (0.9%)
- `household_children` | Missing: 249 (0.9%)

**Top-coding note**: Values capped at 3 (meaning 3+ adults/children), reducing feature granularity.

#### H. Geographic Features (2 features)
- `hhs_geo_region` → 10 unique regions (coded as random character strings) | No missing
- `census_msa` → 3 categories (Non-MSA, MSA Not Principal City, MSA Principal City) | No missing

**Note**: Region codes are obfuscated (e.g., 'oxchjgsf'); use as categorical features.

#### I. Employment Features (2 features - HIGH MISSING)
Categorical variables with substantial missingness (likely indicates unemployment):
- `employment_industry` → 21 unique categories | Missing: 13,330 (49.9%)
- `employment_occupation` → 23 unique categories | Missing: 13,470 (50.4%)

**Critical insight**: ~50% missing values likely indicate non-employed respondents. Consider creating binary "employed" indicator rather than one-hot encoding with many zeros.

### Summary: Missing Data Patterns

| Severity | Features | Percentage | Handling Strategy |
|----------|----------|-----------|-------------------|
| **None** | age_group, race, sex, hhs_geo_region, census_msa | 0% | No action needed |
| **Minimal** | Most behavioral, opinions, h1n1_concern/knowledge | <2% | Simple imputation (mode/median) or drop rows |
| **Moderate** | education, employment_status, chronic_med_condition | 3-6% | Imputation or create "missing" category |
| **High** | income_poverty, rent_or_own | 8-17% | Create "missing" indicator or use sophisticated imputation |
| **Very High** | doctor_recc_*, health_insurance, employment_* | 46-51% | Special handling: imputation, indicator variables, or domain-specific logic |

---

## 5. ARCHITECTURE INSIGHTS

### Model Design Choices

**Single vs. Dual Model Approach**:
- **Option A (Recommended)**: Train two separate binary classifiers (one per vaccine)
  - More interpretable
  - Can use vaccine-specific features effectively
  - Independent probability outputs
- **Option B**: Single multilabel model (e.g., MultiOutputClassifier)
  - Can capture correlations between vaccines
  - More complex but potentially more elegant

### Key Predictive Signals

Based on feature descriptions and domain knowledge:

1. **Strongest predictors** (likely high correlation):
   - `doctor_recc_h1n1` and `doctor_recc_seasonal` — physician recommendations are decision drivers
   - Opinion features (vaccine effectiveness, risk perception, worry) — beliefs drive behavior
   - Health status features — vulnerable populations (chronic conditions, health workers) incentivize vaccination

2. **Moderate predictors**:
   - Behavioral features — past health behaviors correlate with vaccination intent
   - Demographics — age, education, employment status matter
   - Health insurance — proxy for healthcare access and SES

3. **Consider with caution**:
   - Employment industry/occupation — 50% missing, may need special encoding
   - Census region — geographic variation but obfuscated codes
   - Household composition — weak signal but contextual

### Class Imbalance Handling

**H1N1 vaccine (21.2% positive)**: Significant imbalance
- Use `scale_pos_weight` in gradient boosting, class weights in logistic regression
- Consider ROC AUC metric (good for imbalanced) vs. accuracy (misleading)
- Threshold tuning important for calibration

**Seasonal vaccine (46.5% positive)**: More balanced, fewer special measures needed

---

## 6. CRITICAL IMPLEMENTATION REQUIREMENTS

### Multilabel Framework
✅ **Must remember**: Each vaccine is predicted independently
- Train/evaluate **two separate models** or **one model outputting two probabilities**
- DO NOT assume `P(h1n1) + P(seasonal) = 1.0` per row
- Evaluation: `sklearn.metrics.roc_auc_score(..., average="macro")`

### Feature Encoding Requirements

| Feature Type | Strategy | Notes |
|--------------|----------|-------|
| **Ordinal (opinions)** | Preserve order; consider ordinal encoding or numerical | Keep as 1-5 scale or use ordinal encoder |
| **Categorical low-cardinality** | One-hot encoding | age_group, education, race, sex, etc. |
| **Categorical high-cardinality** | Target encoding, embedding, or grouping | hhs_geo_region (10 values), employment_industry/occupation (21-23 values) |
| **Binary** | Keep as-is | behavioral_*, health_worker, etc. |
| **Numeric** | Consider scaling if using distance-based models | household_adults, household_children |
| **Missing categorical** | Create separate "missing" category OR impute with mode | Critical for doctor_recc_*, employment_* |
| **Missing numeric** | Impute with median/mode/forward-fill | Use SimpleImputer or custom logic |

### Data Preprocessing Workflow

1. **Load data** and align training features with labels
2. **Handle missing values** strategically (see table above)
3. **Encode categorical variables** (one-hot, ordinal, target encoding depending on cardinality)
4. **Create interaction features** (e.g., `doctor_recc * opinion_effectiveness` might be predictive)
5. **Scale numerical features** if using algorithms sensitive to scale (SVM, KNN, neural networks)
6. **Split data** (use stratified k-fold to preserve class distributions)
7. **Train models** with appropriate hyperparameters and class weights
8. **Calibrate probabilities** if needed (importance for ROC AUC)
9. **Generate predictions** on test set as probabilities

### Evaluation Approach

```python
from sklearn.metrics import roc_auc_score

# For validation during training:
y_pred_proba = model.predict_proba(X_val)  # Returns [P(vaccine=0), P(vaccine=1)]
auc_h1n1 = roc_auc_score(y_val_h1n1, y_pred_proba[:, 1])
auc_seasonal = roc_auc_score(y_val_seasonal, y_pred_proba[:, 1])
mean_auc = (auc_h1n1 + auc_seasonal) / 2

# Or using macro average:
auc_score = roc_auc_score([y_val_h1n1, y_val_seasonal], 
                           [y_pred_h1n1, y_pred_seasonal], 
                           average="macro")
```

---

## 7. DATA CONFIDENTIALITY & ETHICS

⚠️ **CRITICAL CONSTRAINTS**:
- CDC data is **proprietary and restricted**
- **DO NOT**:
  - Commit data files to version control (already in .gitignore)
  - Share, upload, or expose data publicly
  - Post on GitHub, Slack, Discord, or public repositories
  - Include data in reports shared outside the team
- **Use data exclusively** for this competition project
- Store locally with **restricted file access** (not on shared drives)

This is part of DrivenData competition's terms of service.

---

## 8. TECHNICAL DEPENDENCIES & SETUP

### Current Environment
- **Python**: 3.7+ (macOS, using `.venv`)
- **Virtual environment**: Pre-configured at `.venv/`
- **Current dependencies**: Only `pandas` (incomplete)

### Recommended Complete requirements.txt

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
xgboost>=1.5.0        # Optional but recommended for strong baseline
lightgbm>=3.3.0       # Optional but recommended
jupyter>=1.0.0        # For interactive development
```

### Development Setup Checklist
- ✅ Activate venv: `source .venv/bin/activate`
- ⚠️ Update requirements.txt with full dependencies
- ⚠️ Install all dependencies: `pip install -r requirements.txt`
- ⚠️ Implement main.py with full ML pipeline

---

## 9. KEY FILES REFERENCE

| File | Status | Purpose |
|------|--------|---------|
| [main.py](main.py) | **Empty** | ML pipeline, training, evaluation |
| [requirements.txt](requirements.txt) | **Incomplete** | Needs pandas, sklearn, numpy, etc. |
| [docs/PROBLEM_DESCRIPTION.md](docs/PROBLEM_DESCRIPTION.md) | ✅ Complete | Full feature definitions, submission format |
| [README.md](README.md) | ✅ Complete | Project overview, resources, citation |
| [data/training_set_features.csv](data/training_set_features.csv) | ✅ 26,707 rows | Training features |
| [data/training_set_labels.csv](data/training_set_labels.csv) | ✅ 26,707 rows | Training targets |
| [data/test_set_features.csv](data/test_set_features.csv) | ✅ 26,708 rows | Test features (no labels) |
| [data/submission_format.csv](data/submission_format.csv) | ✅ Template | Output format reference |

---

## 10. SUMMARY & RECOMMENDATIONS

### Current State Assessment
✅ **Well-structured** clean template with complete documentation and data
❌ **No implementation** — everything in main.py needs to be built
⚠️ **Incomplete dependencies** — requirements.txt needs expansion

### Architectural Recommendations

1. **Build modular ML pipeline** in main.py:
   - Data loading and validation
   - Feature engineering (handle missing values, encoding, scaling)
   - Model selection and training (dual classifier approach recommended)
   - Cross-validation with stratified k-fold
   - Hyperparameter tuning
   - Final evaluation and prediction generation
   - Submission file creation

2. **Start with baseline models**:
   - Logistic Regression (fast, interpretable)
   - Random Forest or Gradient Boosting (strong baseline)
   - Ensemble of the above

3. **Focus on probability calibration**:
   - ROC AUC rewards well-calibrated estimates
   - Use calibration curves to diagnose miscalibration
   - Consider CalibratedClassifierCV if needed

4. **Handle class imbalance systematically**:
   - Use class weights for H1N1 (3.7:1 imbalance)
   - Monitor both AUC and precision-recall for rare class
   - Threshold optimization for final predictions

5. **Feature engineering priorities**:
   - Carefully handle doctor recommendations (strongest signal, but 8% missing)
   - Create "employed" indicator from employment_industry/occupation
   - Ordinal encoding for opinion/concern features
   - Target encoding for high-cardinality categorical variables

### Next Steps
1. Install complete dependency set
2. Implement data loading and validation
3. Perform exploratory data analysis (EDA)
4. Implement feature engineering pipeline
5. Train and evaluate baseline models
6. Iterative improvement with feature engineering and hyperparameter tuning
7. Generate final predictions for submission

---

**Report Generated**: January 7, 2026
**Data Analyzed**: 26,707 training samples, 26,708 test samples
**Feature Coverage**: 35 features across 9 categories with varying missingness
**Problem Type**: Multilabel binary classification (ROC AUC evaluation)
