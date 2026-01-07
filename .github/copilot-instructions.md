## Project Overview

This is a machine learning competition project from [DrivenData](https://www.drivendata.org/competitions/66/flu-shot-learning/) focused on **multilabel classification** to predict H1N1 and seasonal flu vaccine uptake based on survey data from the National 2009 H1N1 Flu Survey.

### Key Problem Characteristics

- **Multilabel, not multiclass**: Each respondent can receive neither, one, or both vaccines independently
- **Two target variables**: `h1n1_vaccine` and `seasonal_vaccine` (both binary: 0 or 1)
- **Evaluation metric**: ROC AUC (mean of scores for each vaccine) - predictions must be **probabilities (0.0-1.0), not binary labels**
- **Feature set**: 35 features covering demographics, opinions, health behaviors, medical recommendations, and health status
- **Data source**: CDC's National 2009 H1N1 Flu Survey (survey responses, not clinical data)

## Architecture & Data Flow

The project structure is straightforward:
- **main.py**: Central analysis and model training script (currently empty - where all work happens)
- **requirements.txt**: Python dependencies (currently empty - will need pandas, scikit-learn, etc.)
- **docs/PROBLEM_DESCRIPTION.md**: Complete feature documentation and evaluation details

### Feature Categories

Organize feature engineering and EDA around these logical groupings:

1. **Opinions** (5 features): `opinion_h1n1_*`, `opinion_seas_*` - ordinal scales (1-5)
2. **Behavioral** (7 features): `behavioral_*` - binary preventive actions
3. **Medical** (4 features): `doctor_recc_*`, `chronic_med_condition`, `health_worker`, `health_insurance`
4. **Demographics** (8 features): `age_group`, `education`, `race`, `sex`, `income_poverty`, `marital_status`, `rent_or_own`, `employment_status`
5. **Household** (2 features): `household_adults`, `household_children` (top-coded to 3)
6. **Geographic** (2 features): `hhs_geo_region`, `census_msa`
7. **Concern & Knowledge** (2 features): `h1n1_concern`, `h1n1_knowledge` (ordinal scales)
8. **Employment** (2 features): `employment_industry`, `employment_occupation` (categorical with NaN)

## Critical Implementation Details

### Multilabel Problem Setup

Since this is multilabel (not multiclass):
- Train **two separate models** (one for each vaccine) OR one model that outputs two probabilities
- Do NOT assume probabilities sum to 1 for each row
- Use `sklearn.metrics.roc_auc_score(..., average="macro")` to compute mean AUC
- Each vaccine prediction is independent

### Feature Encoding

- **Ordinal features** (`h1n1_concern`, `h1n1_knowledge`, opinion_*): Preserve order; consider ordinal encoding or keep as-is
- **Categorical features** (demographics, geography): Use one-hot encoding or categorical encoding depending on cardinality
- **Binary features**: Already encoded as 0/1
- **Missing values**: `employment_industry` and `employment_occupation` contain NaN - handle with imputation or drop

### Model Training Workflow

1. Load data, separate features and targets
2. Handle missing values (check both features and targets)
3. Encode categorical variables
4. Scale numerical features if needed (depends on model choice)
5. Train two independent binary classifiers (or one multilabel model)
6. Generate probability predictions (0.0-1.0 range)
7. Evaluate with ROC AUC and create submission CSV with format: `respondent_id,h1n1_vaccine,seasonal_vaccine`

## Key Files & Patterns

- **[docs/PROBLEM_DESCRIPTION.md](docs/PROBLEM_DESCRIPTION.md)**: Authoritative source for feature definitions and submission format
- **[README.md](README.md)**: Competition context and resources

## Development Notes

- No existing models or experiments in the repo - this is a starting template
- Focus on probability calibration (ROC AUC rewards well-calibrated estimates, not just rankings)
- Doctor recommendation (`doctor_recc_h1n1`, `doctor_recc_seasonal`) is likely a strong predictor - explore its correlation with targets
- Handle class imbalance if one vaccine is significantly more/less adopted than the other
- Missing data in employment features may indicate unemployment or non-applicability
