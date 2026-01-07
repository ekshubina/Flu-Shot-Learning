# System Design: Modular ML Pipeline for AUROC Maximization

## TL;DR

A flexible, component-based pipeline with clear abstraction boundaries allowing independent exploration of preprocessing strategies, feature engineering approaches, and model architectures. Experiment tracking captures all variations, enabling systematic comparison of AUROC results across different tool and method combinations to identify the optimal configuration.

## Requirements

1. **Modular architecture**: Each stage (imputation, encoding, model training, calibration) must be independently swappable to test different approaches.
2. **AUROC maximization**: Optimize for mean ROC AUC across both vaccines; support per-vaccine tuning and threshold analysis.
3. **Experiment tracking**: Log and compare all runs (preprocessing choices, model types, hyperparameters, AUROC results) for reproducibility and trend analysis.
4. **Class imbalance flexibility**: Support multiple strategies (class weights, SMOTE, threshold tuning, ensemble weighting) without pipeline restructuring.
5. **Missing data exploration**: Enable testing different imputation strategies (drop, mean, mode, KNN, iterative, flag-as-feature) with impact measurement.
6. **Feature engineering sandbox**: Support ordinal encoding, one-hot encoding, interaction terms, polynomial features, and custom transforms with easy ablation studies.
7. **Model diversity**: Enable rapid prototyping of logistic regression, tree-based (XGBoost, LightGBM, Random Forest), neural networks, and stacking ensembles.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Data Layer (Load, validate, split)                     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Preprocessing Module (Imputation strategies)           │
│  - Drop missing, mean/median fill, KNN, iterative, ...  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Feature Engineering Module (Encoding & transforms)     │
│  - Ordinal, one-hot, interactions, polynomial, custom   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Training Pipeline (Model training & validation)        │
│  - Stratified CV, class balancing, hyperparameter tuning│
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Calibration Module (Probability calibration)           │
│  - Platt, isotonic, temperature scaling, none           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Evaluation & Experiment Tracking (Metrics, logging)    │
│  - Per-vaccine AUROC, curves, diagnostics, runs DB     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Prediction & Submission (Test inference, formatting)   │
└─────────────────────────────────────────────────────────┘
```

Each module accepts configuration, logs decisions, and outputs results that feed to the next stage. **Experiment tracking layer** captures metadata and metrics across all runs, enabling comparative analysis.

## Components

### 1. Data Module
Loads train/test CSVs, validates shapes and types, splits into stratified train/validation sets (5–10 fold), handles respondent_id alignment, prevents leakage. Exposes clean X_train, y_train, X_test, X_val, y_val.

**Key responsibilities:**
- Load training features, labels, and test features from CSV
- Validate data integrity (shape, types, respondent_id consistency)
- Create stratified train/validation splits preserving both vaccine label distributions
- Prevent data leakage by tracking train/val/test indices
- Output aligned feature and target matrices

### 2. Imputation Strategy Registry
Pluggable imputation methods (drop rows, mean/median, forward-fill, KNN, iterative MICE, flag-as-feature, learned models). Each returns preprocessed features + metadata (columns dropped, imputation rates). Easy to add new strategies without modifying core pipeline.

**Supported strategies:**
- **Drop rows**: Remove any row with missing values (baseline, data loss)
- **Drop columns**: Remove high-missing features entirely
- **Mean/Median imputation**: Fast, preserves sample count
- **Mode imputation**: For categorical features
- **KNN imputation**: k-nearest neighbors (structure-preserving, slower)
- **Iterative MICE**: Multiple imputation by chained equations (handles multivariate patterns)
- **Forward/Backward fill**: For time-like data (if applicable)
- **Flag-as-feature**: Create binary "is_missing" flags, impute with 0 or default
- **Target-based imputation**: Impute based on target variable relationship
- **Custom learners**: Train separate models to predict missing values

**Output:** Preprocessed features, imputation metadata (for diagnostics)

### 3. Feature Encoding Module
Applies configurable transforms: ordinal encoding for opinion/concern/knowledge; one-hot or ordinal for categorical demographics/geography; identity for binary features; interaction/polynomial terms on demand. Outputs consistent feature matrix shape across train/val/test.

**Encoding strategies per feature group:**
- **Opinion features** (6): `opinion_h1n1_*`, `opinion_seas_*` – ordinal 1–5, preserve order
- **Concern & Knowledge** (2): `h1n1_concern`, `h1n1_knowledge` – ordinal 1–4, preserve order
- **Behavioral** (7): `behavioral_*` – binary 0/1, identity encoding
- **Doctor recommendations** (2): `doctor_recc_*` – binary 0/1, identity encoding
- **Health status** (4): `chronic_med_condition`, `child_under_6_months`, `health_worker`, `health_insurance` – binary 0/1, identity encoding
- **Demographics** (8): `age_group`, `education`, `race`, `sex`, `income_poverty`, `marital_status`, `rent_or_own`, `employment_status` – one-hot or target encoding
- **Household** (2): `household_adults`, `household_children` – integer identity
- **Geographic** (2): `hhs_geo_region` (10 regions), `census_msa` (4 categories) – one-hot or ordinal
- **Employment** (2): `employment_industry`, `employment_occupation` – one-hot or target encoding (high cardinality, handle missing)

**Advanced transforms (optional):**
- Polynomial features (degree 2–3) on ordinal/numeric features
- Interaction terms (e.g., opinion × demographic)
- Domain-specific features (e.g., doctor_recommendation × concern)
- Log transforms for skewed distributions

**Output:** Transformed feature matrix, feature names/indices for reference

### 4. Model Factory
Wraps multiple model types—logistic regression, XGBoost, LightGBM, Random Forest, neural networks, stacking ensembles—behind a unified interface. Each model registered with default hyperparameters; easy to override and grid-search.

**Supported models:**
- **Logistic Regression**: Fast baseline, interpretable, natural probability outputs
- **XGBoost**: Gradient boosting, handles non-linearity, feature importance
- **LightGBM**: Fast gradient boosting, memory efficient, handles categorical features natively
- **Random Forest**: Ensemble, robust, feature importance, easy to parallelize
- **Neural Networks** (optional): MLP with PyTorch/TensorFlow for non-linear patterns
- **Stacking/Voting Ensembles**: Combine multiple models with meta-learner

**Key interface:**
- `fit(X_train, y_train, X_val, y_val)` – train and validate
- `predict_proba(X)` – return probabilities [0, 1]
- `get_feature_importance()` – extract importance (if available)
- `get_params()` / `set_params()` – enable hyperparameter tuning

### 5. Training & Validation Engine
Stratified k-fold cross-validation, class imbalance handling (class weights, SMOTE, threshold tuning), per-vaccine training, hyperparameter search (grid/random/Bayesian), stores cross-validation fold predictions for stacking/ensemble building.

**Workflow:**
1. Create stratified k-fold splits (k=5–10, preserving both vaccine labels)
2. For each fold:
   - Train model on fold training set
   - Predict on fold validation set
   - Compute per-vaccine ROC AUC
   - Store fold predictions (for stacking/blending)
3. Average AUROC across folds (with std for uncertainty)
4. Hyperparameter search (grid/random/Bayesian Optuna):
   - Define search space (learning rate, depth, regularization, etc.)
   - Minimize: 1 − mean_AUROC
   - Log all attempts

**Class imbalance handling (per vaccine):**
- **Class weights**: Penalize minority class more during training
- **SMOTE**: Synthetic oversampling of minority class
- **Threshold tuning**: Adjust decision threshold post-hoc to optimize AUROC
- **Ensemble weighting**: Different weights for H1N1 vs seasonal predictions

**Output:** 
- Cross-validation AUROC per vaccine + mean
- Fold predictions (X_val × 2 targets × k folds)
- Best hyperparameters
- Training history/logs

### 6. Calibration Module
Post-hoc calibration methods (Platt scaling, isotonic regression, temperature scaling, no calibration) applied to validation predictions, then evaluated. Stores calibration models for test inference.

**Supported calibration methods:**
- **None**: Use raw model probabilities (baseline)
- **Platt scaling**: Logistic regression on validation predictions → test calibration
- **Isotonic regression**: Non-parametric monotonic calibration
- **Temperature scaling**: Single scalar multiplier on logit (fast, effective for neural networks)
- **Beta calibration**: Extension of Platt for extreme probabilities

**Evaluation:**
- Calibration plots (predicted probability vs. true frequency)
- Expected Calibration Error (ECE)
- Brier score (mean squared error of probabilities)

**Output:** Calibrated probabilities + calibration models (pickle/joblib)

### 7. Evaluation & Diagnostics
Computes per-vaccine ROC AUC on validation/test sets, generates ROC curves, confusion matrices, calibration plots, feature importance, prediction confidence histograms. Identifies failure cases for error analysis.

**Metrics:**
- Per-vaccine ROC AUC (main objective)
- Mean AUROC across vaccines
- Precision, Recall, F1 (at various thresholds)
- Brier score, log loss
- Calibration error, sharpness

**Visualizations:**
- ROC curves per vaccine
- Confusion matrices (at default/optimal thresholds)
- Feature importance plots
- Calibration curves
- Prediction confidence histograms
- AUROC vs. hyperparameter trends

**Diagnostics:**
- Per-vaccine vs. overall performance
- High-confidence misclassifications
- Feature correlation with targets
- Error clustering (by demographic, opinion, etc.)

### 8. Experiment Tracker
Logs run metadata (timestamp, imputation method, feature config, model type, hyperparameters), validation metrics (fold-wise AUROC, mean AUROC, std), cross-validation predictions (for ensembling), and final test submission. Enables comparison across runs and trend analysis.

**Tracked information:**
- **Run metadata**: Timestamp, run_id, description, version control commit hash
- **Configuration**: 
  - Imputation strategy + parameters
  - Feature encoding choices + drop list
  - Model type + hyperparameters
  - CV fold count, random seed
- **Results**:
  - Per-fold AUROC (h1n1, seasonal, mean)
  - Final validation AUROC + standard deviation
  - Training time, prediction time
  - Feature importance (top-20)
- **Artifacts**:
  - Trained model checkpoints
  - Calibration models
  - Cross-validation predictions (for stacking)
  - Configuration JSON/YAML

**Storage:**
- **CSV log**: One row per run, columns = configuration + metrics
- **Database** (optional SQLite): Structured querying and filtering
- **MLflow** (optional): Experiment management, parameter/metric logging, model registry
- **Weights & Biases** (optional): Live dashboard, visualization, collaboration

**Comparison interface:**
- Rank runs by AUROC (ascending/descending)
- Filter by model type, imputation method, or metric range
- Export best runs for publication/submission

### 9. Prediction & Submission
Applies trained model + calibration to test set, ensures probabilities in [0.0, 1.0], formats output as CSV, validates against submission template.

**Workflow:**
1. Load test features, apply same preprocessing (imputation, encoding) as training
2. Inference: Run through trained model + calibration
3. Probability clipping: Ensure [0.0, 1.0]
4. Format CSV: `respondent_id, h1n1_vaccine, seasonal_vaccine`
5. Validate against `submission_format.csv` (shape, column order, respondent_id uniqueness)
6. Output: `predictions_<run_id>.csv`

**Safeguards:**
- Verify test features after preprocessing (shape, NaN count)
- Verify prediction ranges [0, 1]
- Compare submission shape to template
- Checksum/hash for reproducibility

## Further Considerations

### 1. Imputation Strategy Trade-off
Drop rows risks losing data; mean/median is fast but naive; KNN/MICE preserves structure but slower. Design allows testing all simultaneously—log which achieves highest AUROC.

**Decision guidance:**
- **High missingness (employment, income, doctor recommendations)**: Test drop-column vs. flag-as-feature first (cheaper)
- **Moderate missingness (health insurance)**: KNN or MICE likely best balance
- **Low missingness (behavioral, opinions)**: Imputation method unlikely to matter—use simple approach

### 2. Feature Cardinality & Encoding
One-hot encoding on 10+ geographic/demographic features balloons dimensionality; ordinal or target encoding may be better. Pipeline should allow swapping encoding methods per feature category.

**Decision guidance:**
- **Ordinal features** (opinion, concern): Always preserve order
- **Low-cardinality categorical** (sex, marital_status, census_msa ≤ 4): One-hot safe
- **High-cardinality categorical** (employment_industry, occupation): Target encoding or ordinal by frequency
- **Baseline test**: Compare one-hot vs. target encoding on validation AUROC

### 3. Class Imbalance Approach
H1N1 (21% positive) vs. seasonal (47% positive) may need different strategies. Design supports per-vaccine configuration: SMOTE only for H1N1, threshold tuning on both, different class weights.

**Decision guidance:**
- **H1N1 (severe imbalance)**: Test class weights + threshold tuning first; try SMOTE if needed
- **Seasonal (mild imbalance)**: Class weights probably sufficient
- **Baseline thresholds**: 0.5 for both; optimize per-vaccine independently post-hoc

### 4. Model Selection Uncertainty
Logistic regression vs. XGBoost vs. ensemble unknowns—experiment tracker will reveal winner. Design avoids premature optimization; any model can be tried with minimal code changes.

**Research strategy:**
- **Phase 1**: Logistic regression baseline (fast, interpretable)
- **Phase 2**: XGBoost/LightGBM with default hyperparameters
- **Phase 3**: Hyperparameter tuning (grid search top-2 models)
- **Phase 4**: Ensembles (stacking, voting) combining best models

### 5. Hyperparameter Search Scope
XGBoost/LightGBM have 20+ hyperparameters; Bayesian optimization more efficient than grid search for exploration. Experiment tracker logs all attempts to avoid redundant searches.

**Search strategy:**
- **Bayesian optimization** (Optuna, Hyperopt) for continuous search
- **Grid search** on top-3 most important hyperparameters first
- **Random search** for initial exploration (10–20 trials)
- **Log all trials** to avoid redundant experiments

### 6. Stacking & Blending Potential
Store cross-validation predictions from multiple models to train a meta-learner. Pipeline must preserve fold structure for valid stacking.

**Stacking workflow:**
1. Train 3–5 diverse base models (logistic, XGBoost, LightGBM, Random Forest)
2. Generate OOF (out-of-fold) predictions on training set
3. Train meta-learner (logistic regression or simple model) on OOF predictions
4. Apply meta-learner to test predictions from base models

**Prerequisite:** Fold structure preserved in experiment tracker outputs

## Data Overview & Feature Mapping

### Dataset Shapes
- **Training**: 26,707 samples, 35 features + 1 target (split into 2 binary targets)
- **Test**: 26,708 samples, 35 features
- **Total respondents**: 53,415

### Class Imbalance
| Metric | H1N1 | Seasonal |
|--------|------|----------|
| Positive rate | 21.2% | 46.5% |
| Imbalance ratio | 3.7:1 | 1.15:1 |
| Both vaccines | 17.6% | – |
| Neither vaccine | 49.8% | – |

### Feature Groups & Cardinality
| Group | Count | Missing % | Notes |
|-------|-------|-----------|-------|
| Opinion | 6 | < 1% | Ordinal 1–5 |
| Behavioral | 7 | < 1% | Binary |
| Doctor recommendations | 2 | 7–8% | Binary, strong predictors |
| Health status | 4 | 3–46% | Binary/mixed, health_insurance high-missing |
| Concern & Knowledge | 2 | < 1% | Ordinal 1–4 |
| Demographics | 8 | 0–19% | Mixed cardinality, categorical |
| Household | 2 | < 1% | Integer, top-coded to 3 |
| Geographic | 2 | < 1% | 10 regions, 4 MSA categories |
| Employment | 2 | 48–50% | High-cardinality, systematic missingness |

### Missing Data Patterns
- **Employment fields** (~50% missing): Likely non-employed respondents
- **Health insurance** (46% missing): Non-coverage or non-response
- **Income** (19% missing): Non-response bias
- **Doctor recommendations** (7–8% missing): Non-response or not applicable
- **Most other fields** (< 2% missing): Sporadic missing data

## Implementation Roadmap (High-Level)

**Phase 1: Foundation**
- Implement data loading, stratified splits, basic imputation options
- Train logistic regression baseline
- Set up experiment tracker (CSV log)

**Phase 2: Feature Engineering**
- Implement encoding strategies per feature group
- Test encoding variations (one-hot vs. ordinal vs. target encoding)
- Log which encoding improves validation AUROC

**Phase 3: Model Exploration**
- Add XGBoost, LightGBM, Random Forest models
- Compare baseline logistic regression to tree-based models
- Experiment with hyperparameter tuning (grid search, Bayesian optimization)

**Phase 4: Advanced Techniques**
- Implement SMOTE, threshold tuning, class weights
- Add calibration module (Platt, isotonic)
- Stacking/ensemble experiments

**Phase 5: Optimization & Submission**
- Identify AUROC-optimal configuration
- Generate final test predictions
- Validate submission format, submit

## Expected Outputs

1. **Experiment log** (CSV or database): Every run with configuration + AUROC results
2. **Best model artifacts** (pickle/joblib): Trained model, calibration model
3. **Diagnostic plots**: ROC curves, feature importance, calibration curves
4. **Final submission**: CSV with respondent_id, h1n1_vaccine (probability), seasonal_vaccine (probability)
5. **Report**: Summary of best configuration, AUROC, key insights

---

**This design prioritizes exploration and modularity.** Each component can be tested independently, and the experiment tracker will systematically reveal which imputation, encoding, model, and calibration choices maximize AUROC.
