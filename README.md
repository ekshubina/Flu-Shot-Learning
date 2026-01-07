# Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines

A machine learning project to predict whether people received H1N1 and seasonal flu vaccines based on their demographics, opinions, and health behaviors.

## Project Overview

This is a practice competition from [DrivenData](https://www.drivendata.org/competitions/66/flu-shot-learning/) designed to be accessible to participants at all levels. The goal is to build predictive models that can accurately forecast vaccination patterns.

### Problem Statement

Using survey data from the National 2009 H1N1 Flu Survey conducted by the U.S. Department of Health and Human Services, predict whether respondents received:
- H1N1 flu vaccine
- Seasonal flu vaccine

The dataset includes information about:
- Social and economic background
- Opinions on health risks and vaccine effectiveness  
- Health behaviors and practices
- Demographics

## Competition Timeline

- **Status**: Ongoing
- **End Date**: July 30, 2026, 11:59 p.m. UTC
- **Joined**: 8,563+ participants

## Getting Started

### Prerequisites

- Python 3.7+
- pip or conda

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/Flu-Shot-Learning.git
   cd Flu-Shot-Learning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data

Download the data from the [competition page](https://www.drivendata.org/competitions/66/flu-shot-learning/data/).

## Project Structure

This project follows a modular, component-based architecture with clear separation of concerns. Each component is independent and can be swapped with alternative implementations.

```
Flu-Shot-Learning/
├── main.py                          # Pipeline orchestrator (entry point)
├── requirements.txt                 # Core dependencies (numpy, scikit-learn, etc.)
├── requirements-dev.txt             # Development dependencies (pytest, black, etc.)
├── README.md                        # This file
├── LICENSE                          # MIT License
├── .gitignore                       # Git ignore rules
│
├── src/                             # Main package
│   ├── __init__.py
│   ├── config.py                    # Configuration system (dataclasses)
│   │
│   ├── data/                        # Component 1: Data Loading
│   │   ├── __init__.py
│   │   └── loader.py                # DataLoader ABC + CSVDataLoader implementation
│   │
│   ├── preprocessing/               # Component 2: Feature Preprocessing
│   │   ├── __init__.py
│   │   ├── imputation.py            # 7 imputation strategies (mean, KNN, MICE, etc.)
│   │   └── encoding.py              # 5 encoding strategies (ordinal, onehot, target, etc.)
│   │
│   ├── models/                      # Component 3: ML Models
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseModel ABC
│   │   ├── factory.py               # ModelFactory (registry pattern)
│   │   ├── logistic_regression.py   # Logistic regression wrapper
│   │   ├── random_forest.py         # Random forest wrapper
│   │   ├── xgboost_model.py         # XGBoost wrapper
│   │   └── lightgbm_model.py        # LightGBM wrapper
│   │
│   ├── training/                    # Component 4: Model Training
│   │   ├── __init__.py
│   │   └── engine.py                # TrainingEngine (CV, hyperparameter search)
│   │
│   ├── calibration/                 # Component 5: Probability Calibration
│   │   ├── __init__.py
│   │   └── calibrator.py            # CalibratorInterface ABC + 4 implementations
│   │
│   ├── evaluation/                  # Component 6: Model Evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py               # Evaluator class (AUROC, confusion matrix, etc.)
│   │   └── plots.py                 # Visualization functions (ROC, calibration, etc.)
│   │
│   ├── tracking/                    # Component 7: Experiment Tracking
│   │   ├── __init__.py
│   │   └── logger.py                # ExperimentTracker ABC + CSVExperimentLogger
│   │
│   ├── prediction/                  # Component 8: Test Set Prediction
│   │   ├── __init__.py
│   │   └── predictor.py             # PredictionEngine (submission formatting)
│   │
│   └── utils/                       # Component 9: Utilities
│       ├── __init__.py
│       ├── logging.py               # Centralized logging setup
│       ├── validation.py            # Data validation functions
│       ├── metrics.py               # Helper metric functions
│       ├── plots.py                 # Plotting utility wrappers
│       └── helpers.py               # General utilities (splits, weights, features)
│
├── data/                            # Competition data (not in repo)
│   ├── training_set_features.csv    # Training features (respondent_id + 35 features)
│   ├── training_set_labels.csv      # Training labels (respondent_id, h1n1, seasonal)
│   ├── test_set_features.csv        # Test features (respondent_id + 35 features)
│   └── submission_format.csv        # Template for submission format
│
├── examples/                        # Configuration examples
│   ├── config_baseline.yaml         # Logistic regression baseline
│   └── config_xgboost.yaml          # Advanced XGBoost with hyperparameter tuning
│
├── docs/                            # Documentation
│   ├── PROBLEM_DESCRIPTION.md       # Feature definitions and evaluation metric
│   ├── SYSTEM_DESIGN.md             # Complete system architecture
│   ├── CONTEXT_REPORT.md            # Dataset analysis and insights
│   ├── architecture.md              # Module relationships and data flow
│   ├── SCAFFOLDING.md               # Guide for extending/modifying components
│   │
│   └── build-scaffolds/             # Implementation documentation
│       ├── build-scaffolds-plan.md  # Implementation plan
│       ├── build-scaffolds-context.md
│       └── build-scaffolds-tasks.md # Task checklist
│
├── logs/                            # Training logs (created at runtime)
│   └── pipeline.log
│
└── submissions/                     # Output predictions (created at runtime)
    └── submission.csv
```

### Component Overview

| Component | Purpose | Key Classes |
|-----------|---------|------------|
| **Data** | Load and validate training/test data | `CSVDataLoader` |
| **Preprocessing** | Handle missing values and encode features | `ImputationStrategy`, `FeatureEncoder` |
| **Models** | Define ML model interfaces | `BaseModel`, `ModelFactory` |
| **Training** | Run cross-validation and hyperparameter search | `TrainingEngine` |
| **Calibration** | Improve probability estimates | `CalibratorInterface` |
| **Evaluation** | Compute metrics and create visualizations | `Evaluator` |
| **Tracking** | Log experiment results | `ExperimentTracker`, `CSVExperimentLogger` |
| **Prediction** | Generate test set predictions | `PredictionEngine` |
| **Utils** | Helper functions for logging, validation, etc. | Various utility functions |

### Data Flow

```
Input Data (CSV)
    ↓
[Data Loading]
    ↓
X_train (n, 35) + y_train (n, 2) + X_test (n, 35)
    ↓
[Imputation] - Handle missing values
    ↓
X_train_imputed + X_test_imputed
    ↓
[Feature Encoding] - Ordinal/OneHot/Target encoding
    ↓
X_train_encoded + X_test_encoded
    ↓
[Model Training] - Cross-validation with hyperparameter search
    ↓
best_model + cv_predictions
    ↓
[Calibration] - Improve probability estimates
    ↓
calibrated_predictions
    ↓
[Evaluation] - Compute metrics (AUROC, calibration error, etc.)
    ↓
metrics + plots
    ↓
[Experiment Tracking] - Log results
    ↓
[Test Prediction] - Apply pipeline to test data
    ↓
test_predictions (n_test, 2)
    ↓
[Submission Formatting] - Format as respondent_id, h1n1_vaccine, seasonal_vaccine
    ↓
submission.csv (competition format)
```

### Running the Pipeline

```bash
# Basic usage (default configuration)
python main.py

# With specific configuration
python main.py --config examples/config_baseline.yaml

# Advanced with hyperparameter tuning
python main.py --config examples/config_xgboost.yaml --run-name exp_001 --verbose

# With custom random seed for reproducibility
python main.py --seed 123
```

See [architecture.md](docs/architecture.md) for detailed component relationships and [SCAFFOLDING.md](docs/SCAFFOLDING.md) for how to extend the pipeline.

## Resources

- [Problem Description](https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/)
- [Benchmark Blog Post](https://www.drivendata.co/blog/predict-flu-vaccine-data-benchmark/)
- [Competition Rules](https://www.drivendata.org/competitions/66/flu-shot-learning/rules/)
- [Leaderboard](https://www.drivendata.org/competitions/66/flu-shot-learning/leaderboard/)

## Data Source

Data is provided courtesy of the United States National Center for Health Statistics (NCHS) at the Centers for Disease Control and Prevention (CDC).

**Citation:**
> U.S. Department of Health and Human Services (DHHS). National Center for Health Statistics. The National 2009 H1N1 Flu Survey. Hyattsville, MD: Centers for Disease Control and Prevention, 2012.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Note: This project is created for educational and learning purposes as part of the DrivenData competition.

## Contributing

Contributions and improvements are welcome! Feel free to:
- Open issues for bugs or suggestions
- Submit pull requests with improvements
- Share your approach and insights

## Contact

For questions about the competition, visit [DrivenData](https://www.drivendata.org/contact/).
