"""
ML Pipeline for Flu Shot Prediction

This package contains the modular, component-based ML pipeline for predicting H1N1 and seasonal
flu vaccine uptake based on survey data from the National 2009 H1N1 Flu Survey.

Architecture:
- data: Data loading and validation
- preprocessing: Imputation and feature encoding strategies
- models: Model implementations and factory
- training: Training orchestration and hyperparameter search
- calibration: Prediction calibration methods
- evaluation: Metrics computation and visualization
- tracking: Experiment logging and results management
- prediction: Submission generation and validation
- utils: Logging, validation, metrics, and plotting helpers
- config: Configuration system (dataclasses and loaders)

See docs/SYSTEM_DESIGN.md for detailed architecture documentation.
"""

__version__ = "0.1.0"
__author__ = "ML Pipeline Team"

# Lazy imports to avoid circular dependencies
# Components are imported at use time in main.py
