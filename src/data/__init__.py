"""
Data loading and validation module.

This module provides interfaces and implementations for loading training and test data,
creating cross-validation splits, and validating data integrity.

Classes:
    DataLoader: Abstract base class for data loading implementations
    CSVDataLoader: Concrete implementation for CSV files
    DataSplit: Container for train/val/test split data
    DataValidationResult: Results of data validation checks

See docs/SYSTEM_DESIGN.md for detailed component architecture.
"""

from src.data.loader import (
    DataLoader,
    CSVDataLoader,
    DataSplit,
    DataValidationResult,
)

__all__ = [
    "DataLoader",
    "CSVDataLoader",
    "DataSplit",
    "DataValidationResult",
]
