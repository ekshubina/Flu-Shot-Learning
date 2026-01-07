"""
Utilities module for helper functions and common operations.

This module provides logging, validation, metrics computation, and visualization
helper functions used across the pipeline.

Submodules:
    logging: Logging setup and configuration
    validation: Data validation helpers
    metrics: Metric computation utilities
    plots: Visualization helpers
    helpers: General utility functions

See docs/SYSTEM_DESIGN.md for detailed component architecture.
"""

__all__ = [
    "setup_logging",
    "get_logger",
]
