"""
Logging utilities for the ML pipeline.

Provides centralized logging configuration and logger factory functions.
All pipeline components can use get_logger() to access module-specific loggers
with consistent formatting.

Logging setup:
- Console: INFO level with formatted output
- File: DEBUG level with detailed tracebacks (if configured)
- Format: [%(asctime)s] %(name)s - %(levelname)s - %(message)s

Example:
    ```python
    import logging
    from src.utils.logging import setup_logging, get_logger
    
    # Set up logging once at program start
    setup_logging(level='DEBUG', log_file='pipeline.log')
    
    # Get module logger
    logger = get_logger(__name__)
    logger.info("Starting data loading...")
    logger.debug("Detailed information for debugging")
    ```

Reference: SYSTEM_DESIGN.md - Component 9: Utilities
"""

import logging
from typing import Optional
from pathlib import Path


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for the entire pipeline.
    
    Sets up the root logger with console handler (and optional file handler).
    Should be called once at program startup before any logging occurs.
    
    Parameters:
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Default: 'INFO'
        log_file (Optional[str]): Path to log file. If None, file logging disabled.
            Default: None
        format_string (Optional[str]): Custom log message format.
            Default: '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            
    Example:
        ```python
        # Console logging only, INFO level
        setup_logging()
        
        # Console + file logging, DEBUG level
        setup_logging(level='DEBUG', log_file='logs/pipeline.log')
        
        # Custom format
        setup_logging(
            level='INFO',
            format_string='%(levelname)s | %(name)s | %(message)s'
        )
        ```
    
    Implementation notes:
        - TODO: Get root logger
        - TODO: Set logging level (convert string to logging.LEVEL)
        - TODO: Create console handler with INFO level
        - TODO: If log_file provided:
              - Create file handler with DEBUG level
              - Create parent directories if needed
        - TODO: Create formatter with format_string
        - TODO: Add formatter to handlers
        - TODO: Add handlers to root logger
        - TODO: Clear any existing handlers first
        - TODO: Set propagate=True for all loggers
    """
    # TODO: Implement
    pass


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module/component name.
    
    Returns a named logger configured to use the root logger's handlers
    and formatting. All loggers created through this function share the
    same configuration.
    
    Parameters:
        name (str): Logger name, typically __name__ from calling module.
            Example: 'src.models.xgboost' or 'src.preprocessing.encoding'
    
    Returns:
        logging.Logger: Configured logger for the module
        
    Example:
        ```python
        # In src/models/xgboost.py
        logger = get_logger(__name__)  # Creates logger named 'src.models.xgboost'
        
        logger.info("Training XGBoost model...")
        logger.debug(f"Parameters: {params}")
        logger.warning("Class imbalance detected!")
        logger.error("Model training failed!")
        ```
    
    Implementation notes:
        - TODO: Call logging.getLogger(name)
        - TODO: Return the logger object
        - TODO: Do NOT set level on individual logger (inherit from root)
    """
    # TODO: Implement
    raise NotImplementedError("get_logger() not yet implemented")


def configure_logger_file_handler(
    logger: logging.Logger,
    log_file: str,
    level: str = 'DEBUG',
) -> None:
    """
    Add a file handler to an existing logger.
    
    Useful for adding component-specific log files without modifying
    global logging configuration.
    
    Parameters:
        logger (logging.Logger): Logger to add file handler to
        log_file (str): Path to log file
        level (str): Logging level for file handler. Default: 'DEBUG'
        
    Example:
        ```python
        logger = get_logger('src.models.training')
        configure_logger_file_handler(logger, 'logs/training.log', 'DEBUG')
        ```
    
    Implementation notes:
        - TODO: Create file handler
        - TODO: Set level
        - TODO: Create formatter
        - TODO: Add formatter to handler
        - TODO: Add handler to logger
    """
    # TODO: Implement
    pass


class LoggingContext:
    """
    Context manager for temporary logging level changes.
    
    Useful for temporarily adjusting logging verbosity within a code block,
    then automatically restoring previous level on exit.
    
    Example:
        ```python
        logger = get_logger(__name__)
        
        with LoggingContext(logger, 'DEBUG'):
            logger.debug("Detailed debugging info")
            # ... code runs with DEBUG level
        # Logging level automatically restored
        ```
    
    Implementation notes:
        - TODO: Store original level in __enter__
        - TODO: Set new level
        - TODO: Restore original level in __exit__
    """
    
    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize context manager.
        
        Parameters:
            logger: Logger to modify
            level: Temporary logging level
        """
        self.logger = logger
        self.level = level
        self.original_level = None
    
    def __enter__(self) -> 'LoggingContext':
        """Enter context: save original level and set new level."""
        # TODO: Implement
        raise NotImplementedError("LoggingContext.__enter__() not yet implemented")
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: restore original level."""
        # TODO: Implement
        pass
