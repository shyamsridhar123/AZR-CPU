"""
Logging utilities for the Absolute Zero Reasoner system.
"""

import logging
import sys
import os
from typing import Optional
from pathlib import Path
import datetime


def setup_logger(name: str, 
                level: str = "INFO", 
                log_file: Optional[str] = None,
                console: bool = True) -> logging.Logger:
    """
    Set up a logger with specified configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_run_logger(experiment_name: str, run_id: str) -> logging.Logger:
    """
    Create a logger for a specific experimental run.
    
    Args:
        experiment_name: Name of the experiment
        run_id: Unique identifier for this run
        
    Returns:
        Logger configured for this run
    """
    # Create log directory structure
    log_dir = Path("logs") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_id}_{timestamp}.log"
    
    # Setup logger
    logger = setup_logger(
        name=f"azr.{experiment_name}.{run_id}",
        level="INFO",
        log_file=str(log_file),
        console=True
    )
    
    logger.info(f"Starting experiment: {experiment_name}, run: {run_id}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup basic logging configuration for the AZR system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure basic logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if not log_file else [logging.FileHandler(log_file)])
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
