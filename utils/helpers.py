# movie_enrichment_project/utils/helpers.py
import yaml
import os
import logging
from typing import List, Dict, Any, Optional # Added Optional for logger type hint

def words_to_tokens(words: int, ratio: float = 1.4) -> int:
    """
    Estimates the number of tokens from a given word count.
    """
    return int(words * ratio)

def load_full_movie_data_from_yaml(filename: str) -> List[Dict[str, Any]]:
    """
    Loads a list of movie data (as dictionaries) from a YAML file.
    In the orchestrator, these dicts will be validated into Pydantic models.
    """
    movie_data_list: List[Dict[str, Any]] = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict) and entry.get("movie_title"): # Basic check
                            movie_data_list.append(entry)
            # print(f"Loaded {len(movie_data_list)} existing movie entries from '{filename}'.")
        except yaml.YAMLError as ye:
            # Using print here as logger might not be set up when this is first called by orchestrator
            print(f"YAML parsing error loading movie data from '{filename}': {ye}. Starting empty.")
        except Exception as e:
            print(f"Error loading full movie data from '{filename}': {e}. Starting empty.")
    # else:
        # print(f"Movie database file '{filename}' not found. Starting empty.")
    return movie_data_list

def save_movie_data_to_yaml(movie_data_list: List[Dict[str, Any]], filename: str, logger: Optional[logging.Logger] = None):
    """
    Saves a list of movie data (dictionaries) to a YAML file.
    The orchestrator should pass data after model_dump(exclude_none=True) from Pydantic models.
    """
    try:
        # Ensure parent directory exists
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty
             os.makedirs(output_dir, exist_ok=True)
             if logger: logger.info(f"Created output directory for YAML: {output_dir}")

        with open(filename, 'w', encoding='utf-8') as f_yaml:
            yaml.dump(movie_data_list, f_yaml, sort_keys=False, indent=2, allow_unicode=True, Dumper=yaml.SafeDumper)
        if logger:
            logger.info(f"Successfully wrote {len(movie_data_list)} total entries to '{filename}'.")
        # else:
            # print(f"      Successfully wrote {len(movie_data_list)} total entries to '{filename}'.")
    except Exception as e:
        if logger:
            logger.error(f"Error writing to YAML file '{filename}': {e}")
        else:
            print(f"      Error writing to YAML file '{filename}': {e}")

def setup_logging(
    log_file_path: str,
    logger_name: str = 'MovieEnrichmentApp', # Default name if not provided
    level=logging.INFO
) -> logging.Logger:
    """
    Sets up basic file and console logging.
    Returns the configured logger instance.
    """
    # Ensure parent directory exists for the log file
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir): # Check if log_dir is not empty string (e.g. if log file is in current dir)
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            # Fallback to console print if directory creation fails, as logger might not be usable for this error
            print(f"CRITICAL: Could not create log directory {log_dir}: {e}. Logging to file might fail.")
            # Depending on severity, you might want to raise the error or exit.

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False # Prevents log messages from being passed to the handlers of ancestor loggers.

    # Clear existing handlers for this logger instance to avoid duplicate logs on re-runs in same session (e.g. during testing)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    try:
        fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        fh.setLevel(level)
        # More detailed formatter for file logs
        formatter_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)
    except Exception as e:
        # If file handler setup fails, critical error, print to console
        print(f"CRITICAL: Error setting up file handler for logging at {log_file_path}: {e}")


    # Console Handler (always add this as a fallback and for interactive use)
    ch = logging.StreamHandler()
    ch.setLevel(level) # Console can have a different level if needed, e.g., logging.WARNING for less verbosity
    formatter_ch = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter_ch)
    logger.addHandler(ch)

    # Test message to confirm logger is working (optional)
    # logger.info(f"Logger '{logger_name}' initialized. Logging to file: {log_file_path} and console.")

    return logger