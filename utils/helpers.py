import yaml
import logging
import math
import os
import re
import requests
import shutil # For copyfileobj
from typing import Optional, Any, Dict, List, Set, Union


def setup_logging(log_file_path: str, logger_name: str = "MovieEnrichmentPipeline"):
    """
    Sets up a basic logger that writes to a file and outputs to console.
    Ensures the log directory exists.
    """
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating log directory {log_dir}: {e}")
            # Fallback to current directory if logging directory can't be created
            log_file_path = os.path.join(os.getcwd(), os.path.basename(log_file_path))

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent duplicate messages
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Console generally shows INFO and above
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

def words_to_tokens(num_words: int, ratio: float = 1.3) -> int:
    """Converts an approximate number of words to tokens."""
    return math.ceil(num_words * ratio)

def load_full_movie_data_from_yaml(filepath: str) -> List[Dict[str, Any]]:
    """Loads all movie entries from a YAML file."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, list) else []
    except yaml.YAMLError as e:
        print(f"Error reading YAML file {filepath}: {e}")
        return []
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        return []

def save_movie_data_to_yaml(data: List[Dict[str, Any]], output_file: str):
    """Saves movie data to a YAML file."""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True, indent=2)
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")

def parse_index_range_string(range_str: str, logger: Optional[Any] = None) -> Set[int]:
    """
    Parses a string like "0-4, 7, 10-12" into a set of unique integers.
    Handles single numbers and ranges. Logs warnings for invalid parts.
    """
    if not range_str:
        return set()

    indices = set()
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if '-' in part:
            try:
                start_str, end_str = part.split('-')
                start = int(start_str.strip())
                end = int(end_str.strip())
                indices.update(range(min(start, end), max(start, end) + 1))
            except ValueError:
                if logger: logger.warning(f"Invalid range part in index string: '{part}'. Skipping.")
        else:
            try:
                indices.add(int(part))
            except ValueError:
                if logger: logger.warning(f"Invalid single index in string: '{part}'. Skipping.")
    return indices

def slugify(text: str) -> str:
    """Converts text to a URL-friendly slug."""
    if not text: return "unknown"
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    text = text.strip('-')
    return text if text else "slug_error"

def download_image(url: str, filepath: str, logger: Optional[Any] = None) -> bool:
    """
    Downloads an image from a URL to a specified filepath.
    Returns True on success, False on failure.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, stream=True, timeout=20, headers=headers)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        if logger: logger.debug(f"    Successfully downloaded image to {filepath}")
        return True
    except requests.exceptions.RequestException as e:
        if logger: logger.warning(f"    Error downloading {url} to {filepath}: {e}")
        return False
    except Exception as e:
        if logger: logger.error(f"    An unexpected error occurred while downloading {url} to {filepath}: {e}")
        return False