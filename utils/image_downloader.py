# utils/image_downloader.py
import os
import time
import requests
from typing import Optional, List, Dict, Any
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException # Import the specific exception

# Project local imports
from utils.helpers import slugify, download_image

# Constants for image fetching (can be overridden by config in calling modules)
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"
TMDB_PROFILE_IMAGE_SIZE = "w500"


def search_and_extract_image_urls_ddg(search_term: str, num_images_to_fetch: int, logger: Optional[Any] = None) -> List[str]:
    """
    Searches DuckDuckGo for images and extracts URLs.
    Handles RatelimitException.
    """
    image_urls = []
    if num_images_to_fetch <= 0:
        return []
    try:
        with DDGS() as ddgs:
            max_ddg_results = max(1, num_images_to_fetch + 5)
            results_iterator = ddgs.images(keywords=search_term, region='wt-wt', safesearch='moderate', max_results=max_ddg_results)
            if results_iterator:
                for i, r in enumerate(results_iterator):
                    if len(image_urls) >= num_images_to_fetch:
                        break
                    if r and 'image' in r:
                        image_urls.append(r['image'])
                    if i > num_images_to_fetch + 10:
                        if logger: logger.debug(f"    DDG search for '{search_term}' hit iteration limit ({i+1}) without enough results.")
                        break
            if not image_urls and logger:
                logger.debug(f"    DDG search for '{search_term}' yielded no image URLs after iterating results.")

    except RatelimitException as rle:
        if logger: logger.warning(f"    DDG RatelimitException for '{search_term}': {rle}. Try increasing delays in config.")
    except Exception as e:
        if logger: logger.warning(f"    An error occurred while searching DDG for '{search_term}': {repr(e)}")
    return image_urls[:num_images_to_fetch]


def download_actor_image_tmdb(
    tmdb_api_key: str,
    person_id: int,
    person_name_for_log: str,
    save_path: str,
    base_image_url: str = TMDB_IMAGE_BASE_URL,
    image_size: str = TMDB_PROFILE_IMAGE_SIZE,
    logger: Optional[Any] = None
) -> Optional[str]:
    """
    Fetches an actor's profile image from TMDB and saves it.
    Returns the local filename if successful, None otherwise.
    """
    if not tmdb_api_key or not person_id:
        if logger: logger.debug(f"  Skipping actor image download: missing TMDB key or person ID for '{person_name_for_log}'.")
        return None

    url = f"https://api.themoviedb.org/3/person/{person_id}/images"
    headers = {"accept": "application/json", "Authorization": f"Bearer {tmdb_api_key}"}

    try:
        if logger: logger.debug(f"  Querying TMDB images for person ID {person_id} ('{person_name_for_log}')...")
        response = requests.get(url, headers=headers, timeout=7)
        response.raise_for_status()
        data = response.json()

        if data.get("profiles") and len(data["profiles"]) > 0:
            file_path_suffix = data["profiles"][0].get("file_path")
            if file_path_suffix:
                image_url = f"{base_image_url}{image_size}{file_path_suffix}"
                _, file_extension = os.path.splitext(file_path_suffix)
                if not file_extension: file_extension = ".jpg"

                local_image_filename = f"{person_id}{file_extension}"
                local_image_full_path = os.path.join(save_path, local_image_filename)

                if os.path.exists(local_image_full_path):
                    if logger: logger.debug(f"    Actor image already exists: {local_image_full_path}. Skipping download.")
                    return local_image_filename

                if download_image(image_url, local_image_full_path, logger):
                    if logger: logger.debug(f"    Successfully saved TMDB actor image: {local_image_full_path}")
                    return local_image_filename
        if logger: logger.info(f"  No TMDB profile image found for person ID {person_id} ('{person_name_for_log}').")
        return None
    except Exception as e:
        if logger: logger.warning(f"  Error during TMDB actor image fetch for person ID {person_id} ('{person_name_for_log}'): {e}")
        return None


def download_character_image_ddg(
    character_name: str,
    movie_title: str,
    tmdb_person_id: Optional[int],
    num_images_to_fetch: int,
    save_path: str,
    sleep_between_downloads: float, # New parameter
    logger: Optional[Any] = None
) -> List[str]:
    """
    Searches DuckDuckGo for character images and saves them.
    Returns a list of local filenames downloaded.
    """
    downloaded_filenames: List[str] = []

    if not character_name:
        if logger: logger.info(f"  Skipping DDG character image download: No character name provided for movie '{movie_title}'.")
        return []

    if num_images_to_fetch <= 0:
        if logger: logger.debug(f"  Skipping DDG character image download for '{character_name}': num_images_to_fetch is {num_images_to_fetch}.")
        return []

    search_term = f"{character_name} {movie_title} character"

    if tmdb_person_id:
        filename_prefix = f"{tmdb_person_id}_char_{slugify(character_name)}"
    else:
        filename_prefix = f"unknown-id_char_{slugify(character_name)}"
        if logger: logger.warning(f"  No TMDB person ID for character '{character_name}'. Using generic ID for filename.")

    if logger: logger.debug(f"  Performing DDG image search for character '{character_name}' ({search_term})...")
    image_urls = search_and_extract_image_urls_ddg(search_term, num_images_to_fetch, logger)

    if not image_urls:
        if logger: logger.info(f"  No image URLs found on DDG for character '{character_name}'.")
        return []

    if logger: logger.debug(f"  Attempting to download {len(image_urls)} DDG images for '{character_name}'...")
    for i, img_url in enumerate(image_urls):
        file_extension = ".jpg"
        try:
            path_part = img_url.split('?')[0].lower()
            if path_part.endswith(".png"): file_extension = ".png"
            elif path_part.endswith(".gif"): file_extension = ".gif"
            elif path_part.endswith(".webp"): file_extension = ".webp"
            elif path_part.endswith(".jpeg"): file_extension = ".jpeg"
            elif path_part.endswith(".jpg"): file_extension = ".jpg"
            if not any(ext in file_extension for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                if logger: logger.debug(f"    Uncommon extension detected in '{img_url}', defaulting to .jpg.")
                file_extension = ".jpg"
        except Exception as e_ext:
            if logger: logger.warning(f"    Error inferring extension for {img_url}: {e_ext}. Defaulting to .jpg.")

        local_image_filename = f"{filename_prefix}_{i+1}{file_extension}"
        local_image_full_path = os.path.join(save_path, local_image_filename)

        if os.path.exists(local_image_full_path):
            if logger: logger.debug(f"    Character image already exists: {local_image_full_path}. Skipping download.")
            downloaded_filenames.append(local_image_filename)
            continue

        if download_image(img_url, local_image_full_path, logger):
            if logger: logger.debug(f"    Downloaded DDG character image: {local_image_full_path}")
            downloaded_filenames.append(local_image_filename)
        else:
            if logger: logger.warning(f"    Failed to download DDG image {img_url} for '{character_name}'.")

        if i < len(image_urls) - 1: # Don't sleep after the last download in this group
            time.sleep(sleep_between_downloads)

    if logger: logger.info(f"  Finished DDG downloads for '{character_name}'. Downloaded {len(downloaded_filenames)} of {len(image_urls)} found URLs.")
    return downloaded_filenames


def download_ddg_image_for_query(
    query: str,
    filename_prefix_base: str,
    num_images_to_fetch: int,
    save_path: str,
    sleep_between_downloads: float, # New parameter
    logger: Optional[Any] = None
) -> List[str]:
    """
    Searches DuckDuckGo for images based on a generic query and saves them.
    Returns a list of local filenames downloaded.
    """
    downloaded_filenames: List[str] = []

    if not query:
        if logger: logger.info(f"  Skipping DDG image download: No query provided.")
        return []

    if num_images_to_fetch <= 0:
        if logger: logger.debug(f"  Skipping DDG image download for query '{query}': num_images_to_fetch is {num_images_to_fetch}.")
        return []

    if logger: logger.debug(f"  Performing DDG image search for query '{query}'...")
    image_urls = search_and_extract_image_urls_ddg(query, num_images_to_fetch, logger)

    if not image_urls:
        if logger: logger.info(f"  No image URLs found on DDG for query '{query}'.")
        return []

    if logger: logger.debug(f"  Attempting to download {len(image_urls)} DDG images for query '{query}'...")
    for i, img_url in enumerate(image_urls):
        file_extension = ".jpg"
        try:
            path_part = img_url.split('?')[0].lower()
            if path_part.endswith(".png"): file_extension = ".png"
            elif path_part.endswith(".gif"): file_extension = ".gif"
            elif path_part.endswith(".webp"): file_extension = ".webp"
            elif path_part.endswith(".jpeg"): file_extension = ".jpeg"
            elif path_part.endswith(".jpg"): file_extension = ".jpg"
            if not any(ext in file_extension for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                if logger: logger.debug(f"    Uncommon extension detected in '{img_url}', defaulting to .jpg for query '{query}'.")
                file_extension = ".jpg"
        except Exception as e_ext:
            if logger: logger.warning(f"    Error inferring extension for {img_url} (query: '{query}'): {e_ext}. Defaulting to .jpg.")

        local_image_filename = f"{filename_prefix_base}_{i+1}{file_extension}"
        local_image_full_path = os.path.join(save_path, local_image_filename)

        if os.path.exists(local_image_full_path):
            if logger: logger.debug(f"    Image for query '{query}' already exists: {local_image_full_path}. Skipping download.")
            downloaded_filenames.append(local_image_filename)
            continue

        if download_image(img_url, local_image_full_path, logger):
            if logger: logger.debug(f"    Downloaded DDG image for query '{query}': {local_image_full_path}")
            downloaded_filenames.append(local_image_filename)
        else:
            if logger: logger.warning(f"    Failed to download DDG image {img_url} for query '{query}'.")

        if i < len(image_urls) - 1: # Don't sleep after the last download in this group
            time.sleep(sleep_between_downloads)

    if logger: logger.info(f"  Finished DDG downloads for query '{query}'. Downloaded {len(downloaded_filenames)} of {len(image_urls)} found URLs.")
    return downloaded_filenames