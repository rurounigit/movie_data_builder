# data_providers/tmdb_api.py
import requests
import os
import shutil
from typing import Optional, List, Dict, Any, Tuple
from models.movie_models import TMDBRawCharacter, TMDBReviewsResponse, TMDBReviewResult, TMDBReviewAuthorDetails # Keeping for Pydantic validation if used elsewhere

# Default base URL and size, can be overridden by config
TMDB_IMAGE_BASE_URL_DEFAULT = "https://image.tmdb.org/t/p/"
TMDB_IMAGE_SIZE_DEFAULT = "w500"


def fetch_top_rated_movies_from_tmdb(
    tmdb_api_key: str,
    page: int = 1,
    logger: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    if not tmdb_api_key:
        log_message = "TMDB_API_KEY not set. Cannot fetch top rated movies."
        if logger: logger.error(log_message)
        else: print(f"Error: {log_message}")
        return None
    url = f"https://api.themoviedb.org/3/movie/top_rated?language=en-US&page={page}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {tmdb_api_key}"}
    try:
        if logger: logger.debug(f"Querying TMDB Top Rated movies (Page {page})...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and "results" in data:
            if logger: logger.debug(f"TMDB Top Rated (Page {page}): Found {len(data['results'])} movies. Total pages: {data.get('total_pages')}")
            return data
        else:
            if logger: logger.warning(f"TMDB Top Rated (Page {page}): No results found or malformed response.")
            return None
    except requests.exceptions.Timeout:
        if logger: logger.error(f"Timeout: TMDB Top Rated API request timed out for page {page}.")
    except requests.exceptions.RequestException as e:
        if logger: logger.error(f"Error calling TMDB Top Rated API for page {page}: {e}")
    except Exception as e:
        if logger: logger.error(f"Unexpected error during TMDB Top Rated lookup for page {page}: {e}")
    return None

def search_tmdb_for_movie_id(
    tmdb_api_key: str,
    movie_title: str,
    year: Optional[str] = None,
    logger: Optional[Any] = None
) -> Tuple[Optional[int], Optional[str]]:
    if not tmdb_api_key: return None, None
    if not movie_title or not movie_title.strip(): return None, None

    safe_title = requests.utils.quote(movie_title)
    url = f"https://api.themoviedb.org/3/search/movie?query={safe_title}&include_adult=false&language=en-US&page=1"
    year_to_query_tmdb = None
    if year:
        year_str = str(year).strip()
        if year_str.isdigit() and len(year_str) == 4:
            year_to_query_tmdb = year_str
            url += f"&primary_release_year={year_to_query_tmdb}"

    headers = {"accept": "application/json", "Authorization": f"Bearer {tmdb_api_key}"}
    try:
        if logger: logger.debug(f"Querying TMDB search for '{movie_title}' (Year: {year_to_query_tmdb or 'Any'})...")
        response = requests.get(url, headers=headers, timeout=7)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            target_title_lower = movie_title.lower()
            best_match = None
            for result in data["results"]:
                tmdb_title_result_lower = result.get("title", "").lower()
                tmdb_year_result_api = result.get("release_date", "N/A")[:4]
                if year_to_query_tmdb and tmdb_year_result_api == year_to_query_tmdb and tmdb_title_result_lower == target_title_lower:
                    best_match = result
                    break
                if not best_match and tmdb_title_result_lower == target_title_lower:
                    best_match = result
                if not best_match and year_to_query_tmdb and tmdb_year_result_api == year_to_query_tmdb and target_title_lower in tmdb_title_result_lower:
                    best_match = result

            if not best_match and data["results"]: best_match = data["results"][0]

            if best_match:
                tmdb_id = best_match.get("id")
                tmdb_year_result_found = best_match.get("release_date", "N/A")[:4]
                return tmdb_id, tmdb_year_result_found
    except Exception as e:
        log_msg = f"Error/Timeout during TMDB search for '{movie_title}': {e}"
        if logger: logger.warning(log_msg)
    return None, None

def get_imdb_id_from_tmdb_details(
    tmdb_api_key: str,
    tmdb_movie_id: int,
    movie_title_for_log: str = "",
    logger: Optional[Any] = None
) -> Optional[str]:
    if not tmdb_api_key or not tmdb_movie_id: return None
    url = f"https://api.themoviedb.org/3/movie/{tmdb_movie_id}/external_ids"
    headers = {"accept": "application/json", "Authorization": f"Bearer {tmdb_api_key}"}
    try:
        log_msg_query = f"Querying TMDB external IDs for TMDB ID: {tmdb_movie_id} ('{movie_title_for_log}')..."
        if logger: logger.debug(log_msg_query)
        response = requests.get(url, headers=headers, timeout=7)
        response.raise_for_status()
        data = response.json()
        imdb_id = data.get("imdb_id")
        if imdb_id and imdb_id.startswith("tt"):
            return imdb_id
    except Exception as e:
        log_msg_error = f"Error/Timeout during TMDB external IDs lookup for TMDB ID {tmdb_movie_id}: {e}"
        if logger: logger.warning(log_msg_error)
    return None

def fetch_raw_character_actor_list_from_tmdb(
    tmdb_api_key: str,
    tmdb_movie_id: int,
    movie_title_for_log: str,
    max_chars: int,
    logger: Optional[Any] = None
) -> Optional[List[TMDBRawCharacter]]:
    if not tmdb_api_key: return None
    if not tmdb_movie_id: return None

    url = f"https://api.themoviedb.org/3/movie/{tmdb_movie_id}/credits?language=en-US"
    headers = {"accept": "application/json", "Authorization": f"Bearer {tmdb_api_key}"}
    raw_char_actor_pydantic_list: List[TMDBRawCharacter] = []
    try:
        log_msg_query = f"Querying TMDB Credits for '{movie_title_for_log}' (TMDB ID: {tmdb_movie_id})..."
        if logger: logger.debug(log_msg_query)

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "cast" in data and data["cast"]:
            sorted_cast = sorted(data["cast"], key=lambda x: x.get("order", float('inf')))
            for cast_member in sorted_cast[:max_chars]:
                char_name_raw = cast_member.get("character", "").strip()
                actor_name_raw = cast_member.get("name", "").strip()
                person_id = cast_member.get("id")

                if not (char_name_raw and 2 <= len(char_name_raw) <= 70 and actor_name_raw and person_id):
                    continue

                try:
                    raw_char_actor_pydantic_list.append(
                        TMDBRawCharacter(
                            tmdb_character_name=char_name_raw,
                            tmdb_actor_name=actor_name_raw,
                            tmdb_person_id=person_id
                        )
                    )
                except Exception as e:
                    log_msg_pydantic_error = f"Pydantic validation error for TMDB char: {e}. Skipping."
                    if logger: logger.warning(f"  {log_msg_pydantic_error}")
                    continue

            if raw_char_actor_pydantic_list:
                log_msg_success = f"TMDB Credits: Retrieved {len(raw_char_actor_pydantic_list)} raw characters for '{movie_title_for_log}'."
                if logger: logger.debug(log_msg_success)
                return raw_char_actor_pydantic_list
            else:
                log_msg_no_valid = f"TMDB Credits: No valid raw character data for '{movie_title_for_log}'."
                if logger: logger.info(log_msg_no_valid)
                return None
        else:
            log_msg_no_cast = f"TMDB Credits: No 'cast' array for '{movie_title_for_log}'."
            if logger: logger.info(log_msg_no_cast)
            return None
    except Exception as e:
        log_msg_error = f"Error during TMDB Credits lookup for '{movie_title_for_log}': {e}"
        if logger: logger.error(log_msg_error)
    return None


def fetch_movie_reviews_from_tmdb(
    tmdb_api_key: str,
    movie_id: int,
    movie_title_for_log: str,
    logger: Optional[Any] = None,
    max_reviews_to_process: int = 5,
    max_review_length_chars: int = 1000
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches user reviews for a movie from TMDB.
    Returns a list of review content strings, or None if an error occurs or no reviews.
    """
    if not tmdb_api_key:
        if logger: logger.error(f"TMDB API key not provided. Cannot fetch reviews for movie ID {movie_id}.")
        return None
    if not movie_id:
        if logger: logger.error("Movie ID not provided. Cannot fetch reviews.")
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?language=en-US&page=1"
    headers = {"accept": "application/json", "Authorization": f"Bearer {tmdb_api_key}"}

    review_contents: List[str] = []

    try:
        if logger: logger.debug(f"Querying TMDB for reviews for '{movie_title_for_log}' (ID: {movie_id})...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data and data.get("results"):
            results = data["results"]
            if logger: logger.debug(f"TMDB Reviews: Found {len(results)} reviews on page 1 for '{movie_title_for_log}'. Processing up to {max_reviews_to_process}.")

            count = 0
            for review_data in results:
                if count >= max_reviews_to_process:
                    break
                content = review_data.get("content")
                author = review_data.get("author", "Unknown Author")
                if content:
                    truncated_content = content[:max_review_length_chars]
                    if len(content) > max_review_length_chars:
                        truncated_content += "..."
                    review_contents.append(f"Review by {author}:\n{truncated_content}\n---")
                    count += 1

            if not review_contents:
                if logger: logger.info(f"No review content found or extracted for '{movie_title_for_log}'.")
                return None
            return review_contents
        else:
            if logger: logger.info(f"No review results found in TMDB response for '{movie_title_for_log}'.")
            return None
    except requests.exceptions.Timeout:
        if logger: logger.error(f"Timeout: TMDB reviews API request timed out for movie ID {movie_id}.")
    except requests.exceptions.RequestException as e:
        if logger: logger.error(f"Error calling TMDB reviews API for movie ID {movie_id}: {e}")
    except Exception as e:
        if logger: logger.error(f"Unexpected error during TMDB reviews lookup for movie ID {movie_id}: {e}")
    return None