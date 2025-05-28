# data_providers/omdb_api.py
import requests
import urllib.parse
from typing import Optional, Dict, Any, List # List is not used here, can be removed if not needed elsewhere

def get_imdb_id_from_omdb(
    omdb_api_key: str,
    movie_title: str,
    year: Optional[str] = None,
    logger: Optional[Any] = None # ADDED logger argument
) -> Optional[str]:
    if not omdb_api_key:
        if logger: logger.debug("OMDB API key not provided. Skipping OMDB lookup.")
        return None
    if not movie_title or not movie_title.strip():
        if logger: logger.debug("Movie title not provided for OMDB lookup. Skipping.")
        return None

    safe_title = urllib.parse.quote_plus(movie_title)
    url = f"http://www.omdbapi.com/?apikey={omdb_api_key}&s={safe_title}"
    if year:
        url += f"&y={year}"

    log_context_for_omdb = f"OMDB for '{movie_title}' (Year: {year or 'Any'})"

    try:
        if logger: logger.debug(f"Querying {log_context_for_omdb}...")
        # else: print(f"      Querying {log_context_for_omdb}...") # Keep for direct calls without logger

        response = requests.get(url, timeout=7)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()

        if data.get("Response") == "True":
            if "Search" in data and data["Search"]:
                if year:
                    for item in data["Search"]:
                        item_year_str = item.get("Year", "")
                        if str(year) == item_year_str or str(year) in item_year_str:
                            imdb_id = item.get("imdbID")
                            if imdb_id:
                                if logger: logger.debug(f"Found IMDb ID {imdb_id} via OMDB search (exact year match).")
                                return imdb_id
                # Fallback to the first result
                if data["Search"]: # Ensure Search list is not empty
                    imdb_id = data["Search"][0].get("imdbID")
                    if imdb_id:
                        if logger: logger.debug(f"Found IMDb ID {imdb_id} via OMDB search (first result fallback).")
                        return imdb_id
            elif "imdbID" in data:
                imdb_id = data.get("imdbID")
                if logger: logger.debug(f"Found IMDb ID {imdb_id} via OMDB direct match.")
                return imdb_id
        elif data.get("Error"):
            if logger: logger.info(f"{log_context_for_omdb}: OMDB API error: {data['Error']}")
            # else: print(f"      {log_context_for_omdb}: OMDB API error: {data['Error']}")
        else:
            if logger: logger.info(f"{log_context_for_omdb}: No valid IMDb ID found in OMDB response.")

        return None
    except requests.exceptions.Timeout:
        if logger: logger.warning(f"{log_context_for_omdb}: OMDB API request timed out.")
        # else: print(f"      {log_context_for_omdb}: OMDB API request timed out.")
    except requests.exceptions.RequestException as e:
        if logger: logger.warning(f"{log_context_for_omdb}: Error calling OMDB API: {e}")
        # else: print(f"      {log_context_for_omdb}: Error calling OMDB API: {e}")
    except Exception as e:
        if logger: logger.error(f"{log_context_for_omdb}: Unexpected error during OMDB lookup: {e}")
        # else: print(f"      {log_context_for_omdb}: Unexpected error during OMDB lookup: {e}")
    return None