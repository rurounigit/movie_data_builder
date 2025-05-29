# movie_enrichment_project/main_orchestrator.py
import os
import time
import yaml
from dotenv import load_dotenv
import openai # For the client
from typing import Optional, Dict, Any, List

# Project local imports
from utils.helpers import (
    load_full_movie_data_from_yaml, save_movie_data_to_yaml,
    words_to_tokens, setup_logging
)
from models.movie_models import (
    MovieEntry, LLMCall1Output, LLMCall2Output, LLMCall3Output,
    TMDBMovieResult, TMDBRawCharacter, RelatedMovie, Recommendation,
    CharacterListItem, Relationship # Ensure all are imported
)
from data_providers import tmdb_api, omdb_api, llm_clients
from enrichers import movie_data_enricher, character_enricher, analytical_enricher

# Load environment variables from .env file
load_dotenv()

# --- Global API Keys (loaded once) ---
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# --- Configuration Loading Functions ---
def load_app_config(config_path="configs/main_config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"CRITICAL: Main configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"CRITICAL: Error parsing main configuration file {config_path}: {e}")
        exit(1)

def load_llm_providers_config(config_path="configs/llm_providers_config.yaml", logger: Optional[Any] = None) -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if "providers" in config_data and isinstance(config_data["providers"], dict):
                return config_data["providers"]
            else:
                msg = f"'providers' key missing or not a dictionary in {config_path}"
                if logger: logger.critical(msg)
                else: print(f"CRITICAL: {msg}")
                exit(1)
    except FileNotFoundError:
        msg = f"LLM providers configuration file not found at {config_path}"
        if logger: logger.critical(msg)
        else: print(f"CRITICAL: {msg}")
        exit(1)
    except yaml.YAMLError as e:
        msg = f"Error parsing LLM providers configuration file {config_path}: {e}"
        if logger: logger.critical(msg)
        else: print(f"CRITICAL: {msg}")
        exit(1)

def load_prompt_template(prompt_path: str, logger: Optional[Any] = None) -> str:
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        message = f"CRITICAL: Prompt template file not found at {prompt_path}"
        if logger: logger.critical(message)
        else: print(message)
        exit(1)

# --- IMDb ID Fetching (Master Function) ---
def fetch_master_imdb_id(
    app_config_for_api_keys: Dict[str, Any], # Not used currently as keys are global
    logger: Any,
    title_or_tmdb_id: Any,
    year_hint: Optional[str] = None,
    is_tmdb_id: bool = False,
    object_type_for_log: str = "movie"
) -> Optional[str]:
    log_prefix = f"IMDbFetch ({object_type_for_log} '{str(title_or_tmdb_id)[:30]}'):"
    imdb_id: Optional[str] = None

    if not TMDB_API_KEY and not OMDB_API_KEY: # Uses global TMDB_API_KEY, OMDB_API_KEY
        logger.warning(f"{log_prefix} Both TMDB and OMDB API keys missing. Cannot fetch IMDb ID.")
        return None

    title_str: str = ""
    tmdb_id_int: Optional[int] = None

    if is_tmdb_id and title_or_tmdb_id:
        try:
            tmdb_id_int = int(title_or_tmdb_id)
            if TMDB_API_KEY:
                logger.debug(f"{log_prefix} Attempting IMDb ID from TMDB details using TMDB ID {tmdb_id_int}.")
                imdb_id = tmdb_api.get_imdb_id_from_tmdb_details(TMDB_API_KEY, tmdb_id_int, str(title_or_tmdb_id), logger)
                if imdb_id: return imdb_id
        except ValueError:
            logger.warning(f"{log_prefix} Provided TMDB ID '{title_or_tmdb_id}' is not an int. Treating as title.")
            title_str = str(title_or_tmdb_id)
            is_tmdb_id = False # No longer treating as TMDB ID for search
    else:
        title_str = str(title_or_tmdb_id)

    valid_year_for_api: Optional[str] = None
    if year_hint:
        year_str_temp = str(year_hint).strip()
        if year_str_temp.isdigit() and len(year_str_temp) == 4:
            valid_year_for_api = year_str_temp
        elif logger:
            logger.debug(f"{log_prefix} Invalid year_hint format '{year_hint}'. Ignoring for API calls.")

    if OMDB_API_KEY and title_str and valid_year_for_api:
        logger.debug(f"{log_prefix} Attempting OMDB with title '{title_str}' and year '{valid_year_for_api}'.")
        imdb_id = omdb_api.get_imdb_id_from_omdb(OMDB_API_KEY, title_str, valid_year_for_api, logger)
        if imdb_id: return imdb_id

    if TMDB_API_KEY and title_str and valid_year_for_api:
        logger.debug(f"{log_prefix} Attempting TMDB search for '{title_str}' year '{valid_year_for_api}', then details.")
        tmdb_id_from_search, _ = tmdb_api.search_tmdb_for_movie_id(TMDB_API_KEY, title_str, valid_year_for_api, logger)
        if tmdb_id_from_search:
            imdb_id = tmdb_api.get_imdb_id_from_tmdb_details(TMDB_API_KEY, tmdb_id_from_search, title_str, logger)
            if imdb_id: return imdb_id

    if OMDB_API_KEY and title_str:
        logger.debug(f"{log_prefix} Attempting OMDB with title '{title_str}' only.")
        imdb_id = omdb_api.get_imdb_id_from_omdb(OMDB_API_KEY, title_str, None, logger)
        if imdb_id: return imdb_id

    if TMDB_API_KEY and title_str:
        logger.debug(f"{log_prefix} Attempting TMDB search for '{title_str}' only, then details.")
        tmdb_id_from_search_title_only, _ = tmdb_api.search_tmdb_for_movie_id(TMDB_API_KEY, title_str, None, logger)
        if tmdb_id_from_search_title_only:
            imdb_id = tmdb_api.get_imdb_id_from_tmdb_details(TMDB_API_KEY, tmdb_id_from_search_title_only, title_str, logger)
            if imdb_id: return imdb_id

    if not imdb_id:
        logger.info(f"{log_prefix} Failed to find IMDb ID for '{str(title_or_tmdb_id)}' (Year hint: {year_hint or 'N/A'}) after all attempts.")
    return None


# --- Main Orchestration Function ---
def run_enrichment_pipeline():
    app_config = load_app_config()
    logger = setup_logging(app_config['raw_log_file'], logger_name="MovieEnrichmentPipeline")
    llm_providers_config = load_llm_providers_config(logger=logger)

    # --- Determine Active LLM Configuration ---
    active_provider_id = app_config.get("active_llm_provider_id")
    if not active_provider_id or active_provider_id not in llm_providers_config:
        logger.critical(f"Active LLM provider ID '{active_provider_id}' not specified in main_config.yaml or not found in llm_providers_config.yaml. Exiting.")
        return

    active_llm_config = llm_providers_config[active_provider_id]
    llm_description = active_llm_config.get("description", active_provider_id)
    llm_base_url_from_provider = active_llm_config.get("base_url")
    api_key_env_var_name = active_llm_config.get("api_key_env_var")
    llm_api_key_value_from_env = os.getenv(api_key_env_var_name) if api_key_env_var_name else None
    llm_model_id_for_api_calls = active_llm_config.get("model_id")
    llm_provider_type = active_llm_config.get("type", "openai_compatible")

    logger.info(f"===== MOVIE ENRICHMENT SESSION STARTED =====")
    logger.info(f"Using LLM Provider: {llm_description} (ID: {active_provider_id})")
    logger.info(f"LLM Model ID for API calls: {llm_model_id_for_api_calls}")
    logger.info(f"Update existing movies: {app_config.get('update_existing_movies', False)}")
    if app_config.get('update_existing_movies', False):
        logger.info(f"Keys to update for existing: {app_config.get('keys_to_update_for_existing', 'ALL (if enricher active)')}")
    logger.info(f"Active enrichers: {app_config.get('active_enrichers', {})}")

    if not TMDB_API_KEY:
        logger.critical("TMDB_API_KEY not set. Core movie fetching will fail. Exiting.")
        return
    if not OMDB_API_KEY:
        logger.warning("OMDB_API_KEY not set. IMDb ID lookups will be limited to TMDB only.")

    if not llm_model_id_for_api_calls:
        logger.critical(f"No 'model_id' specified for active LLM provider '{active_provider_id}'. Exiting.")
        return

    # --- Initialize LLM Client ---
    llm_client_instance = None
    if llm_provider_type == "openai_compatible":
        # For official OpenAI, base_url might be None to use library default.
        # For others like LM Studio or Gemini compatible, base_url is required.
        resolved_base_url = llm_base_url_from_provider
        if not llm_base_url_from_provider and "openai_" in active_provider_id: # Heuristic for official OpenAI
            logger.info(f"Using default OpenAI base URL for provider '{active_provider_id}'.")
            resolved_base_url = None # Let openai client use its default
        elif not llm_base_url_from_provider:
             logger.critical(f"LLM_BASE_URL not configured for non-official OpenAI provider '{active_provider_id}' in llm_providers_config.yaml. Exiting.")
             return


        if not llm_api_key_value_from_env and api_key_env_var_name:
            logger.warning(f"API key environment variable '{api_key_env_var_name}' for provider '{active_provider_id}' is not set in .env file.")
            # Some OpenAI-compatible servers (like local LM Studio) might not require an API key.
            # Official OpenAI or Google Gemini compatible endpoint will require it.

        try:
            llm_client_instance = openai.OpenAI(base_url=resolved_base_url, api_key=llm_api_key_value_from_env)
            logger.info(f"OpenAI-compatible LLM client initialized. Provider: {active_provider_id}, Base URL: {resolved_base_url or 'OpenAI Default'}")
        except Exception as e:
            logger.critical(f"Failed to initialize OpenAI-compatible LLM client for provider '{active_provider_id}': {e}. Exiting.")
            return
    else:
        logger.critical(f"Unsupported LLM provider type: '{llm_provider_type}' for provider '{active_provider_id}'. Exiting.")
        return

    if not os.path.exists(app_config['character_image_save_path']):
        try:
            os.makedirs(app_config['character_image_save_path'], exist_ok=True)
            logger.info(f"Created character image directory: {app_config['character_image_save_path']}")
        except OSError as e:
            logger.error(f"Could not create character image directory {app_config['character_image_save_path']}: {e}. Images may not save.")

    all_movie_entries_master_list: List[MovieEntry] = []
    raw_data_from_file = load_full_movie_data_from_yaml(app_config['output_file'])
    for item_dict in raw_data_from_file:
        try:
            all_movie_entries_master_list.append(MovieEntry.model_validate(item_dict))
        except Exception as e:
            logger.warning(f"Could not validate existing movie data for '{item_dict.get('movie_title', 'Unknown Title')}': {e}. Skipping this entry from file.")
    logger.info(f"Loaded {len(all_movie_entries_master_list)} valid movie entries from '{app_config['output_file']}'.")

    processed_movie_titles_lower_set = {entry.movie_title.lower().strip() for entry in all_movie_entries_master_list}

    prompt_call1_template = load_prompt_template(app_config["prompts"]["call1_initial_data"], logger)
    prompt_call2_template = load_prompt_template(app_config["prompts"]["call2_chars_rels"], logger)
    prompt_call3_template = load_prompt_template(app_config["prompts"]["call3_analytical"], logger)

    new_movies_added_this_session = 0
    current_tmdb_page = 1
    session_api_movie_attempt_count = 0

    active_enrichers_cfg = app_config.get('active_enrichers', {})
    update_existing_cfg = app_config.get('update_existing_movies', False)
    keys_to_update_cfg = app_config.get('keys_to_update_for_existing', [])
    update_all_active_fields_for_existing = not bool(keys_to_update_cfg)

    key_to_enricher_group_map = {
        "character_profile": "initial_data", "critical_reception": "initial_data",
        "visual_style": "initial_data", "most_talked_about_related_topic": "initial_data",
        "complex_search_queries": "initial_data", "sequel": "initial_data", "prequel": "initial_data",
        "spin_off_of": "initial_data", "spin_off": "initial_data", "remake_of": "initial_data",
        "remake": "initial_data",
        "character_list": "characters_and_relations", "relationships": "characters_and_relations",
        "character_profile_big5": "analytical_data", "character_profile_myersbriggs": "analytical_data",
        "genre_mix": "analytical_data", "matching_tags": "analytical_data", "recommendations": "analytical_data",
        "imdb_id": "fetch_imdb_ids" # Main movie imdb_id is special
    }

    while (new_movies_added_this_session < app_config['num_new_movies_to_fetch_this_session'] or \
           (update_existing_cfg and current_tmdb_page <= app_config['max_tmdb_top_rated_pages_to_check'])) and \
          current_tmdb_page <= app_config['max_tmdb_top_rated_pages_to_check']:

        if not update_existing_cfg and new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session']:
            logger.info("Target for new movies reached and not updating existing. Ending TMDB fetch.")
            break

        logger.info(f"--- Fetching TMDB Top Rated Page: {current_tmdb_page} ---")
        tmdb_page_data_raw = tmdb_api.fetch_top_rated_movies_from_tmdb(TMDB_API_KEY, current_tmdb_page, logger)

        if not tmdb_page_data_raw or not tmdb_page_data_raw.get("results"):
            logger.warning(f"No results on TMDB Top Rated page {current_tmdb_page} or failed to fetch.")
            if not tmdb_page_data_raw or ("total_pages" in tmdb_page_data_raw and current_tmdb_page >= tmdb_page_data_raw.get("total_pages", current_tmdb_page)):
                logger.info("Reached end of TMDB pages or TMDB fetch error limit.")
                break
            current_tmdb_page += 1
            time.sleep(app_config.get('api_request_delay_seconds_tmdb_page', 1))
            continue

        movies_on_this_page_raw = tmdb_page_data_raw["results"]
        total_tmdb_pages = tmdb_page_data_raw.get("total_pages", current_tmdb_page)
        found_processable_movie_on_page = False

        for tmdb_movie_raw_dict in movies_on_this_page_raw:
            session_api_movie_attempt_count += 1
            try:
                tmdb_movie_candidate = TMDBMovieResult.model_validate(tmdb_movie_raw_dict)
            except Exception as e:
                logger.warning(f"Skipping TMDB entry due to validation error: {e} - Data: {str(tmdb_movie_raw_dict)[:100]}")
                continue

            if not tmdb_movie_candidate.title or tmdb_movie_candidate.id is None or not tmdb_movie_candidate.year:
                logger.info(f"Skipping TMDB entry with missing title, ID, or year: '{tmdb_movie_candidate.title}' ID:{tmdb_movie_candidate.id} Year:{tmdb_movie_candidate.year}")
                continue

            found_processable_movie_on_page = True
            current_movie_title_lower = tmdb_movie_candidate.title.lower().strip()
            is_updating_existing_movie = current_movie_title_lower in processed_movie_titles_lower_set
            existing_movie_entry: Optional[MovieEntry] = None

            if is_updating_existing_movie:
                if not update_existing_cfg:
                    logger.debug(f"Movie '{tmdb_movie_candidate.title}' already processed and update_existing_movies is false. Skipping.")
                    continue
                existing_movie_entry = next((m for m in all_movie_entries_master_list if m.movie_title.lower().strip() == current_movie_title_lower), None)
                if not existing_movie_entry:
                    logger.error(f"Consistency Error: Movie '{tmdb_movie_candidate.title}' in processed set but not found in master list. Skipping update.")
                    continue
                logger.info(f"--- Updating Existing Movie: '{existing_movie_entry.movie_title}' ({existing_movie_entry.movie_year}) TMDB_ID: {existing_movie_entry.tmdb_movie_id or tmdb_movie_candidate.id} ---")
                movie_title_for_calls = existing_movie_entry.movie_title
                movie_year_for_calls = existing_movie_entry.movie_year
                current_tmdb_id_for_calls = existing_movie_entry.tmdb_movie_id or tmdb_movie_candidate.id
                working_data_dict = existing_movie_entry.model_dump(exclude_none=False)
            else:
                if new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session']:
                    logger.info(f"Target for new movies ({app_config['num_new_movies_to_fetch_this_session']}) reached. Skipping new movie '{tmdb_movie_candidate.title}'.")
                    continue

                logger.info(f"--- Processing New Movie: '{tmdb_movie_candidate.title}' ({tmdb_movie_candidate.year}) TMDB_ID: {tmdb_movie_candidate.id} ---")
                movie_title_for_calls = tmdb_movie_candidate.title
                movie_year_for_calls = tmdb_movie_candidate.year
                current_tmdb_id_for_calls = tmdb_movie_candidate.id
                working_data_dict = {"movie_title": movie_title_for_calls, "movie_year": movie_year_for_calls, "tmdb_movie_id": current_tmdb_id_for_calls}

            def should_update_field(field_name: str) -> bool:
                if not is_updating_existing_movie: return True
                if update_all_active_fields_for_existing: return True
                return field_name in keys_to_update_cfg

            llm1_data_generated: Optional[LLMCall1Output] = None
            if active_enrichers_cfg.get('initial_data'):
                if is_updating_existing_movie and not any(k in keys_to_update_cfg for k_group in ['initial_data', 'fetch_imdb_ids'] for k in key_to_enricher_group_map if key_to_enricher_group_map[k] == k_group and k in keys_to_update_cfg) and not update_all_active_fields_for_existing:
                    logger.info(f"  Skipping Initial Data regeneration for '{movie_title_for_calls}' (update existing, relevant keys not targeted).")
                else:
                    logger.info(f"  Running: Initial Data Enrichment for '{movie_title_for_calls}'")
                    max_tokens_c1 = words_to_tokens(app_config['max_tokens_call_1_words'], app_config['words_to_tokens_ratio'])
                    llm1_data_generated = movie_data_enricher.generate_initial_movie_data(
                        llm_client_instance, llm_model_id_for_api_calls, # Use configured LLM client and model ID
                        movie_title_for_calls, movie_year_for_calls,
                        prompt_call1_template, max_tokens_c1, app_config, logger
                    )
                    if llm1_data_generated:
                        logger.info(f"  Success: Initial Data for '{movie_title_for_calls}'.")
                        update_dict_c1 = {}
                        for key, value in llm1_data_generated.model_dump(exclude={"movie_title", "movie_year"}, exclude_none=False).items():
                            if key in ["sequel", "prequel", "spin_off_of", "spin_off", "remake_of", "remake"]:
                                if isinstance(value, str) and value.strip():
                                    update_dict_c1[key] = RelatedMovie(title=value.strip()).model_dump()
                                elif isinstance(value, dict):
                                    try: update_dict_c1[key] = RelatedMovie.model_validate(value).model_dump()
                                    except Exception: update_dict_c1[key] = RelatedMovie(title=str(value.get("title","Unknown Related"))).model_dump() if value.get("title") else None
                                else: update_dict_c1[key] = None
                            else: update_dict_c1[key] = value
                        working_data_dict.update(update_dict_c1)
                    else:
                        logger.error(f"  Failure: Initial Data for '{movie_title_for_calls}'.")
                        if not is_updating_existing_movie: continue

            if active_enrichers_cfg.get('characters_and_relations'):
                if is_updating_existing_movie and not any(k in keys_to_update_cfg for k_group in ['characters_and_relations', 'fetch_character_images'] for k in key_to_enricher_group_map if key_to_enricher_group_map[k] == k_group and k in keys_to_update_cfg) and not update_all_active_fields_for_existing:
                     logger.info(f"  Skipping Characters/Relations for '{movie_title_for_calls}' (update existing, relevant keys not targeted).")
                else:
                    logger.info(f"  Running: Character and Relations Enrichment for '{movie_title_for_calls}'")
                    raw_chars_data = tmdb_api.fetch_raw_character_actor_list_from_tmdb(
                        TMDB_API_KEY, current_tmdb_id_for_calls, movie_title_for_calls,
                        app_config['max_characters_from_tmdb'], logger
                    )
                    if raw_chars_data:
                        raw_chars_yaml_for_prompt = yaml.dump(
                            [char.model_dump() for char in raw_chars_data],
                            sort_keys=False, allow_unicode=True, indent=2
                        )
                        num_chars = len(raw_chars_data)
                        dynamic_words_c2 = app_config['max_tokens_enrich_rel_call_base_words'] + \
                                           (num_chars * app_config['max_tokens_enrich_rel_char_desc_words']) + \
                                           (num_chars * app_config['max_tokens_enrich_rel_char_rels_words'])
                        max_tokens_c2 = words_to_tokens(dynamic_words_c2, app_config['words_to_tokens_ratio'])

                        llm2_output = character_enricher.enrich_characters_and_get_relationships(
                            llm_client_instance, llm_model_id_for_api_calls, # Use configured LLM client and model ID
                            movie_title_for_calls, movie_year_for_calls,
                            raw_chars_yaml_for_prompt, prompt_call2_template, max_tokens_c2, app_config, logger
                        )
                        if llm2_output and llm2_output.character_list:
                            logger.info(f"  Success: LLM Call 2 (Chars/Rels) for '{movie_title_for_calls}'.")
                            temp_char_list_models = llm2_output.character_list # These are CharacterListItem models
                            if active_enrichers_cfg.get('fetch_character_images'):
                                logger.info(f"    Fetching character images for '{movie_title_for_calls}'")
                                temp_char_list_models = character_enricher.fetch_and_assign_character_images(
                                    temp_char_list_models, app_config['character_image_save_path'],
                                    TMDB_API_KEY, app_config['tmdb_image_base_url'], app_config['tmdb_image_size'], logger
                                )
                            else:
                                logger.info(f"    Skipping character image fetching for '{movie_title_for_calls}' as per config.")

                            working_data_dict["character_list"] = [char.model_dump() for char in temp_char_list_models]
                            working_data_dict["relationships"] = [rel.model_dump() for rel in character_enricher.deduplicate_and_normalize_relationships(
                                temp_char_list_models, llm2_output.relationships or [], logger # Ensure relationships is a list
                            )]
                        else:
                            logger.error(f"  Failure: LLM Call 2 (Chars/Rels) for '{movie_title_for_calls}'.")
                            if not is_updating_existing_movie: continue
                    else:
                        logger.error(f"  Failure: Could not fetch TMDB raw characters for '{movie_title_for_calls}'. Skipping Chars/Rels block.")
                        if not is_updating_existing_movie: continue

            llm3_output_data = None # Initialize to handle cases where analytical data is skipped or fails
            if active_enrichers_cfg.get('analytical_data'):
                if is_updating_existing_movie and not any(k in keys_to_update_cfg for k_group in ['analytical_data', 'fetch_imdb_ids'] for k in key_to_enricher_group_map if key_to_enricher_group_map[k] == k_group and k in keys_to_update_cfg) and not update_all_active_fields_for_existing:
                    logger.info(f"  Skipping Analytical Data regeneration for '{movie_title_for_calls}' (update existing, relevant keys not targeted).")
                else:
                    logger.info(f"  Running: Analytical Data Enrichment for '{movie_title_for_calls}'")
                    max_tokens_c3 = words_to_tokens(app_config['max_tokens_analytical_call_words'], app_config['words_to_tokens_ratio'])
                    llm3_output_data = analytical_enricher.generate_analytical_data(
                        llm_client_instance, llm_model_id_for_api_calls, # Use configured LLM client and model ID
                        movie_title_for_calls, movie_year_for_calls,
                        prompt_call3_template, max_tokens_c3, app_config, logger
                    )
                    if llm3_output_data:
                        logger.info(f"  Success: Analytical Data for '{movie_title_for_calls}'.")
                        working_data_dict.update(llm3_output_data.model_dump(exclude_none=False))
                    else:
                        logger.warning(f"  Failure or incomplete: Analytical Data for '{movie_title_for_calls}'.")
                        if not is_updating_existing_movie:
                            logger.error(f"    Critical analytical data missing for new movie '{movie_title_for_calls}'. Skipping final assembly.")
                            continue

            # Ensure analytical fields are None if not generated, for MovieEntry validation
            if not llm3_output_data and active_enrichers_cfg.get('analytical_data'):
                for fld_key in LLMCall3Output.model_fields.keys():
                    if fld_key not in working_data_dict:
                        working_data_dict[fld_key] = None


            if active_enrichers_cfg.get('fetch_imdb_ids'):
                if is_updating_existing_movie and not ("imdb_id" in keys_to_update_cfg or \
                    any(k in keys_to_update_cfg for k in ["sequel","prequel","recommendations", "spin_off", "spin_off_of", "remake", "remake_of"]) ) and \
                    not update_all_active_fields_for_existing :
                    logger.info(f"  Skipping IMDb ID fetching for '{movie_title_for_calls}' (update existing, relevant keys not targeted).")
                else:
                    logger.info(f"  Fetching/Updating IMDb IDs for '{movie_title_for_calls}' and its relations.")
                    if not is_updating_existing_movie or should_update_field("imdb_id"):
                        working_data_dict["imdb_id"] = fetch_master_imdb_id(app_config, logger, current_tmdb_id_for_calls, movie_year_for_calls, True, f"main movie {movie_title_for_calls}")

                    for rel_key in ["sequel", "prequel", "spin_off_of", "spin_off", "remake_of", "remake"]:
                        related_movie_val = working_data_dict.get(rel_key)
                        if related_movie_val and (not is_updating_existing_movie or should_update_field(rel_key)):
                            title_to_search = None
                            year_hint_for_related = None # LLM Call 1 does not provide year for these simple relations

                            # Ensure related_movie_val is a dict for processing
                            if isinstance(related_movie_val, str) and related_movie_val.strip():
                                title_to_search = related_movie_val.strip()
                                working_data_dict[rel_key] = {"title": title_to_search, "imdb_id": None} # Convert to dict
                            elif isinstance(related_movie_val, dict) and related_movie_val.get("title"):
                                title_to_search = related_movie_val["title"]
                            # Else, related_movie_val might be None or an improperly structured dict from previous state

                            if title_to_search:
                                related_imdb_id = fetch_master_imdb_id(app_config, logger, title_to_search, year_hint_for_related, False, f"related {rel_key} {title_to_search}")
                                # working_data_dict[rel_key] should already be a dict here if title_to_search was set
                                if isinstance(working_data_dict.get(rel_key), dict):
                                     working_data_dict[rel_key]["imdb_id"] = related_imdb_id
                                else: # Should ideally not happen if conversion above worked
                                     logger.warning(f"Could not set IMDb ID for {rel_key} '{title_to_search}' as its structure in working_data_dict is not a dict. Value: {working_data_dict.get(rel_key)}")


                    if working_data_dict.get("recommendations") and isinstance(working_data_dict["recommendations"], list):
                        recs_list_of_dicts = working_data_dict["recommendations"] # Should be list of Recommendation model_dumps
                        for rec_dict_idx, rec_dict_item in enumerate(recs_list_of_dicts):
                             if isinstance(rec_dict_item, dict) and rec_dict_item.get("title") and (not is_updating_existing_movie or should_update_field("recommendations")):
                                rec_title = rec_dict_item["title"]
                                rec_year = str(rec_dict_item.get("year",""))
                                rec_imdb_id = fetch_master_imdb_id(app_config, logger, rec_title, rec_year, False, f"recommendation {rec_title}")
                                # Update the specific dictionary in the list
                                working_data_dict["recommendations"][rec_dict_idx]["imdb_id"] = rec_imdb_id


            logger.info(f"  Finalizing entry for '{movie_title_for_calls}'.")
            try:
                final_movie_entry = MovieEntry.model_validate(working_data_dict)

                if is_updating_existing_movie and existing_movie_entry:
                    idx_to_replace = -1
                    for i, entry in enumerate(all_movie_entries_master_list):
                        if entry.movie_title.lower().strip() == final_movie_entry.movie_title.lower().strip():
                            idx_to_replace = i; break
                    if idx_to_replace != -1:
                        all_movie_entries_master_list[idx_to_replace] = final_movie_entry
                        logger.info(f"  Successfully updated entry for '{final_movie_entry.movie_title}' in list.")
                    else:
                        all_movie_entries_master_list.append(final_movie_entry) # Should not happen ideally
                        processed_movie_titles_lower_set.add(final_movie_entry.movie_title.lower().strip())
                else:
                    all_movie_entries_master_list.append(final_movie_entry)
                    processed_movie_titles_lower_set.add(final_movie_entry.movie_title.lower().strip())
                    new_movies_added_this_session += 1

                save_movie_data_to_yaml(
                    [entry.model_dump(exclude_none=True) for entry in all_movie_entries_master_list],
                    app_config['output_file']
                )
                logger.info(f"  Saved '{final_movie_entry.movie_title}' to '{app_config['output_file']}'.")

            except Exception as e:
                logger.error(f"  CRITICAL: Failed to validate final MovieEntry for '{movie_title_for_calls}': {e}")
                logger.debug(f"  Problematic working_data_dict for '{movie_title_for_calls}': {str(working_data_dict)[:1500]}...")
                if not is_updating_existing_movie :
                    processed_movie_titles_lower_set.add(movie_title_for_calls.lower().strip()) # Mark as attempted

            time.sleep(app_config.get('api_request_delay_seconds_general', 2))

            if not is_updating_existing_movie and new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session']:
                logger.info(f"Target for new movies ({app_config['num_new_movies_to_fetch_this_session']}) reached during page processing.")
                break # Break from movies_on_this_page loop

        # --- End of for loop for movies on current TMDB page ---

        if not update_existing_cfg and new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session']:
            logger.info("Target for new movies reached. Ending TMDB page fetching.")
            break # Break from while loop for pages

        if not found_processable_movie_on_page and not update_existing_cfg :
            logger.info(f"No new processable movies found on TMDB page {current_tmdb_page} to add. Advancing page.")

        current_tmdb_page += 1
        if current_tmdb_page > total_tmdb_pages :
             logger.info(f"Reached reported end of TMDB pages ({total_tmdb_pages}).")
             break # Reached end of all pages
        time.sleep(app_config.get('api_request_delay_seconds_tmdb_page', 1))
    # --- End of while loop for TMDB pages ---

    logger.info(f"===== MOVIE ENRICHMENT SESSION FINISHED =====")
    logger.info(f"Attempted to process {session_api_movie_attempt_count} movie candidates from TMDB API.")
    logger.info(f"Added {new_movies_added_this_session} new movie(s) this session.")
    logger.info(f"Total movies in '{app_config['output_file']}': {len(all_movie_entries_master_list)}")
    logger.info(f"Raw log: '{app_config['raw_log_file']}'")
    logger.info(f"Clean YAML data: '{app_config['output_file']}'")
    if active_enrichers_cfg.get('fetch_character_images'):
        logger.info(f"Character images saved to: '{app_config['character_image_save_path']}'")

if __name__ == "__main__":
    run_enrichment_pipeline()