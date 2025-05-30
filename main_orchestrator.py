# movie_enrichment_project/main_orchestrator.py
import os
import time
import yaml
from dotenv import load_dotenv
import openai # For the client
from typing import Optional, Dict, Any, List, Set

# Project local imports
from utils.helpers import (
    load_full_movie_data_from_yaml, save_movie_data_to_yaml, parse_index_range_string,
    words_to_tokens, setup_logging
)
from models.movie_models import (
    MovieEntry, LLMCall1Output, LLMCall2Output, LLMCall3Output,
    TMDBMovieResult, TMDBRawCharacter, RelatedMovie, Recommendation,
    CharacterListItem, Relationship # Ensure all are imported
)
from data_providers import tmdb_api, omdb_api, llm_clients
from enrichers import movie_data_enricher, character_enricher, analytical_enricher, review_summarizer_enricher, constrained_plot_rel_enricher

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
        resolved_base_url = llm_base_url_from_provider
        if not llm_base_url_from_provider and "openai_" in active_provider_id: # Heuristic for official OpenAI
            logger.info(f"Using default OpenAI base URL for provider '{active_provider_id}'.")
            resolved_base_url = None # Let openai client use its default
        elif not llm_base_url_from_provider:
             logger.critical(f"LLM_BASE_URL not configured for non-official OpenAI provider '{active_provider_id}' in llm_providers_config.yaml. Exiting.")
             return


        if not llm_api_key_value_from_env and api_key_env_var_name:
            logger.warning(f"API key environment variable '{api_key_env_var_name}' for provider '{active_provider_id}' is not set in .env file.")

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
    prompt_call4_review_summary_template = load_prompt_template(app_config["prompts"]["call4_tmdb_review_summary"], logger)
    prompt_constrained_plot_rel_template = load_prompt_template(app_config["prompts"]["call_constrained_plot_relations"], logger)

    active_enrichers_cfg = app_config.get('active_enrichers', {})
    fields_to_update_cfg = app_config.get('fields_to_update', []) # Renamed from keys_to_update_for_existing
    update_all_active_fields_for_existing = not bool(fields_to_update_cfg)

    key_to_enricher_group_map = {
        "character_profile": "initial_data", "critical_reception": "initial_data", "visual_style": "initial_data", "most_talked_about_related_topic": "initial_data", "complex_search_queries": "initial_data", "sequel": "initial_data", "prequel": "initial_data", "spin_off_of": "initial_data", "spin_off": "initial_data", "remake_of": "initial_data", "remake": "initial_data",
        "character_list": "characters_and_relations", "relationships": "characters_and_relations",
        "character_profile_big5": "analytical_data", "character_profile_myersbriggs": "analytical_data",
        "genre_mix": "analytical_data", "matching_tags": "analytical_data", "recommendations": "analytical_data",
        "imdb_id": "fetch_imdb_ids",
        "tmdb_user_review_summary": "tmdb_review_summary",
        "plot_with_character_constraints_and_relations": "constrained_plot_with_relations",
    }

    # --- COMMON MOVIE ENRICHMENT FUNCTION ---
    def _enrich_and_update_movie_data(
        movie_entry_input: MovieEntry,
        is_new_movie: bool, # True if this is a newly fetched movie, False if updating existing
        llm_client_instance: openai.OpenAI,
        llm_model_id_for_api_calls: str,
        prompt_call1_template: str,
        prompt_call2_template: str,
        prompt_call3_template: str,
        prompt_call4_review_summary_template: str,
        prompt_constrained_plot_rel_template: str,
        app_config: Dict[str, Any], # To access config like 'max_characters_from_tmdb'
        TMDB_API_KEY: str,
        OMDB_API_KEY: Optional[str],
        logger: Any,
        active_enrichers_cfg: Dict[str, Any],
        update_all_active_fields_for_existing: bool, # Derived from fields_to_update being empty/full
        fields_to_update_cfg: List[str], # The list of specific fields to update
        key_to_enricher_group_map: Dict[str, str],
    ) -> Optional[MovieEntry]:
        """
        Encapsulates the core movie data enrichment logic, applicable to both new and existing movies.
        Returns the updated MovieEntry object or None if critical validation fails.
        """
        movie_title_for_calls = movie_entry_input.movie_title
        movie_year_for_calls = movie_entry_input.movie_year
        current_tmdb_id_for_calls = movie_entry_input.tmdb_movie_id

        # If it's an existing movie being updated, start with its current data.
        # If it's a new movie, initialize with basic details and other fields as None.
        working_data_dict = movie_entry_input.model_dump(exclude_none=False) if not is_new_movie else {
            "movie_title": movie_title_for_calls,
            "movie_year": movie_year_for_calls,
            "tmdb_movie_id": current_tmdb_id_for_calls,
            # Initialize other fields to None for new movies to ensure they exist for pydantic
            "imdb_id": None, "character_profile": None, "character_profile_big5": None,
            "character_profile_myersbriggs": None, "critical_reception": None,
            "visual_style": None, "most_talked_about_related_topic": None,
            "genre_mix": None, "matching_tags": None, "complex_search_queries": [],
            "sequel": None, "prequel": None, "spin_off_of": None, "spin_off": None,
            "remake_of": None, "remake": None, "recommendations": [],
            "character_list": [], "relationships": [], "tmdb_user_review_summary": None,
            "plot_with_character_constraints_and_relations": None
        }

        # Helper for conditional field updates based on `fields_to_update_cfg`
        def should_update_field_local(field_name: str) -> bool:
            if is_new_movie: return True # Always update fields for new movies
            if update_all_active_fields_for_existing: return True # If updating all fields (fields_to_update_cfg is empty)
            return field_name in fields_to_update_cfg # If specific keys are targeted

        llm1_data_generated: Optional[LLMCall1Output] = None
        # Initial Data Enrichment (LLM Call 1)
        if active_enrichers_cfg.get('initial_data'):
            # Determine if any field associated with 'initial_data' enricher is targeted for update
            initial_data_fields_to_update = [k for k, v in key_to_enricher_group_map.items() if v == 'initial_data']
            if not is_new_movie and not update_all_active_fields_for_existing and not any(f in fields_to_update_cfg for f in initial_data_fields_to_update):
                logger.info(f"  Skipping Initial Data regeneration for '{movie_title_for_calls}' (existing, initial_data fields not targeted).")
            else:
                logger.info(f"  Running: Initial Data Enrichment for '{movie_title_for_calls}'")
                max_tokens_c1 = words_to_tokens(app_config['max_tokens_call_1_words'], app_config['words_to_tokens_ratio'])
                llm1_data_generated = movie_data_enricher.generate_initial_movie_data(
                    llm_client_instance, llm_model_id_for_api_calls,
                    movie_title_for_calls, movie_year_for_calls,
                    prompt_call1_template, max_tokens_c1, app_config, logger
                )
                if llm1_data_generated:
                    logger.info(f"  Success: Initial Data for '{movie_title_for_calls}'.")
                    for key, value in llm1_data_generated.model_dump(exclude={"movie_title", "movie_year"}, exclude_none=False).items():
                        if should_update_field_local(key):
                            if key in ["sequel", "prequel", "spin_off_of", "spin_off", "remake_of", "remake"]:
                                if isinstance(value, str) and value.strip():
                                    working_data_dict[key] = RelatedMovie(title=value.strip()).model_dump()
                                elif isinstance(value, dict):
                                    try: working_data_dict[key] = RelatedMovie.model_validate(value).model_dump()
                                    except Exception: working_data_dict[key] = RelatedMovie(title=str(value.get("title","Unknown Related"))).model_dump() if value.get("title") else None
                                else: working_data_dict[key] = None
                            else:
                                working_data_dict[key] = value
                else:
                    logger.error(f"  Failure: Initial Data for '{movie_title_for_calls}'.")

        # Character and Relations Enrichment (LLM Call 2)
        raw_chars_data: Optional[List[TMDBRawCharacter]] = None # Declare outside for constrained plot access
        if active_enrichers_cfg.get('characters_and_relations'):
            char_rel_fields_to_update = [k for k, v in key_to_enricher_group_map.items() if v in ['characters_and_relations', 'constrained_plot_with_relations']]
            if not is_new_movie and not update_all_active_fields_for_existing and not any(f in fields_to_update_cfg for f in char_rel_fields_to_update + ["fetch_character_images"]):
                 logger.info(f"  Skipping Characters/Relations for '{movie_title_for_calls}' (existing, relevant fields not targeted).")
            else:
                logger.info(f"  Running: Character and Relations Enrichment for '{movie_title_for_calls}'")
                if current_tmdb_id_for_calls: # TMDB ID is required for this step
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
                            llm_client_instance, llm_model_id_for_api_calls,
                            movie_title_for_calls, movie_year_for_calls,
                            raw_chars_yaml_for_prompt, prompt_call2_template, max_tokens_c2, app_config, logger
                        )
                        if llm2_output:
                            logger.info(f"  Success: LLM Call 2 (Chars/Rels) for '{movie_title_for_calls}'.")
                            temp_char_list_models = llm2_output.character_list
                            if active_enrichers_cfg.get('fetch_character_images') and should_update_field_local("character_list"): # Fetch images only if char list is being updated
                                logger.info(f"    Fetching character images for '{movie_title_for_calls}'")
                                temp_char_list_models = character_enricher.fetch_and_assign_character_images(
                                    temp_char_list_models, app_config['character_image_save_path'],
                                    TMDB_API_KEY, app_config['tmdb_image_base_url'], app_config['tmdb_image_size'], logger
                                )
                            else:
                                logger.info(f"    Skipping character image fetching for '{movie_title_for_calls}' as per config or update target.")

                            if should_update_field_local("character_list"):
                                working_data_dict["character_list"] = [char.model_dump() for char in temp_char_list_models]
                            if should_update_field_local("relationships"):
                                working_data_dict["relationships"] = [rel.model_dump() for rel in character_enricher.deduplicate_and_normalize_relationships(
                                    temp_char_list_models, llm2_output.relationships or [], logger
                                )]

                            if active_enrichers_cfg.get('constrained_plot_with_relations'):
                                if should_update_field_local("plot_with_character_constraints_and_relations"):
                                    if raw_chars_data: # Ensure it was successfully fetched
                                        tmdb_original_char_names = [char.tmdb_character_name for char in raw_chars_data if char.tmdb_character_name]
                                        relationships_for_context = llm2_output.relationships if llm2_output.relationships else []
                                        if tmdb_original_char_names: # Must have characters to constrain plot
                                            logger.info(f"  Generating Constrained Plot (with relations) for '{movie_title_for_calls}'.")
                                            max_tokens_plot_rel = words_to_tokens(
                                                app_config.get('max_tokens_constrained_plot_relations_words', 350),
                                                app_config['words_to_tokens_ratio']
                                            )
                                            plot_rel_output = constrained_plot_rel_enricher.generate_constrained_plot_with_relations(
                                                llm_client_instance,
                                                llm_model_id_for_api_calls,
                                                movie_title_for_calls,
                                                movie_year_for_calls,
                                                tmdb_original_char_names,
                                                relationships_for_context,
                                                prompt_constrained_plot_rel_template,
                                                max_tokens_plot_rel,
                                                logger
                                            )
                                            if plot_rel_output and plot_rel_output.plot_with_character_constraints_and_relations:
                                                working_data_dict["plot_with_character_constraints_and_relations"] = plot_rel_output.plot_with_character_constraints_and_relations
                                                logger.info(f"    Success: Generated constrained plot (with relations) for '{movie_title_for_calls}'.")
                                            else:
                                                logger.warning(f"    Could not generate constrained plot (with relations) for '{movie_title_for_calls}'.")
                                                working_data_dict["plot_with_character_constraints_and_relations"] = None
                                        else:
                                            logger.info(f"    No original TMDB character names available for '{movie_title_for_calls}', skipping constrained plot with relations.")
                                            working_data_dict["plot_with_character_constraints_and_relations"] = None
                                    else: # raw_chars_data was not available
                                        logger.warning(f"    TMDB raw character data not available for '{movie_title_for_calls}', cannot generate constrained plot with relations.")
                                        working_data_dict["plot_with_character_constraints_and_relations"] = None
                                else: # should_update_field is false
                                    logger.info(f"  Skipping Constrained Plot (with relations) for existing movie '{movie_title_for_calls}' as per update config.")
                            elif active_enrichers_cfg.get('constrained_plot_with_relations') and "plot_with_character_constraints_and_relations" not in working_data_dict:
                                 working_data_dict["plot_with_character_constraints_and_relations"] = None
                        else: # LLM Call 2 failed or returned no output
                            logger.error(f"  Failure: LLM Call 2 (Chars/Rels) for '{movie_title_for_calls}'.")
                    else: # Could not fetch TMDB raw characters
                        logger.error(f"  Failure: Could not fetch TMDB raw characters for '{movie_title_for_calls}'. Skipping Chars/Rels block.")
                else:
                    logger.error(f"  Failure: No TMDB ID for '{movie_title_for_calls}'. Skipping Chars/Rels block.")

        llm3_output_data = None
        if active_enrichers_cfg.get('analytical_data'):
            analytical_fields_to_update = [k for k, v in key_to_enricher_group_map.items() if v == 'analytical_data']
            if not is_new_movie and not update_all_active_fields_for_existing and not any(f in fields_to_update_cfg for f in analytical_fields_to_update):
                logger.info(f"  Skipping Analytical Data regeneration for '{movie_title_for_calls}' (existing, analytical_data fields not targeted).")
            else:
                logger.info(f"  Running: Analytical Data Enrichment for '{movie_title_for_calls}'")
                max_tokens_c3 = words_to_tokens(app_config['max_tokens_analytical_call_words'], app_config['words_to_tokens_ratio'])
                llm3_output_data = analytical_enricher.generate_analytical_data(
                    llm_client_instance, llm_model_id_for_api_calls,
                    movie_title_for_calls, movie_year_for_calls,
                    prompt_call3_template, max_tokens_c3, app_config, logger
                )
                if llm3_output_data:
                    logger.info(f"  Success: Analytical Data for '{movie_title_for_calls}'.")
                    for key, value in llm3_output_data.model_dump(exclude_none=False).items():
                        if should_update_field_local(key):
                            working_data_dict[key] = value
                else:
                    logger.warning(f"  Failure or incomplete: Analytical Data for '{movie_title_for_calls}'.")

        # Ensure analytical fields are None if not generated, for MovieEntry validation
        if not llm3_output_data and active_enrichers_cfg.get('analytical_data'):
            for fld_key in LLMCall3Output.model_fields.keys():
                if fld_key not in working_data_dict:
                    working_data_dict[fld_key] = None

        if active_enrichers_cfg.get('tmdb_review_summary'):
            if should_update_field_local("tmdb_user_review_summary"):
                logger.info(f"  Fetching and Summarizing TMDB User Reviews for '{movie_title_for_calls}'")
                if current_tmdb_id_for_calls: # TMDB ID is required for this
                    tmdb_review_snippets = tmdb_api.fetch_movie_reviews_from_tmdb(
                        TMDB_API_KEY, current_tmdb_id_for_calls, movie_title_for_calls, logger,
                        max_reviews_to_process=app_config.get('max_tmdb_reviews_for_summary', 3),
                        max_review_length_chars=app_config.get('max_tmdb_review_length_chars', 750)
                    )

                    if tmdb_review_snippets:
                        max_tokens_c4_review_summary = words_to_tokens(
                            app_config.get('max_tokens_review_summary_words', 250),
                            app_config['words_to_tokens_ratio']
                        )
                        llm_summary_output = review_summarizer_enricher.generate_tmdb_review_summary(
                            llm_client_instance, llm_model_id_for_api_calls,
                            movie_title_for_calls, movie_year_for_calls,
                            tmdb_review_snippets, prompt_call4_review_summary_template,
                            max_tokens_c4_review_summary, logger
                        )
                        if llm_summary_output and llm_summary_output.tmdb_user_review_summary:
                            working_data_dict["tmdb_user_review_summary"] = llm_summary_output.tmdb_user_review_summary
                            logger.info(f"    Success: Generated TMDB user review summary for '{movie_title_for_calls}'.")
                        else:
                            logger.warning(f"    Could not generate TMDB user review summary for '{movie_title_for_calls}'.")
                            working_data_dict["tmdb_user_review_summary"] = None
                    else:
                        logger.info(f"    No TMDB reviews found or fetched for '{movie_title_for_calls}', skipping summary.")
                        working_data_dict["tmdb_user_review_summary"] = None
                else:
                    logger.warning(f"  Skipping TMDB User Review Summary for '{movie_title_for_calls}' as it's missing a TMDB ID.")
                    working_data_dict["tmdb_user_review_summary"] = None
            else:
                logger.info(f"  Skipping TMDB User Review Summary for existing movie '{movie_title_for_calls}' as per update config.")
        else:
            working_data_dict["tmdb_user_review_summary"] = None


        if active_enrichers_cfg.get('fetch_imdb_ids'):
            relevant_imdb_keys = [k for k, v in key_to_enricher_group_map.items() if v == 'fetch_imdb_ids' or k in ["sequel","prequel","recommendations", "spin_off", "spin_off_of", "remake", "remake_of"]]
            if not is_new_movie and not update_all_active_fields_for_existing and not any(f in fields_to_update_cfg for f in relevant_imdb_keys):
                logger.info(f"  Skipping IMDb ID fetching for '{movie_title_for_calls}' (existing, relevant keys not targeted).")
            else:
                logger.info(f"  Fetching/Updating IMDb IDs for '{movie_title_for_calls}' and its relations.")
                if should_update_field_local("imdb_id"):
                    if current_tmdb_id_for_calls:
                        working_data_dict["imdb_id"] = fetch_master_imdb_id(app_config, logger, current_tmdb_id_for_calls, movie_year_for_calls, True, f"main movie {movie_title_for_calls}")
                    elif movie_title_for_calls:
                        working_data_dict["imdb_id"] = fetch_master_imdb_id(app_config, logger, movie_title_for_calls, movie_year_for_calls, False, f"main movie {movie_title_for_calls}")
                    else:
                        logger.warning(f"  Cannot fetch IMDb ID for main movie due to missing title/TMDB ID for '{movie_title_for_calls}'.")


                for rel_key in ["sequel", "prequel", "spin_off_of", "spin_off", "remake_of", "remake"]:
                    related_movie_val = working_data_dict.get(rel_key)
                    if related_movie_val and should_update_field_local(rel_key):
                        title_to_search = None
                        year_hint_for_related = None

                        if isinstance(related_movie_val, str) and related_movie_val.strip():
                            title_to_search = related_movie_val.strip()
                            working_data_dict[rel_key] = {"title": title_to_search, "imdb_id": None}
                        elif isinstance(related_movie_val, dict) and related_movie_val.get("title"):
                            title_to_search = related_movie_val["title"]

                        if title_to_search:
                            related_imdb_id = fetch_master_imdb_id(app_config, logger, title_to_search, year_hint_for_related, False, f"related {rel_key} {title_to_search}")
                            if isinstance(working_data_dict.get(rel_key), dict):
                                 working_data_dict[rel_key]["imdb_id"] = related_imdb_id
                            else:
                                 logger.warning(f"Could not set IMDb ID for {rel_key} '{title_to_search}' as its structure in working_data_dict is not a dict. Value: {working_data_dict.get(rel_key)}")

                if working_data_dict.get("recommendations") and isinstance(working_data_dict["recommendations"], list) and should_update_field_local("recommendations"):
                    recs_list_of_dicts = working_data_dict["recommendations"]
                    for rec_dict_idx, rec_dict_item in enumerate(recs_list_of_dicts):
                         if isinstance(rec_dict_item, dict) and rec_dict_item.get("title"):
                            rec_title = rec_dict_item["title"]
                            rec_year = str(rec_dict_item.get("year",""))
                            rec_imdb_id = fetch_master_imdb_id(app_config, logger, rec_title, rec_year, False, f"recommendation {rec_title}")
                            working_data_dict["recommendations"][rec_dict_idx]["imdb_id"] = rec_imdb_id

        logger.info(f"  Finalizing entry for '{movie_title_for_calls}'.")
        try:
            final_movie_entry = MovieEntry.model_validate(working_data_dict)
            return final_movie_entry
        except Exception as e:
            logger.error(f"  CRITICAL: Failed to validate final MovieEntry for '{movie_title_for_calls}': {e}")
            logger.debug(f"  Problematic working_data_dict for '{movie_title_for_calls}': {str(working_data_dict)[:1500]}...")
            return None

    # --- MAIN PROCESSING LOGIC BRANCHES ---
    new_movies_added_this_session = 0
    session_api_movie_attempt_count = 0 # This count is primarily for 'fetch_and_add_new' mode

    operation_mode = app_config.get('operation_mode', 'fetch_and_add_new')
    logger.info(f"Operation Mode: '{operation_mode}'")

    if operation_mode == "fetch_and_add_new":
        current_tmdb_page = 1
        update_existing_if_encountered_during_fetch = app_config.get('update_existing_if_encountered_during_fetch', False)
        logger.info(f"Update existing movies if encountered during fetch: {update_existing_if_encountered_during_fetch}")

        while current_tmdb_page <= app_config['max_tmdb_top_rated_pages_to_check']:

            # Break if target for new movies is met and we're not updating existing ones encountered
            if new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session'] and not update_existing_if_encountered_during_fetch:
                logger.info("Target for new movies reached and not updating existing if encountered. Ending TMDB fetch.")
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
            found_processable_movie_on_page = False # Keep this to determine if we should advance page more aggressively

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

                found_processable_movie_on_page = True # This movie candidate is valid enough to proceed

                current_movie_title_lower = tmdb_movie_candidate.title.lower().strip()
                is_existing_movie = current_movie_title_lower in processed_movie_titles_lower_set

                movie_input_for_enrichment: MovieEntry # Declare type hint

                if is_existing_movie:
                    # Check if we should update this existing movie or skip it
                    if not update_existing_if_encountered_during_fetch:
                        logger.debug(f"Movie '{tmdb_movie_candidate.title}' already exists and `update_existing_if_encountered_during_fetch` is false. Skipping.")
                        continue # Skip to the next movie on the TMDB page

                    # If we are updating, find the existing entry to pass to enrichment function
                    existing_movie_entry = next((m for m in all_movie_entries_master_list if m.movie_title.lower().strip() == current_movie_title_lower), None)
                    if not existing_movie_entry:
                        logger.error(f"Consistency Error: Movie '{tmdb_movie_candidate.title}' in processed set but not found in master list. Skipping update.")
                        continue

                    logger.info(f"--- Updating Existing Movie: '{existing_movie_entry.movie_title}' ({existing_movie_entry.movie_year}) TMDB_ID: {existing_movie_entry.tmdb_movie_id or tmdb_movie_candidate.id} ---")
                    movie_input_for_enrichment = existing_movie_entry
                    is_new_movie_for_enrichment = False
                else: # It's a new movie
                    if new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session']:
                        logger.info(f"Target for new movies ({app_config['num_new_movies_to_fetch_this_session']}) reached. Skipping new movie '{tmdb_movie_candidate.title}'.")
                        continue # Skip to the next movie on the TMDB page

                    logger.info(f"--- Processing New Movie: '{tmdb_movie_candidate.title}' ({tmdb_movie_candidate.year}) TMDB_ID: {tmdb_movie_candidate.id} ---")
                    # For a new movie, create a basic MovieEntry to pass as input
                    movie_input_for_enrichment = MovieEntry(movie_title=tmdb_movie_candidate.title, movie_year=tmdb_movie_candidate.year, tmdb_movie_id=tmdb_movie_candidate.id)
                    is_new_movie_for_enrichment = True

                # Pass through the common enrichment logic function
                final_movie_entry = _enrich_and_update_movie_data(
                    movie_entry_input=movie_input_for_enrichment,
                    is_new_movie=is_new_movie_for_enrichment,
                    llm_client_instance=llm_client_instance,
                    llm_model_id_for_api_calls=llm_model_id_for_api_calls,
                    prompt_call1_template=prompt_call1_template,
                    prompt_call2_template=prompt_call2_template,
                    prompt_call3_template=prompt_call3_template,
                    prompt_call4_review_summary_template=prompt_call4_review_summary_template,
                    prompt_constrained_plot_rel_template=prompt_constrained_plot_rel_template,
                    app_config=app_config,
                    TMDB_API_KEY=TMDB_API_KEY,
                    OMDB_API_KEY=OMDB_API_KEY,
                    logger=logger,
                    active_enrichers_cfg=active_enrichers_cfg,
                    update_all_active_fields_for_existing=update_all_active_fields_for_existing,
                    fields_to_update_cfg=fields_to_update_cfg,
                    key_to_enricher_group_map=key_to_enricher_group_map,
                )

                if final_movie_entry: # Only proceed if enrichment was successful
                    if is_existing_movie: # This was an existing movie that we updated
                        idx_to_replace = -1
                        for i, entry in enumerate(all_movie_entries_master_list):
                            if entry.movie_title.lower().strip() == final_movie_entry.movie_title.lower().strip():
                                idx_to_replace = i; break
                        if idx_to_replace != -1:
                            all_movie_entries_master_list[idx_to_replace] = final_movie_entry
                            logger.info(f"  Successfully updated entry for '{final_movie_entry.movie_title}' in list.")
                        else:
                            logger.warning(f"  Consistency Error: Updated movie '{final_movie_entry.movie_title}' was not found for in-place update, appended instead.")
                            all_movie_entries_master_list.append(final_movie_entry)
                    else: # This is a truly new movie being added
                        all_movie_entries_master_list.append(final_movie_entry)
                        processed_movie_titles_lower_set.add(final_movie_entry.movie_title.lower().strip())
                        new_movies_added_this_session += 1

                    save_movie_data_to_yaml(
                        [entry.model_dump(exclude_none=True) for entry in all_movie_entries_master_list],
                        app_config['output_file']
                    )
                    logger.info(f"  Saved '{final_movie_entry.movie_title}' to '{app_config['output_file']}'.")
                else:
                    logger.error(f"  Skipping save for '{tmdb_movie_candidate.title}' due to critical validation failure during enrichment.")
                    if not is_new_movie_for_enrichment: # If it was meant to be an existing movie, mark it as attempted
                        processed_movie_titles_lower_set.add(tmdb_movie_candidate.title.lower().strip())

                time.sleep(app_config.get('api_request_delay_seconds_general', 2))

                # This break logic ensures we stop fetching new movies if the limit is reached,
                # but only if we're not also processing existing movies.
                if new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session'] and not update_existing_if_encountered_during_fetch:
                    logger.info(f"Target for new movies ({app_config['num_new_movies_to_fetch_this_session']}) reached during page processing. And not updating existing. Breaking page loop.")
                    break # Break from movies_on_this_page loop


            # --- End of for loop for movies on current TMDB page ---

            # If target for new movies reached AND we are not updating existing encountered, then break outer page loop too.
            if new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session'] and not update_existing_if_encountered_during_fetch:
                logger.info("Target for new movies reached (outer loop). Ending TMDB page fetching.")
                break

            if not found_processable_movie_on_page and (not update_existing_if_encountered_during_fetch or new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session']):
                logger.info(f"No (more) movies found on TMDB page {current_tmdb_page} to add or update based on config. Advancing page.")

            current_tmdb_page += 1
            if current_tmdb_page > total_tmdb_pages :
                 logger.info(f"Reached reported end of TMDB pages ({total_tmdb_pages}).")
                 break # Reached end of all pages
            time.sleep(app_config.get('api_request_delay_seconds_tmdb_page', 1))
    # --- End of while loop for TMDB pages ---
    elif operation_mode in ["update_by_list", "update_by_range", "update_all_existing"]:
        movies_to_target_for_session: List[MovieEntry] = []
        target_indices_for_update: Optional[Set[int]] = None
        target_specifiers_for_update: Optional[List[Dict[str, Any]]] = None

        if operation_mode == "update_by_range":
            range_str = app_config.get('target_existing_movies_by_index_range', '')
            if not range_str:
                logger.error("Operation mode is 'update_by_range' but 'target_existing_movies_by_index_range' is empty in config. Exiting.")
                return
            target_indices_for_update = parse_index_range_string(range_str, logger)
            logger.info(f"Targeting existing movies by indices: {sorted(list(target_indices_for_update))}")
            if not target_indices_for_update:
                logger.warning(f"No valid indices found from range string '{range_str}'. No movies will be updated.")
                return

            for i, entry in enumerate(all_movie_entries_master_list):
                if i in target_indices_for_update:
                    movies_to_target_for_session.append(entry)

        elif operation_mode == "update_by_list":
            target_specifiers_for_update = app_config.get('target_movies_to_update', [])
            if not target_specifiers_for_update:
                logger.error("Operation mode is 'update_by_list' but 'target_movies_to_update' list is empty in config. Exiting.")
                return
            logger.info(f"Targeting existing movies by specifiers: {target_specifiers_for_update}")

            for target_spec in target_specifiers_for_update:
                found_match_for_spec = False
                for entry in all_movie_entries_master_list:
                    is_title_match = target_spec.get('title') and entry.movie_title.lower() == target_spec['title'].lower()
                    is_year_match = target_spec.get('year') and str(entry.movie_year) == str(target_spec['year'])
                    is_imdb_match = target_spec.get('imdb_id') and entry.imdb_id == target_spec['imdb_id']
                    is_tmdb_match = target_spec.get('tmdb_id') and entry.tmdb_movie_id == target_spec['tmdb_id']

                    if is_imdb_match or is_tmdb_match:
                        movies_to_target_for_session.append(entry)
                        found_match_for_spec = True
                        logger.info(f"  Found exact ID match for target specifier: {target_spec} -> '{entry.movie_title}'")
                        break
                    elif is_title_match and is_year_match:
                        movies_to_target_for_session.append(entry)
                        found_match_for_spec = True
                        logger.info(f"  Found title+year match for target specifier: {target_spec} -> '{entry.movie_title}'")
                        break
                    elif is_title_match:
                        if not target_spec.get('imdb_id') and not target_spec.get('tmdb_id') and not target_spec.get('year'):
                             movies_to_target_for_session.append(entry)
                             found_match_for_spec = True
                             logger.info(f"  Found title-only match for target specifier: {target_spec} -> '{entry.movie_title}'")
                             break
                if not found_match_for_spec:
                    logger.warning(f"  No existing movie found in database matching specifier: {target_spec}")

        elif operation_mode == "update_all_existing":
            movies_to_target_for_session = list(all_movie_entries_master_list)
            logger.info(f"Targeting ALL {len(movies_to_target_for_session)} existing movies for update.")

        if not movies_to_target_for_session:
            logger.info("No movies identified for update based on the specified criteria. Exiting.")
            return

        for movie_entry_to_update in movies_to_target_for_session:
            logger.info(f"--- Updating Targeted Movie: '{movie_entry_to_update.movie_title}' ({movie_entry_to_update.movie_year}) TMDB_ID: {movie_entry_to_update.tmdb_movie_id or 'N/A'} ---")

            final_movie_entry = _enrich_and_update_movie_data(
                movie_entry_input=movie_entry_to_update,
                is_new_movie=False, # Always False for these update modes
                llm_client_instance=llm_client_instance,
                llm_model_id_for_api_calls=llm_model_id_for_api_calls,
                prompt_call1_template=prompt_call1_template,
                prompt_call2_template=prompt_call2_template,
                prompt_call3_template=prompt_call3_template,
                prompt_call4_review_summary_template=prompt_call4_review_summary_template,
                prompt_constrained_plot_rel_template=prompt_constrained_plot_rel_template,
                app_config=app_config,
                TMDB_API_KEY=TMDB_API_KEY,
                OMDB_API_KEY=OMDB_API_KEY,
                logger=logger,
                active_enrichers_cfg=active_enrichers_cfg,
                update_all_active_fields_for_existing=update_all_active_fields_for_existing,
                fields_to_update_cfg=fields_to_update_cfg,
                key_to_enricher_group_map=key_to_enricher_group_map,
            )

            if final_movie_entry:
                idx_to_replace = -1
                for i, entry in enumerate(all_movie_entries_master_list):
                    if entry.tmdb_movie_id == movie_entry_to_update.tmdb_movie_id and \
                       entry.movie_title.lower().strip() == movie_entry_to_update.movie_title.lower().strip():
                        idx_to_replace = i
                        break

                if idx_to_replace != -1:
                    all_movie_entries_master_list[idx_to_replace] = final_movie_entry
                    logger.info(f"  Successfully updated entry for '{final_movie_entry.movie_title}' in master list.")
                else:
                    logger.warning(f"  Could not find original entry for '{final_movie_entry.movie_title}' in master list after update. Appending.")
                    all_movie_entries_master_list.append(final_movie_entry)

                save_movie_data_to_yaml(
                    [entry.model_dump(exclude_none=True) for entry in all_movie_entries_master_list],
                    app_config['output_file']
                )
                logger.info(f"  Saved '{final_movie_entry.movie_title}' to '{app_config['output_file']}'.")
            else:
                logger.error(f"  Skipping save for '{movie_entry_to_update.movie_title}' due to critical validation failure during enrichment.")

            time.sleep(app_config.get('api_request_delay_seconds_general', 2))

    else:
        logger.critical(f"Unsupported operation mode: '{operation_mode}'. Exiting.")
        return

    logger.info(f"===== MOVIE ENRICHMENT SESSION FINISHED =====")
    # The session_api_movie_attempt_count and new_movies_added_this_session are primarily for fetch_and_add_new mode
    logger.info(f"Final total movies in '{app_config['output_file']}': {len(all_movie_entries_master_list)}")
    logger.info(f"Raw log: '{app_config['raw_log_file']}'")
    logger.info(f"Clean YAML data: '{app_config['output_file']}'")
    if active_enrichers_cfg.get('fetch_character_images'):
        logger.info(f"Character images saved to: '{app_config['character_image_save_path']}'")

if __name__ == "__main__":
    run_enrichment_pipeline()