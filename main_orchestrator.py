# movie_enrichment_project/main_orchestrator.py
import os
import time
import yaml
from dotenv import load_dotenv
import openai # For the client
from typing import Optional, Dict, Any, List, Set, Union

# Project local imports
from utils.helpers import (
    load_full_movie_data_from_yaml, save_movie_data_to_yaml, parse_index_range_string,
    words_to_tokens, setup_logging
)
from models.movie_models import (
    MovieEntry, LLMCall1Output, LLMCall2Output, LLMCall3Output,
    TMDBMovieResult, TMDBRawCharacter, RelatedMovie, Recommendation,
    CharacterListItem, Relationship
)
from data_providers import tmdb_api, omdb_api, llm_clients
from enrichers import movie_data_enricher, character_enricher, analytical_enricher, review_summarizer_enricher, constrained_plot_rel_enricher
from utils import image_downloader

# Load environment variables from .env file
load_dotenv()

# --- Global API Keys (loaded once) ---
OMDB_API_KEY_GLOBAL = os.getenv("OMDB_API_KEY") # Renamed to avoid conflict in function signatures
TMDB_API_KEY_GLOBAL = os.getenv("TMDB_API_KEY") # Renamed to avoid conflict

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
    # app_config_for_api_keys: Dict[str, Any], # Not used currently as keys are global
    logger: Any,
    title_or_tmdb_id: Any,
    year_hint: Optional[str] = None,
    is_tmdb_id: bool = False,
    object_type_for_log: str = "movie",
    tmdb_api_key_for_fetch: Optional[str] = None, # Allow passing for specific calls
    omdb_api_key_for_fetch: Optional[str] = None  # Allow passing for specific calls
) -> Optional[str]:
    log_prefix = f"IMDbFetch ({object_type_for_log} '{str(title_or_tmdb_id)[:30]}'):"
    imdb_id: Optional[str] = None

    # Use passed keys if available, otherwise fallback to global
    effective_tmdb_key = tmdb_api_key_for_fetch if tmdb_api_key_for_fetch else TMDB_API_KEY_GLOBAL
    effective_omdb_key = omdb_api_key_for_fetch if omdb_api_key_for_fetch else OMDB_API_KEY_GLOBAL

    if not effective_tmdb_key and not effective_omdb_key:
        logger.warning(f"{log_prefix} Both TMDB and OMDB API keys missing for fetch. Cannot fetch IMDb ID.")
        return None

    title_str: str = ""
    tmdb_id_int: Optional[int] = None

    if is_tmdb_id and title_or_tmdb_id:
        try:
            tmdb_id_int = int(title_or_tmdb_id)
            if effective_tmdb_key:
                logger.debug(f"{log_prefix} Attempting IMDb ID from TMDB details using TMDB ID {tmdb_id_int}.")
                imdb_id = tmdb_api.get_imdb_id_from_tmdb_details(effective_tmdb_key, tmdb_id_int, str(title_or_tmdb_id), logger)
                if imdb_id: return imdb_id
        except ValueError:
            logger.warning(f"{log_prefix} Provided TMDB ID '{title_or_tmdb_id}' is not an int. Treating as title.")
            title_str = str(title_or_tmdb_id)
            is_tmdb_id = False
    else:
        title_str = str(title_or_tmdb_id)

    valid_year_for_api: Optional[str] = None
    if year_hint:
        year_str_temp = str(year_hint).strip()
        if year_str_temp.isdigit() and len(year_str_temp) == 4:
            valid_year_for_api = year_str_temp
        elif logger:
            logger.debug(f"{log_prefix} Invalid year_hint format '{year_hint}'. Ignoring for API calls.")

    if effective_omdb_key and title_str and valid_year_for_api:
        logger.debug(f"{log_prefix} Attempting OMDB with title '{title_str}' and year '{valid_year_for_api}'.")
        imdb_id = omdb_api.get_imdb_id_from_omdb(effective_omdb_key, title_str, valid_year_for_api, logger)
        if imdb_id: return imdb_id

    if effective_tmdb_key and title_str and valid_year_for_api:
        logger.debug(f"{log_prefix} Attempting TMDB search for '{title_str}' year '{valid_year_for_api}', then details.")
        tmdb_id_from_search, _ = tmdb_api.search_tmdb_for_movie_id(effective_tmdb_key, title_str, valid_year_for_api, logger)
        if tmdb_id_from_search:
            imdb_id = tmdb_api.get_imdb_id_from_tmdb_details(effective_tmdb_key, tmdb_id_from_search, title_str, logger)
            if imdb_id: return imdb_id

    if effective_omdb_key and title_str:
        logger.debug(f"{log_prefix} Attempting OMDB with title '{title_str}' only.")
        imdb_id = omdb_api.get_imdb_id_from_omdb(effective_omdb_key, title_str, None, logger)
        if imdb_id: return imdb_id

    if effective_tmdb_key and title_str:
        logger.debug(f"{log_prefix} Attempting TMDB search for '{title_str}' only, then details.")
        tmdb_id_from_search_title_only, _ = tmdb_api.search_tmdb_for_movie_id(effective_tmdb_key, title_str, None, logger)
        if tmdb_id_from_search_title_only:
            imdb_id = tmdb_api.get_imdb_id_from_tmdb_details(effective_tmdb_key, tmdb_id_from_search_title_only, title_str, logger)
            if imdb_id: return imdb_id

    if not imdb_id:
        logger.info(f"{log_prefix} Failed to find IMDb ID for '{str(title_or_tmdb_id)}' (Year hint: {year_hint or 'N/A'}) after all attempts.")
    return None


# --- Main Orchestration Function ---
def run_enrichment_pipeline():
    app_config = load_app_config()
    logger = setup_logging(app_config['raw_log_file'], logger_name="MovieEnrichmentPipeline")
    llm_providers_config = load_llm_providers_config(logger=logger)

    active_provider_id = app_config.get("active_llm_provider_id")
    if not active_provider_id or active_provider_id not in llm_providers_config:
        logger.critical(f"Active LLM provider ID '{active_provider_id}' not specified or found. Exiting.")
        return

    active_llm_config = llm_providers_config[active_provider_id]
    llm_model_id_for_api_calls_param = active_llm_config.get("model_id") # Parameter for function calls

    logger.info(f"===== MOVIE ENRICHMENT SESSION STARTED =====")
    logger.info(f"Using LLM Provider: {active_llm_config.get('description', active_provider_id)} (ID: {active_provider_id})")
    logger.info(f"LLM Model ID for API calls: {llm_model_id_for_api_calls_param}")
    logger.info(f"Active enrichers: {app_config.get('active_enrichers', {})}")

    if not TMDB_API_KEY_GLOBAL: logger.critical("TMDB_API_KEY (global) not set. Exiting."); return
    if not OMDB_API_KEY_GLOBAL: logger.warning("OMDB_API_KEY (global) not set. IMDb ID lookups limited.")
    if not llm_model_id_for_api_calls_param: logger.critical(f"No 'model_id' for LLM provider '{active_provider_id}'. Exiting."); return

    llm_client_instance_param = None # Parameter for function calls
    if active_llm_config.get("type", "openai_compatible") == "openai_compatible":
        api_key_env_var = active_llm_config.get("api_key_env_var")
        llm_api_key_val = os.getenv(api_key_env_var) if api_key_env_var else None
        base_url_val = active_llm_config.get("base_url")
        if not base_url_val and "openai_" in active_provider_id: base_url_val = None
        elif not base_url_val : logger.critical(f"LLM_BASE_URL not configured for non-official OpenAI provider. Exiting."); return
        if not llm_api_key_val and api_key_env_var: logger.warning(f"API key env var '{api_key_env_var}' not set.")
        try:
            llm_client_instance_param = openai.OpenAI(base_url=base_url_val, api_key=llm_api_key_val)
            logger.info(f"OpenAI-compatible LLM client initialized. Provider: {active_provider_id}, Base URL: {base_url_val or 'OpenAI Default'}")
        except Exception as e: logger.critical(f"Failed to init LLM client: {e}. Exiting."); return
    else: logger.critical(f"Unsupported LLM provider type. Exiting."); return

    if not os.path.exists(app_config['character_image_save_path']):
        try: os.makedirs(app_config['character_image_save_path'], exist_ok=True); logger.info(f"Created image dir: {app_config['character_image_save_path']}")
        except OSError as e: logger.error(f"Could not create image dir: {e}.")

    all_movie_entries_master_list: List[MovieEntry] = []
    raw_data_from_file = load_full_movie_data_from_yaml(app_config['output_file'])
    for item_dict in raw_data_from_file:
        try: all_movie_entries_master_list.append(MovieEntry.model_validate(item_dict))
        except Exception as e: logger.warning(f"Invalid existing movie data '{item_dict.get('movie_title', 'Unknown')}': {e}. Skipping.")
    logger.info(f"Loaded {len(all_movie_entries_master_list)} valid movie entries from '{app_config['output_file']}'.")

    processed_movie_titles_lower_set = {entry.movie_title.lower().strip() for entry in all_movie_entries_master_list}

    prompt_call1_template_param = load_prompt_template(app_config["prompts"]["call1_initial_data"], logger)
    prompt_call2_template_param = load_prompt_template(app_config["prompts"]["call2_chars_rels"], logger)
    prompt_call3_template_param = load_prompt_template(app_config["prompts"]["call3_analytical"], logger)
    prompt_call4_review_summary_template_param = load_prompt_template(app_config["prompts"]["call4_tmdb_review_summary"], logger)
    prompt_constrained_plot_rel_template_param = load_prompt_template(app_config["prompts"]["call_constrained_plot_relations"], logger)

    active_enrichers_cfg = app_config.get('active_enrichers', {})
    fields_to_update_cfg = app_config.get('fields_to_update', [])
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
        movie_data_input: Union[MovieEntry, Dict[str, Any]],
        is_new_movie: bool,
        # Parameters passed from the main orchestrator scope
        llm_client: openai.OpenAI,
        llm_model_id: str,
        prompt_c1_template: str,
        prompt_c2_template: str,
        prompt_c3_template: str,
        prompt_c4_template: str,
        prompt_plot_rel_template: str,
        current_app_config: Dict[str, Any], # Use a distinct name
        passed_tmdb_api_key: str, # Use a distinct name
        passed_omdb_api_key: Optional[str], # Use a distinct name
        logger_instance: Any, # Use a distinct name
        current_active_enrichers_cfg: Dict[str, Any], # Use a distinct name
        current_update_all_active_fields: bool, # Use a distinct name
        current_fields_to_update_cfg: List[str], # Use a distinct name
        current_key_to_enricher_group_map: Dict[str, str], # Use a distinct name
    ) -> Optional[MovieEntry]:

        if isinstance(movie_data_input, MovieEntry):
            movie_title_for_calls = movie_data_input.movie_title
            movie_year_for_calls = movie_data_input.movie_year
            current_tmdb_id_for_calls = movie_data_input.tmdb_movie_id
        else:
            movie_title_for_calls = movie_data_input.get("movie_title", "")
            movie_year_for_calls = movie_data_input.get("movie_year", "")
            current_tmdb_id_for_calls = movie_data_input.get("tmdb_movie_id")

        if is_new_movie:
            working_data_dict = {
                "movie_title": movie_title_for_calls, "movie_year": movie_year_for_calls, "tmdb_movie_id": current_tmdb_id_for_calls,
                **{field: None for field in MovieEntry.model_fields if field not in ["movie_title", "movie_year", "tmdb_movie_id"]}
            }
            for field_name, field_info in MovieEntry.model_fields.items():
                is_list_field = getattr(field_info.annotation, '__origin__', None) is list
                is_optional_list_field = (getattr(field_info.annotation, '__origin__', None) is Union and
                                          any(getattr(arg, '__origin__', None) is list for arg in getattr(field_info.annotation, '__args__', [])))
                if (is_list_field or is_optional_list_field) and working_data_dict.get(field_name) is None :
                    working_data_dict[field_name] = []
        else:
            working_data_dict = movie_data_input.model_dump(exclude_none=False)

        def should_update_field_local(field_name: str) -> bool:
            if is_new_movie: return True
            if current_update_all_active_fields: return True
            return field_name in current_fields_to_update_cfg

        if current_active_enrichers_cfg.get('initial_data'):
            initial_data_fields_to_update = [k for k, v in current_key_to_enricher_group_map.items() if v == 'initial_data']
            if not is_new_movie and not current_update_all_active_fields and not any(f in current_fields_to_update_cfg for f in initial_data_fields_to_update):
                logger_instance.info(f"  Skipping Initial Data for '{movie_title_for_calls}'.")
            else:
                logger_instance.info(f"  Running: Initial Data for '{movie_title_for_calls}'")
                max_tokens_c1 = words_to_tokens(current_app_config['max_tokens_call_1_words'], current_app_config['words_to_tokens_ratio'])
                llm1_data_generated = movie_data_enricher.generate_initial_movie_data(
                    llm_client, llm_model_id, movie_title_for_calls, movie_year_for_calls,
                    prompt_c1_template, max_tokens_c1, current_app_config, logger_instance
                )
                if llm1_data_generated:
                    logger_instance.info(f"  Success: Initial Data for '{movie_title_for_calls}'.")
                    for key, value in llm1_data_generated.model_dump(exclude={"movie_title", "movie_year"}, exclude_none=False).items():
                        if should_update_field_local(key):
                            if key in ["sequel", "prequel", "spin_off_of", "spin_off", "remake_of", "remake"]:
                                if isinstance(value, str) and value.strip(): working_data_dict[key] = RelatedMovie(title=value.strip()).model_dump()
                                elif isinstance(value, dict) and value.get("title"):
                                    try: working_data_dict[key] = RelatedMovie.model_validate(value).model_dump()
                                    except Exception: working_data_dict[key] = RelatedMovie(title=str(value.get("title","Unknown"))).model_dump()
                                else: working_data_dict[key] = None
                            else: working_data_dict[key] = value
                else: logger_instance.error(f"  Failure: Initial Data for '{movie_title_for_calls}'.")

        raw_chars_data: Optional[List[TMDBRawCharacter]] = None
        deduplicated_relationships_models: List[Relationship] = []

        if current_active_enrichers_cfg.get('characters_and_relations'):
            char_rel_fields_to_update = [k for k,v in current_key_to_enricher_group_map.items() if v in ['characters_and_relations', 'constrained_plot_with_relations']]
            if not is_new_movie and not current_update_all_active_fields and not any(f in current_fields_to_update_cfg for f in char_rel_fields_to_update):
                logger_instance.info(f"  Skipping Chars/Rels for '{movie_title_for_calls}'.")
            else:
                logger_instance.info(f"  Running: Chars/Rels for '{movie_title_for_calls}'")
                if current_tmdb_id_for_calls:
                    raw_chars_data = tmdb_api.fetch_raw_character_actor_list_from_tmdb(
                        passed_tmdb_api_key, current_tmdb_id_for_calls, movie_title_for_calls,
                        current_app_config['max_characters_from_tmdb'], logger_instance
                    )
                    if raw_chars_data:
                        raw_chars_yaml_for_prompt = yaml.dump([char.model_dump() for char in raw_chars_data], sort_keys=False, allow_unicode=True, indent=2)
                        num_chars = len(raw_chars_data)
                        dynamic_words_c2 = current_app_config['max_tokens_enrich_rel_call_base_words'] + \
                                           (num_chars * current_app_config['max_tokens_enrich_rel_char_desc_words']) + \
                                           (num_chars * current_app_config['max_tokens_enrich_rel_char_rels_words'])
                        max_tokens_c2 = words_to_tokens(dynamic_words_c2, current_app_config['words_to_tokens_ratio'])
                        llm2_output = character_enricher.enrich_characters_and_get_relationships(
                            llm_client, llm_model_id, movie_title_for_calls, movie_year_for_calls,
                            raw_chars_yaml_for_prompt, prompt_c2_template, max_tokens_c2, current_app_config, logger_instance
                        )
                        if llm2_output:
                            logger_instance.info(f"  Success: LLM Call 2 for '{movie_title_for_calls}'.")
                            temp_char_list_models = llm2_output.character_list
                            if current_active_enrichers_cfg.get('fetch_character_images'):
                                logger_instance.info(f"    Triggering character image downloads for '{movie_title_for_calls}'...")
                                character_enricher.trigger_character_image_downloads(
                                    character_list_from_llm=temp_char_list_models,
                                    movie_title=movie_title_for_calls,
                                    movie_tmdb_id=current_tmdb_id_for_calls,
                                    save_path_base=current_app_config['character_image_save_path'],
                                    tmdb_api_key=passed_tmdb_api_key,
                                    tmdb_image_base_url=current_app_config['tmdb_image_base_url'],
                                    tmdb_image_size=current_app_config['tmdb_image_size'],
                                    ddg_num_images_per_search=current_app_config.get('ddg_num_images_per_character_search', 1),
                                    ddg_sleep_after_character_group=current_app_config.get('ddg_sleep_after_character_image_group', 1.0),
                                    ddg_sleep_between_individual_downloads=current_app_config.get('ddg_sleep_between_individual_image_downloads', 0.5),
                                    logger=logger_instance
                                )
                            if should_update_field_local("character_list"):
                                working_data_dict["character_list"] = [char.model_dump() for char in temp_char_list_models]

                            deduplicated_relationships_models = character_enricher.deduplicate_and_normalize_relationships(
                                temp_char_list_models, llm2_output.relationships or [], logger_instance
                            )
                            if should_update_field_local("relationships"):
                                working_data_dict["relationships"] = [rel.model_dump() for rel in deduplicated_relationships_models]

                            if current_active_enrichers_cfg.get('fetch_relationship_images') and deduplicated_relationships_models:
                                logger_instance.info(f"    Triggering relationship image downloads for '{movie_title_for_calls}'...")
                                character_enricher.trigger_relationship_image_downloads(
                                    relationships=deduplicated_relationships_models,
                                    movie_title=movie_title_for_calls,
                                    save_path_base=current_app_config['character_image_save_path'],
                                    ddg_num_images_per_relationship_search=current_app_config.get('ddg_num_images_per_relationship_search', 1),
                                    max_relationships_to_process=current_app_config.get('max_relationships_for_image_download', 10),
                                    ddg_sleep_after_relationship_group=current_app_config.get('ddg_sleep_after_relationship_image_group', 1.5),
                                    ddg_sleep_between_individual_downloads=current_app_config.get('ddg_sleep_between_individual_image_downloads', 0.5),
                                    logger=logger_instance
                                )

                            if current_active_enrichers_cfg.get('constrained_plot_with_relations'):
                                if should_update_field_local("plot_with_character_constraints_and_relations"):
                                    if raw_chars_data:
                                        tmdb_original_char_names = [char.tmdb_character_name for char in raw_chars_data if char.tmdb_character_name]
                                        relationships_for_context = deduplicated_relationships_models
                                        if tmdb_original_char_names:
                                            logger_instance.info(f"  Generating Constrained Plot for '{movie_title_for_calls}'.")
                                            max_tokens_plot_rel = words_to_tokens(current_app_config.get('max_tokens_constrained_plot_relations_words', 350), current_app_config['words_to_tokens_ratio'])
                                            plot_rel_output = constrained_plot_rel_enricher.generate_constrained_plot_with_relations(
                                                llm_client, llm_model_id, movie_title_for_calls, movie_year_for_calls,
                                                tmdb_original_char_names, relationships_for_context,
                                                prompt_plot_rel_template, max_tokens_plot_rel, logger_instance
                                            )
                                            if plot_rel_output and plot_rel_output.plot_with_character_constraints_and_relations:
                                                working_data_dict["plot_with_character_constraints_and_relations"] = plot_rel_output.plot_with_character_constraints_and_relations
                                                logger_instance.info(f"    Success: Constrained plot for '{movie_title_for_calls}'.")
                                            else:
                                                logger_instance.warning(f"    Could not generate constrained plot for '{movie_title_for_calls}'.")
                                                working_data_dict["plot_with_character_constraints_and_relations"] = None
                                        else: working_data_dict["plot_with_character_constraints_and_relations"] = None
                                    else: working_data_dict["plot_with_character_constraints_and_relations"] = None
                                else: logger_instance.info(f"  Skipping Constrained Plot update for '{movie_title_for_calls}'.")
                            elif current_active_enrichers_cfg.get('constrained_plot_with_relations') and "plot_with_character_constraints_and_relations" not in working_data_dict:
                                 working_data_dict["plot_with_character_constraints_and_relations"] = None
                        else: logger_instance.error(f"  Failure: LLM Call 2 for '{movie_title_for_calls}'.")
                    else: logger_instance.error(f"  Failure: Could not fetch TMDB raw chars for '{movie_title_for_calls}'.")
                else: logger_instance.error(f"  Failure: No TMDB ID for '{movie_title_for_calls}'.")

        if current_active_enrichers_cfg.get('analytical_data'):
            analytical_fields_to_update = [k for k,v in current_key_to_enricher_group_map.items() if v == 'analytical_data']
            if not is_new_movie and not current_update_all_active_fields and not any(f in current_fields_to_update_cfg for f in analytical_fields_to_update):
                logger_instance.info(f"  Skipping Analytical Data for '{movie_title_for_calls}'.")
            else:
                logger_instance.info(f"  Running: Analytical Data for '{movie_title_for_calls}'")
                max_tokens_c3 = words_to_tokens(current_app_config['max_tokens_analytical_call_words'], current_app_config['words_to_tokens_ratio'])
                llm3_output_data = analytical_enricher.generate_analytical_data(
                    llm_client, llm_model_id, movie_title_for_calls, movie_year_for_calls,
                    prompt_c3_template, max_tokens_c3, current_app_config, logger_instance
                )
                if llm3_output_data:
                    logger_instance.info(f"  Success: Analytical Data for '{movie_title_for_calls}'.")
                    for key, value in llm3_output_data.model_dump(exclude_none=False).items():
                        if should_update_field_local(key): working_data_dict[key] = value
                else:
                    logger_instance.warning(f"  Failure: Analytical Data for '{movie_title_for_calls}'.")
                    for fld_key in LLMCall3Output.model_fields.keys():
                        if should_update_field_local(fld_key):
                             working_data_dict[fld_key] = None

        if current_active_enrichers_cfg.get('tmdb_review_summary'):
            if should_update_field_local("tmdb_user_review_summary"):
                logger_instance.info(f"  Running: TMDB Review Summary for '{movie_title_for_calls}'")
                if current_tmdb_id_for_calls:
                    tmdb_review_snippets = tmdb_api.fetch_movie_reviews_from_tmdb(
                        passed_tmdb_api_key, current_tmdb_id_for_calls, movie_title_for_calls, logger_instance,
                        max_reviews_to_process=current_app_config.get('max_tmdb_reviews_for_summary', 3),
                        max_review_length_chars=current_app_config.get('max_tmdb_review_length_chars', 750)
                    )
                    if tmdb_review_snippets:
                        max_tokens_c4_review_summary = words_to_tokens(current_app_config.get('max_tokens_review_summary_words', 250), current_app_config['words_to_tokens_ratio'])
                        llm_summary_output = review_summarizer_enricher.generate_tmdb_review_summary(
                            llm_client, llm_model_id, movie_title_for_calls, movie_year_for_calls,
                            tmdb_review_snippets, prompt_c4_template, max_tokens_c4_review_summary, logger_instance
                        )
                        if llm_summary_output and llm_summary_output.tmdb_user_review_summary:
                            working_data_dict["tmdb_user_review_summary"] = llm_summary_output.tmdb_user_review_summary
                            logger_instance.info(f"    Success: Review summary for '{movie_title_for_calls}'.")
                        else: working_data_dict["tmdb_user_review_summary"] = None; logger_instance.warning(f"    Failure: Review summary for '{movie_title_for_calls}'.")
                    else: working_data_dict["tmdb_user_review_summary"] = None; logger_instance.info(f"    No reviews for '{movie_title_for_calls}'.")
                else: working_data_dict["tmdb_user_review_summary"] = None; logger_instance.warning(f"    No TMDB ID for review summary '{movie_title_for_calls}'.")
            else: logger_instance.info(f"  Skipping Review Summary update for '{movie_title_for_calls}'.")
        elif "tmdb_user_review_summary" not in working_data_dict: working_data_dict["tmdb_user_review_summary"] = None

        if current_active_enrichers_cfg.get('fetch_imdb_ids'):
            relevant_imdb_keys = [k for k,v in current_key_to_enricher_group_map.items() if v == 'fetch_imdb_ids' or k in ["sequel","prequel","recommendations", "spin_off", "spin_off_of", "remake", "remake_of"]]
            if not is_new_movie and not current_update_all_active_fields and not any(f in current_fields_to_update_cfg for f in relevant_imdb_keys):
                logger_instance.info(f"  Skipping IMDb ID fetching for '{movie_title_for_calls}'.")
            else:
                logger_instance.info(f"  Fetching IMDb IDs for '{movie_title_for_calls}'.")
                if should_update_field_local("imdb_id") and working_data_dict.get("imdb_id") is None:
                    id_to_search = current_tmdb_id_for_calls if current_tmdb_id_for_calls else movie_title_for_calls
                    is_tmdb = bool(current_tmdb_id_for_calls)
                    working_data_dict["imdb_id"] = fetch_master_imdb_id(
                        logger_instance, id_to_search, movie_year_for_calls, is_tmdb, f"main movie {movie_title_for_calls}",
                        tmdb_api_key_for_fetch=passed_tmdb_api_key, omdb_api_key_for_fetch=passed_omdb_api_key
                    )

                for rel_key in ["sequel", "prequel", "spin_off_of", "spin_off", "remake_of", "remake"]:
                    related_movie_val = working_data_dict.get(rel_key)
                    if isinstance(related_movie_val, dict) and should_update_field_local(rel_key) and related_movie_val.get("title") and related_movie_val.get("imdb_id") is None:
                        related_imdb_id = fetch_master_imdb_id(
                            logger_instance, related_movie_val["title"], None, False, f"related {rel_key}",
                            tmdb_api_key_for_fetch=passed_tmdb_api_key, omdb_api_key_for_fetch=passed_omdb_api_key
                        )
                        working_data_dict[rel_key]["imdb_id"] = related_imdb_id

                if isinstance(working_data_dict.get("recommendations"), list) and should_update_field_local("recommendations"):
                    for rec_idx, rec_dict in enumerate(working_data_dict["recommendations"]):
                        if isinstance(rec_dict, dict) and rec_dict.get("title") and rec_dict.get("imdb_id") is None:
                            rec_year = str(rec_dict.get("year","")) if rec_dict.get("year") else None
                            rec_imdb_id = fetch_master_imdb_id(
                                logger_instance, rec_dict["title"], rec_year, False, f"recommendation {rec_dict['title']}",
                                tmdb_api_key_for_fetch=passed_tmdb_api_key, omdb_api_key_for_fetch=passed_omdb_api_key
                            )
                            working_data_dict["recommendations"][rec_idx]["imdb_id"] = rec_imdb_id
        elif "imdb_id" not in working_data_dict: working_data_dict["imdb_id"] = None

        logger_instance.info(f"  Finalizing entry for '{movie_title_for_calls}'.")
        try:
            for field_name, field_info in MovieEntry.model_fields.items():
                if field_name not in working_data_dict:
                    is_list_field = getattr(field_info.annotation, '__origin__', None) is list
                    is_optional_list_field = (getattr(field_info.annotation, '__origin__', None) is Union and
                                              any(getattr(arg, '__origin__', None) is list for arg in getattr(field_info.annotation, '__args__', [])))
                    working_data_dict[field_name] = [] if is_list_field or is_optional_list_field else None

            final_movie_entry = MovieEntry.model_validate(working_data_dict)
            return final_movie_entry
        except Exception as e:
            logger_instance.error(f"  CRITICAL: Failed to validate final MovieEntry for '{movie_title_for_calls}': {e}")
            logger_instance.debug(f"  Problematic working_data_dict: {str(working_data_dict)[:1500]}...")
            return None
    # END OF _enrich_and_update_movie_data

    # --- MAIN PROCESSING LOGIC BRANCHES ---
    new_movies_added_this_session = 0
    session_api_movie_attempt_count = 0

    operation_mode = app_config.get('operation_mode', 'fetch_and_add_new')
    logger.info(f"Operation Mode: '{operation_mode}'")

    if operation_mode == "fetch_and_add_new":
        current_tmdb_page = 1
        update_existing_if_encountered_during_fetch = app_config.get('update_existing_if_encountered_during_fetch', False)
        logger.info(f"Update existing movies if encountered during fetch: {update_existing_if_encountered_during_fetch}")

        while current_tmdb_page <= app_config['max_tmdb_top_rated_pages_to_check']:
            if new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session'] and not update_existing_if_encountered_during_fetch:
                logger.info("Target for new movies reached and not updating existing. Ending TMDB fetch.")
                break

            logger.info(f"--- Fetching TMDB Top Rated Page: {current_tmdb_page} ---")
            tmdb_page_data_raw = tmdb_api.fetch_top_rated_movies_from_tmdb(TMDB_API_KEY_GLOBAL, current_tmdb_page, logger)

            if not tmdb_page_data_raw or not tmdb_page_data_raw.get("results"):
                logger.warning(f"No results on TMDB Page {current_tmdb_page}.")
                if not tmdb_page_data_raw or ("total_pages" in tmdb_page_data_raw and current_tmdb_page >= tmdb_page_data_raw.get("total_pages", current_tmdb_page)):
                    logger.info("Reached end of TMDB pages or fetch error limit.")
                    break
                current_tmdb_page += 1; time.sleep(app_config.get('api_request_delay_seconds_tmdb_page', 1)); continue

            movies_on_this_page_raw = tmdb_page_data_raw["results"]
            total_tmdb_pages = tmdb_page_data_raw.get("total_pages", current_tmdb_page)
            found_processable_movie_on_page = False

            for tmdb_movie_raw_dict in movies_on_this_page_raw:
                try: tmdb_movie_candidate = TMDBMovieResult.model_validate(tmdb_movie_raw_dict)
                except Exception as e: logger.warning(f"Skipping TMDB entry validation error: {e}"); continue
                if not tmdb_movie_candidate.title or tmdb_movie_candidate.id is None or not tmdb_movie_candidate.year:
                    logger.info(f"Skipping TMDB entry missing core info: '{tmdb_movie_candidate.title}'"); continue
                found_processable_movie_on_page = True
                current_movie_title_lower = tmdb_movie_candidate.title.lower().strip()
                is_existing_movie = current_movie_title_lower in processed_movie_titles_lower_set

                movie_input_for_enrichment: Union[MovieEntry, Dict[str, Any]]
                is_new_movie_for_enrichment: bool

                if is_existing_movie:
                    if not update_existing_if_encountered_during_fetch:
                        logger.debug(f"Movie '{tmdb_movie_candidate.title}' exists, skipping update."); continue
                    existing_movie_entry = next((m for m in all_movie_entries_master_list if m.movie_title.lower().strip() == current_movie_title_lower), None)
                    if not existing_movie_entry: logger.error(f"Consistency Error: '{tmdb_movie_candidate.title}' in set but not list. Skipping."); continue
                    logger.info(f"--- Updating Existing Movie: '{existing_movie_entry.movie_title}' ---")
                    movie_input_for_enrichment = existing_movie_entry
                    is_new_movie_for_enrichment = False
                else:
                    if new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session']:
                        logger.info(f"Target for new movies reached. Skipping '{tmdb_movie_candidate.title}'."); continue
                    logger.info(f"--- Processing New Movie: '{tmdb_movie_candidate.title}' ({tmdb_movie_candidate.year}) TMDB_ID: {tmdb_movie_candidate.id} ---")
                    movie_input_for_enrichment = {"movie_title": tmdb_movie_candidate.title, "movie_year": tmdb_movie_candidate.year, "tmdb_movie_id": tmdb_movie_candidate.id}
                    is_new_movie_for_enrichment = True

                final_movie_entry = _enrich_and_update_movie_data(
                    movie_data_input=movie_input_for_enrichment,
                    is_new_movie=is_new_movie_for_enrichment,
                    llm_client=llm_client_instance_param,
                    llm_model_id=llm_model_id_for_api_calls_param,
                    prompt_c1_template=prompt_call1_template_param,
                    prompt_c2_template=prompt_call2_template_param,
                    prompt_c3_template=prompt_call3_template_param,
                    prompt_c4_template=prompt_call4_review_summary_template_param,
                    prompt_plot_rel_template=prompt_constrained_plot_rel_template_param,
                    current_app_config=app_config,
                    passed_tmdb_api_key=TMDB_API_KEY_GLOBAL,
                    passed_omdb_api_key=OMDB_API_KEY_GLOBAL,
                    logger_instance=logger,
                    current_active_enrichers_cfg=active_enrichers_cfg,
                    current_update_all_active_fields=update_all_active_fields_for_existing,
                    current_fields_to_update_cfg=fields_to_update_cfg,
                    current_key_to_enricher_group_map=key_to_enricher_group_map,
                )

                if final_movie_entry:
                    if is_existing_movie:
                        idx_to_replace = next((i for i, entry in enumerate(all_movie_entries_master_list) if entry.movie_title.lower().strip() == final_movie_entry.movie_title.lower().strip()), -1)
                        if idx_to_replace != -1: all_movie_entries_master_list[idx_to_replace] = final_movie_entry; logger.info(f"  Updated '{final_movie_entry.movie_title}'.")
                        else: all_movie_entries_master_list.append(final_movie_entry); logger.warning(f"  Appended updated '{final_movie_entry.movie_title}'.")
                    else:
                        all_movie_entries_master_list.append(final_movie_entry)
                        processed_movie_titles_lower_set.add(final_movie_entry.movie_title.lower().strip())
                        new_movies_added_this_session += 1
                    save_movie_data_to_yaml([entry.model_dump(exclude_none=True) for entry in all_movie_entries_master_list], app_config['output_file'])
                    logger.info(f"  Saved '{final_movie_entry.movie_title}' to '{app_config['output_file']}'.")
                else:
                    logger.error(f"  Skipping save for '{tmdb_movie_candidate.title}' due to enrichment failure.")
                    if is_new_movie_for_enrichment: processed_movie_titles_lower_set.add(tmdb_movie_candidate.title.lower().strip())

                time.sleep(app_config.get('api_request_delay_seconds_general', 2))
                if new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session'] and not update_existing_if_encountered_during_fetch:
                    logger.info(f"Target for new movies reached. Breaking page loop."); break

            if new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session'] and not update_existing_if_encountered_during_fetch:
                logger.info("Target for new movies reached. Ending TMDB page fetching."); break
            if not found_processable_movie_on_page and (not update_existing_if_encountered_during_fetch or new_movies_added_this_session >= app_config['num_new_movies_to_fetch_this_session']):
                logger.info(f"No more processable movies on page {current_tmdb_page}. Advancing.")

            current_tmdb_page += 1
            if current_tmdb_page > total_tmdb_pages: logger.info(f"Reached end of TMDB pages ({total_tmdb_pages})."); break
            time.sleep(app_config.get('api_request_delay_seconds_tmdb_page', 1))

    elif operation_mode in ["update_by_list", "update_by_range", "update_all_existing"]:
        movies_to_target_for_session: List[MovieEntry] = []
        if operation_mode == "update_by_range":
            range_str = app_config.get('target_existing_movies_by_index_range', '')
            if not range_str: logger.error("Range string empty for 'update_by_range'. Exiting."); return
            target_indices = parse_index_range_string(range_str, logger)
            if not target_indices: logger.warning(f"No valid indices from '{range_str}'. Exiting."); return
            logger.info(f"Targeting indices: {sorted(list(target_indices))}")
            movies_to_target_for_session = [entry for i, entry in enumerate(all_movie_entries_master_list) if i in target_indices]
        elif operation_mode == "update_by_list":
            target_specifiers = app_config.get('target_movies_to_update', [])
            if not target_specifiers: logger.error("Target list empty for 'update_by_list'. Exiting."); return
            logger.info(f"Targeting by specifiers: {target_specifiers}")
            matched_ids = set()
            for spec in target_specifiers:
                for entry in all_movie_entries_master_list:
                    match = (spec.get('imdb_id') and entry.imdb_id == spec['imdb_id']) or \
                            (spec.get('tmdb_id') and entry.tmdb_movie_id == spec['tmdb_id']) or \
                            (spec.get('title') and entry.movie_title.lower() == spec['title'].lower() and \
                             (not spec.get('year') or str(entry.movie_year) == str(spec['year'])))
                    if match and entry.tmdb_movie_id not in matched_ids :
                        movies_to_target_for_session.append(entry)
                        if entry.tmdb_movie_id: matched_ids.add(entry.tmdb_movie_id)
                        logger.info(f"  Matched target: {spec} -> '{entry.movie_title}'")
                        break
        elif operation_mode == "update_all_existing":
            movies_to_target_for_session = list(all_movie_entries_master_list)
            logger.info(f"Targeting ALL {len(movies_to_target_for_session)} existing movies.")

        if not movies_to_target_for_session: logger.info("No movies identified for update. Exiting."); return
        logger.info(f"Total unique movies to update: {len(movies_to_target_for_session)}")

        for movie_entry_to_update in movies_to_target_for_session:
            logger.info(f"--- Updating Targeted Movie: '{movie_entry_to_update.movie_title}' ---")
            final_movie_entry = _enrich_and_update_movie_data(
                movie_data_input=movie_entry_to_update,
                is_new_movie=False,
                llm_client=llm_client_instance_param,
                llm_model_id=llm_model_id_for_api_calls_param,
                prompt_c1_template=prompt_call1_template_param,
                prompt_c2_template=prompt_call2_template_param,
                prompt_c3_template=prompt_call3_template_param,
                prompt_c4_template=prompt_call4_review_summary_template_param,
                prompt_plot_rel_template=prompt_constrained_plot_rel_template_param,
                current_app_config=app_config,
                passed_tmdb_api_key=TMDB_API_KEY_GLOBAL,
                passed_omdb_api_key=OMDB_API_KEY_GLOBAL,
                logger_instance=logger,
                current_active_enrichers_cfg=active_enrichers_cfg,
                current_update_all_active_fields=update_all_active_fields_for_existing,
                current_fields_to_update_cfg=fields_to_update_cfg,
                current_key_to_enricher_group_map=key_to_enricher_group_map,
            )
            if final_movie_entry:
                idx_to_replace = -1
                if movie_entry_to_update.tmdb_movie_id is not None:
                    idx_to_replace = next((i for i,e in enumerate(all_movie_entries_master_list) if e.tmdb_movie_id == movie_entry_to_update.tmdb_movie_id), -1)
                else:
                    idx_to_replace = next((i for i,e in enumerate(all_movie_entries_master_list) if e.movie_title.lower() == movie_entry_to_update.movie_title.lower() and e.movie_year == movie_entry_to_update.movie_year), -1)

                if idx_to_replace != -1: all_movie_entries_master_list[idx_to_replace] = final_movie_entry; logger.info(f"  Updated '{final_movie_entry.movie_title}'.")
                else: all_movie_entries_master_list.append(final_movie_entry); logger.warning(f"  Appended updated '{final_movie_entry.movie_title}' (original not found by ID/Title).")

                save_movie_data_to_yaml([entry.model_dump(exclude_none=True) for entry in all_movie_entries_master_list], app_config['output_file'])
                logger.info(f"  Saved '{final_movie_entry.movie_title}' to '{app_config['output_file']}'.")
            else: logger.error(f"  Skipping save for '{movie_entry_to_update.movie_title}' due to enrichment failure.")
            time.sleep(app_config.get('api_request_delay_seconds_general', 2))
    else: logger.critical(f"Unsupported operation mode: '{operation_mode}'. Exiting."); return

    logger.info(f"===== MOVIE ENRICHMENT SESSION FINISHED =====")
    logger.info(f"Final total movies in '{app_config['output_file']}': {len(all_movie_entries_master_list)}")
    if active_enrichers_cfg.get('fetch_character_images') or active_enrichers_cfg.get('fetch_relationship_images'):
        logger.info(f"Images saved to: '{app_config['character_image_save_path']}'")

if __name__ == "__main__":
    run_enrichment_pipeline()