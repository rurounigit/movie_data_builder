# enrichers/movie_data_enricher.py
from typing import Optional, Dict, Any, List
from models.movie_models import LLMCall1Output # Pydantic model for output
from data_providers.llm_clients import get_llm_response # Your LLM call function
import yaml
import openai # if client is passed

def generate_initial_movie_data(
    llm_client: openai.OpenAI, # Pass the initialized client
    llm_model_id: str, # Added: The specific model ID to use
    movie_title_from_tmdb: str,
    movie_year_from_tmdb: str,
    prompt_template: str, # Loaded by orchestrator
    max_tokens: int,
    config: Dict[str, Any], # Global app config
    logger: Optional[Any] = None # Added: Logger instance
) -> Optional[LLMCall1Output]:
    num_call_1_keys = len(LLMCall1Output.model_fields.keys())
    prompt_content = prompt_template.format(
        movie_title_from_tmdb=movie_title_from_tmdb,
        movie_year_from_tmdb=movie_year_from_tmdb,
        expected_title_key="movie_title",
        expected_year_key="movie_year",
        num_call_1_keys=num_call_1_keys
    )
    messages = [
        {"role": "system", "content": "You are an assistant that provides movie information in strict YAML format for a given movie. Ensure the output adheres to the requested structure."},
        {"role": "user", "content": prompt_content}
    ]

    # Updated call to get_llm_response
    raw_response = get_llm_response(
        client=llm_client,
        model_id_for_call=llm_model_id,
        messages_history=messages,
        max_tokens=max_tokens,
        logger=logger
    )

    if not raw_response:
        if logger: logger.error(f"LLM Call 1 FAILED to respond for '{movie_title_from_tmdb}'.")
        else: print(f"      LLM Call 1 FAILED to respond for '{movie_title_from_tmdb}'.")
        return None

    try:
        if raw_response.startswith("```yaml"):
            raw_response = raw_response[7:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3].strip()
        elif raw_response.startswith("yaml"):
             raw_response = raw_response[4:].lstrip()
        if raw_response.endswith("```"):
             raw_response = raw_response[:-3].strip()

        data = yaml.safe_load(raw_response)

        llm_title = data.get("movie_title")
        llm_year = data.get("movie_year")

        if not (isinstance(llm_title, str) and llm_title.strip()):
            log_msg = f"Error (LLM Call 1): Key 'movie_title' missing or invalid. LLM value: '{llm_title}'"
            if logger: logger.error(log_msg)
            else: print(f"      {log_msg}")
            return None
        if str(llm_title).strip().lower() != str(movie_title_from_tmdb).strip().lower():
            log_msg = f"Error (LLM Call 1): LLM returned title '{llm_title}' != given '{movie_title_from_tmdb}'."
            if logger: logger.error(log_msg)
            else: print(f"      {log_msg}")
            return None

        year_validated_from_llm = None
        if llm_year is not None:
            year_str_temp = str(llm_year).strip()
            if year_str_temp.isdigit() and len(year_str_temp) == 4:
                year_validated_from_llm = year_str_temp

        if year_validated_from_llm and str(year_validated_from_llm) != str(movie_year_from_tmdb):
            log_msg = f"Warning (LLM Call 1): LLM year '{year_validated_from_llm}' differs from GIVEN year '{movie_year_from_tmdb}'. Using GIVEN year."
            if logger: logger.warning(log_msg)
            else: print(f"      {log_msg}")

        data["movie_title"] = movie_title_from_tmdb
        data["movie_year"] = movie_year_from_tmdb

        if "complex_search_queries" in data and isinstance(data["complex_search_queries"], str):
            data["complex_search_queries"] = [data["complex_search_queries"]]

        return LLMCall1Output.model_validate(data)
    except yaml.YAMLError as ye:
        log_msg = f"Critical: LLM Call 1 YAML parsing error: {ye}. Text: {str(raw_response)[:300]}"
        if logger: logger.error(log_msg)
        else: print(f"      {log_msg}")
    except Exception as e:
        log_msg = f"Critical: LLM Call 1 Data validation error: {e}. Text: {str(raw_response)[:300]}"
        if logger: logger.error(log_msg)
        else: print(f"      {log_msg}")
    return None