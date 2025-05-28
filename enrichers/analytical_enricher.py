# enrichers/analytical_enricher.py
import openai
from typing import List, Optional, Dict, Any
import ast

from models.movie_models import LLMCall3Output, Recommendation # MatchingTags is part of LLMCall3Output
from data_providers.llm_clients import get_llm_response_and_parse

def generate_analytical_data(
    llm_client: openai.OpenAI,
    llm_model_id: str,
    movie_title: str,
    movie_year: str,
    prompt_template: str,
    max_tokens: int,
    config: Dict[str, Any],
    logger: Optional[Any] = None
) -> Optional[LLMCall3Output]:
    num_analytical_keys = len(LLMCall3Output.model_fields.keys())

    prompt_user_content = prompt_template.format(
        movie_title_from_call_1=movie_title,
        movie_year_from_call_1=movie_year,
        num_analytical_keys=num_analytical_keys,
    )
    messages = [
        {"role": "system", "content": "You provide analytical movie information in strict JSON or YAML format, adhering to the requested structure."},
        {"role": "user", "content": prompt_user_content}
    ]

    parsing_context = f"LLM Call 3 (Analytical) for '{movie_title}'"
    data = get_llm_response_and_parse(
        client=llm_client,
        model_id_for_call=llm_model_id,
        messages_history=messages,
        max_tokens=max_tokens,
        logger=logger,
        parsing_context=parsing_context
    )

    if not data:
        return None

    try:
        # --- Data Transformation for Pydantic Validation ---
        if "recommendations" in data and isinstance(data.get("recommendations"), list):
            transformed_recommendations = []
            for i, rec_item_from_llm in enumerate(data["recommendations"]):
                rec_title, rec_year, rec_explanation = None, None, None

                if isinstance(rec_item_from_llm, dict): # LLM provided a dict, ideal case
                    rec_title = rec_item_from_llm.get("title")
                    rec_year = rec_item_from_llm.get("year")
                    rec_explanation = rec_item_from_llm.get("explanation")
                elif isinstance(rec_item_from_llm, list) and len(rec_item_from_llm) == 3: # LLM provided a list
                    rec_title, rec_year, rec_explanation = rec_item_from_llm[0], rec_item_from_llm[1], rec_item_from_llm[2]
                elif isinstance(rec_item_from_llm, str): # LLM provided a stringified list
                    try:
                        potential_list = ast.literal_eval(rec_item_from_llm)
                        if isinstance(potential_list, list) and len(potential_list) == 3:
                            rec_title, rec_year, rec_explanation = potential_list[0], potential_list[1], potential_list[2]
                        else:
                            if logger: logger.warning(f"Warning ({parsing_context}): Recommendation at index {i}: ast.literal_eval on string did not yield a 3-element list. String: '{rec_item_from_llm}'")
                    except (ValueError, SyntaxError, TypeError) as e_eval:
                        if logger: logger.warning(f"Warning ({parsing_context}): Recommendation at index {i}: ast.literal_eval failed for string '{rec_item_from_llm}'. Error: {e_eval}")

                if rec_title is not None and rec_year is not None and rec_explanation is not None:
                    transformed_recommendations.append({
                        "title": str(rec_title).strip(),
                        "year": str(rec_year).strip(), # Pydantic model handles Union[int, str]
                        "explanation": str(rec_explanation).strip()
                        # imdb_id will be fetched later
                    })
                else: # If not a dict, and not a processable list/string
                    if logger: logger.warning(f"Warning ({parsing_context}): Recommendation at index {i}: Malformed item. Item: '{str(rec_item_from_llm)[:100]}'. Skipping.")

            data["recommendations"] = transformed_recommendations
        elif "recommendations" in data:
            log_msg = f"Warning ({parsing_context}): 'recommendations' from LLM was not a list (Type: {type(data.get('recommendations'))}). Setting to empty. Value: {str(data.get('recommendations'))[:100]}"
            if logger: logger.warning(log_msg)
            data["recommendations"] = []


        # Pydantic: class GenreMix(BaseModel): genres: Dict[str, int]
        # LLMCall3Output: genre_mix: Optional[GenreMix] = None
        # LLM might give: "genre_mix": {"action": 80} OR "genre_mix": null
        # We need data["genre_mix"] to be {"genres": {"action": 80}} or None for LLMCall3Output
        if "genre_mix" in data:
            gm_val = data["genre_mix"]
            if isinstance(gm_val, dict):
                if not ("genres" in gm_val and isinstance(gm_val.get("genres"), dict)):
                    data["genre_mix"] = {"genres": gm_val} # Wrap it
                # else: already correctly structured as {"genres": {...}}
            elif gm_val is None:
                data["genre_mix"] = None # Correct for Optional[GenreMix]
            else: # Malformed, set to None to allow Pydantic to handle Optional
                if logger: logger.warning(f"Warning ({parsing_context}): 'genre_mix' from LLM was not a dict or null. Setting to None. Value: {str(gm_val)[:50]}")
                data["genre_mix"] = None
        # If "genre_mix" is not in data, Pydantic will handle it as None due to Optional[GenreMix]

        # Pydantic: class MatchingTags(BaseModel): tags: Optional[Dict[str, str]] = None
        # LLMCall3Output: matching_tags: Optional[MatchingTags] = None
        # LLM might give: "matching_tags": {"tag": "expl"} OR "matching_tags": null
        # We need data["matching_tags"] to be {"tags": {"tag": "expl"}} or None for LLMCall3Output
        if "matching_tags" in data:
            mt_val = data["matching_tags"]
            if isinstance(mt_val, dict):
                if not ("tags" in mt_val and (mt_val.get("tags") is None or isinstance(mt_val.get("tags"), dict))):
                    data["matching_tags"] = {"tags": mt_val} # Wrap it
                # else: already correctly structured as {"tags": {...}}
            elif mt_val is None:
                data["matching_tags"] = None # Correct for Optional[MatchingTags]
            else: # Malformed
                if logger: logger.warning(f"Warning ({parsing_context}): 'matching_tags' from LLM was not a dict or null. Setting to None. Value: {str(mt_val)[:50]}")
                data["matching_tags"] = None
        # If "matching_tags" is not in data, Pydantic will handle it as None

        # For character_profile_big5 and character_profile_myersbriggs,
        # Pydantic expects a dict that matches the structure of Big5ProfileModel and MyersBriggsProfileModel or None.
        # No special transformation needed here if the LLM provides the correct dict structure or null directly.
        # Example: "character_profile_big5": {"Openness": {"score": 1, "explanation": "..."}} or "character_profile_big5": null

        return LLMCall3Output.model_validate(data)
    except Exception as e:
        log_msg = f"Critical ({parsing_context}): Data validation error for Pydantic model LLMCall3Output: {e}. Parsed data: {str(data)[:500]}"
        if logger: logger.error(log_msg)
    return None