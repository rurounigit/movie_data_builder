# enrichers/character_enricher.py
# import yaml # No longer needed here for loading
import openai
from typing import List, Optional, Dict, Any

from models.movie_models import CharacterListItem, Relationship, LLMCall2Output
from data_providers.llm_clients import get_llm_response_and_parse
from data_providers.tmdb_api import fetch_and_save_character_image # Keep this direct import


def enrich_characters_and_get_relationships(
    llm_client: openai.OpenAI,
    llm_model_id: str,
    movie_title: str,
    movie_year: str,
    raw_tmdb_characters_yaml_str: str,
    prompt_template: str,
    max_tokens: int,
    config: Dict[str, Any],
    logger: Optional[Any] = None
) -> Optional[LLMCall2Output]:
    prompt_user_content = prompt_template.format(
        movie_title=movie_title,
        movie_year=movie_year,
        raw_tmdb_characters_yaml=raw_tmdb_characters_yaml_str
    )
    messages = [
        {"role": "system", "content": "You enrich character lists and generate relationships in YAML or JSON format, adhering to the provided structure."},
        {"role": "user", "content": prompt_user_content}
    ]

    parsing_context = f"LLM Call 2 (Chars/Rels) for '{movie_title}'"
    data = get_llm_response_and_parse(
        client=llm_client,
        model_id_for_call=llm_model_id,
        messages_history=messages,
        max_tokens=max_tokens,
        temperature=0.4,
        logger=logger,
        parsing_context=parsing_context
    )

    if not data:
        return None

    try:
        if 'character_list' not in data or not isinstance(data.get('character_list'), list):
            if logger: logger.warning(f"Warning ({parsing_context}): 'character_list' was missing or not a list from LLM. Defaulted to empty.")
            data['character_list'] = []

        if 'relationships' not in data or not isinstance(data.get('relationships'), list):
            if logger: logger.warning(f"Warning ({parsing_context}): 'relationships' was missing or not a list from LLM. Defaulted to empty.")
            data['relationships'] = []

        return LLMCall2Output.model_validate(data)
    except Exception as e:
        log_msg = f"Critical ({parsing_context}): Data validation error for Pydantic model: {e}. Parsed data: {str(data)[:500]}"
        if logger: logger.error(log_msg)
    return None


def fetch_and_assign_character_images(
    character_list_from_llm: List[CharacterListItem],
    save_path_base: str,
    tmdb_api_key: str,
    tmdb_image_base_url: str,
    tmdb_image_size: str,
    logger: Optional[Any] = None
) -> List[CharacterListItem]:
    updated_list = []
    for char_data in character_list_from_llm:
        image_local_path = None
        if char_data.tmdb_person_id is not None:
            try:
                person_id_int = int(char_data.tmdb_person_id)
                image_local_path = fetch_and_save_character_image( # Direct call to imported function
                    tmdb_api_key=tmdb_api_key,
                    person_id=person_id_int,
                    person_name_for_log=char_data.name,
                    save_path=save_path_base,
                    base_image_url=tmdb_image_base_url,
                    image_size=tmdb_image_size,
                    logger=logger # Pass the logger through
                )
            except ValueError:
                log_msg = f"Warning: Invalid tmdb_person_id format '{char_data.tmdb_person_id}' for character '{char_data.name}'. Cannot fetch image."
                if logger: logger.warning(f"    {log_msg}") # Indent for context
            except Exception as e:
                log_msg = f"Error during image fetch process for character {char_data.name} (ID: {char_data.tmdb_person_id}): {e}"
                if logger: logger.error(f"    {log_msg}") # Indent for context
        char_data.image_file = image_local_path
        updated_list.append(char_data)
    return updated_list


def deduplicate_and_normalize_relationships(
    llm_enriched_character_list: List[CharacterListItem],
    relationships_data_from_llm: List[Relationship],
    logger: Optional[Any] = None
) -> List[Relationship]:
    if not llm_enriched_character_list:
        return relationships_data_from_llm or []
    if not relationships_data_from_llm:
        return []

    name_map: Dict[str, str] = {}
    for char_entry in llm_enriched_character_list:
        canonical_name = char_entry.name.strip()
        if canonical_name:
            name_map[canonical_name.lower()] = canonical_name
            if char_entry.aliases:
                for alias in char_entry.aliases:
                    alias_str = str(alias).strip()
                    if alias_str:
                        name_map[alias_str.lower()] = canonical_name

    if not name_map:
        if logger: logger.warning("Cannot normalize relationships: no valid character names in character list.")
        return relationships_data_from_llm

    unique_relationships: List[Relationship] = []
    seen_mutual_pairs = set()
    seen_directed_pairs = set()

    for rel_model in relationships_data_from_llm:
        original_source_llm = rel_model.source.strip()
        original_target_llm = rel_model.target.strip()

        if not original_source_llm or not original_target_llm:
            if logger: logger.debug(f"Skipping relationship with empty source/target: '{original_source_llm}' -> '{original_target_llm}'.")
            continue

        source_norm = name_map.get(original_source_llm.lower())
        target_norm = name_map.get(original_target_llm.lower())

        if not source_norm or not target_norm:
            if logger: logger.debug(f"Could not normalize relationship: '{original_source_llm}' -> '{original_target_llm}'. Source/Target not in character list. Skipping.")
            continue
        if source_norm == target_norm:
            if logger: logger.debug(f"Skipping self-relationship for '{source_norm}'.")
            continue

        rel_dict = rel_model.model_dump()
        rel_dict['source'] = source_norm
        rel_dict['target'] = target_norm
        is_directed = rel_model.directed

        current_pair_key: Any
        is_new_relationship = False

        if not is_directed:
            current_pair_key = tuple(sorted((source_norm.lower(), target_norm.lower())))
            if current_pair_key not in seen_mutual_pairs:
                seen_mutual_pairs.add(current_pair_key)
                seen_directed_pairs.add((source_norm.lower(), target_norm.lower()))
                seen_directed_pairs.add((target_norm.lower(), source_norm.lower()))
                is_new_relationship = True
        else:
            current_pair_key = (source_norm.lower(), target_norm.lower())
            mutual_equivalent_key = tuple(sorted((source_norm.lower(), target_norm.lower())))
            if current_pair_key not in seen_directed_pairs and mutual_equivalent_key not in seen_mutual_pairs:
                seen_directed_pairs.add(current_pair_key)
                is_new_relationship = True

        if is_new_relationship:
            try:
                unique_relationships.append(Relationship.model_validate(rel_dict))
            except Exception as e:
                 log_msg = f"Error validating normalized relationship after deduplication: {e}. Data: {rel_dict}"
                 if logger: logger.error(f"    {log_msg}") # Indent for context

    if logger and (len(unique_relationships) < len(relationships_data_from_llm)):
        logger.debug(f"  Normalized/deduplicated relationships from {len(relationships_data_from_llm)} to {len(unique_relationships)}.")
    return unique_relationships