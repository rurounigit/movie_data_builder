# enrichers/character_enricher.py
import yaml
import openai
import time # For sleep in image downloading
from typing import List, Optional, Dict, Any

from models.movie_models import CharacterListItem, Relationship, LLMCall2Output
from data_providers.llm_clients import get_llm_response_and_parse
from utils.image_downloader import (
    download_actor_image_tmdb,
    download_character_image_ddg,
    download_ddg_image_for_query
)
from utils.helpers import slugify


def enrich_characters_and_get_relationships(
    llm_client: openai.OpenAI,
    llm_model_id: str,
    movie_title: str,
    movie_year: str,
    raw_tmdb_characters_yaml_str: str,
    prompt_template: str,
    max_tokens: int,
    config: Dict[str, Any], # Pass the whole app_config
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


def trigger_character_image_downloads(
    character_list_from_llm: List[CharacterListItem],
    movie_title: str,
    movie_tmdb_id: Optional[int],
    save_path_base: str,
    tmdb_api_key: str,
    tmdb_image_base_url: str,
    tmdb_image_size: str,
    ddg_num_images_per_search: int,
    ddg_sleep_after_character_group: float, # New
    ddg_sleep_between_individual_downloads: float, # New
    logger: Optional[Any] = None
) -> None:
    if not character_list_from_llm:
        if logger: logger.info("  No characters in list for image download.")
        return

    for char_idx, char_data in enumerate(character_list_from_llm):
        if not char_data.tmdb_person_id:
            if logger: logger.warning(f"  Skipping image download for '{char_data.name}': Missing TMDB Person ID.")
            continue

        person_id_int = int(char_data.tmdb_person_id)

        if tmdb_api_key:
            if logger: logger.info(f"    Downloading Actor Image (TMDB) for '{char_data.actor_name}' (ID: {person_id_int})...")
            download_actor_image_tmdb(
                tmdb_api_key=tmdb_api_key,
                person_id=person_id_int,
                person_name_for_log=char_data.actor_name,
                save_path=save_path_base,
                base_image_url=tmdb_image_base_url,
                image_size=tmdb_image_size,
                logger=logger
            )
        else:
            if logger: logger.warning(f"    TMDB API key missing. Skipping actor image download for '{char_data.actor_name}'.")

        if logger: logger.info(f"    Downloading Character Image (DDG) for '{char_data.name}' from '{movie_title}'...")
        download_character_image_ddg(
            character_name=char_data.name,
            movie_title=movie_title,
            tmdb_person_id=person_id_int,
            num_images_to_fetch=ddg_num_images_per_search,
            save_path=save_path_base,
            sleep_between_downloads=ddg_sleep_between_individual_downloads, # Pass through
            logger=logger
        )

        # Sleep after processing a character's image group (TMDB actor + DDG character)
        # Don't sleep after the very last character in the list
        if char_idx < len(character_list_from_llm) - 1:
            if logger: logger.debug(f"    Sleeping for {ddg_sleep_after_character_group}s after processing images for '{char_data.name}'...")
            time.sleep(ddg_sleep_after_character_group)


def _sanitize_for_filename_component(name: str) -> str:
    return slugify(name)


def trigger_relationship_image_downloads(
    relationships: List[Relationship],
    movie_title: str,
    save_path_base: str,
    ddg_num_images_per_relationship_search: int,
    max_relationships_to_process: int,
    ddg_sleep_after_relationship_group: float, # New
    ddg_sleep_between_individual_downloads: float, # New
    logger: Optional[Any] = None
) -> None:
    if not relationships:
        if logger: logger.info("  No relationships provided for relationship image download.")
        return

    if logger: logger.info(f"  Attempting to download images for up to {max_relationships_to_process} relationships for '{movie_title}'.")

    processed_count = 0
    for rel_idx, rel in enumerate(relationships):
        if processed_count >= max_relationships_to_process:
            if logger: logger.info(f"    Reached limit of {max_relationships_to_process} relationships for image download.")
            break

        source_name = str(rel.source).strip()
        target_name = str(rel.target).strip()

        if not source_name or not target_name:
            if logger: logger.debug(f"    Skipping relationship image download due to empty source/target string after strip: '{source_name}' -> '{target_name}'")
            continue

        search_query = f"{source_name} and {target_name} {movie_title}"
        sane_source = _sanitize_for_filename_component(source_name)
        sane_target = _sanitize_for_filename_component(target_name)
        filename_prefix = f"rel_{sane_source}_{sane_target}"

        if logger: logger.info(f"    Downloading Relationship Image (DDG) for '{source_name}' & '{target_name}' from '{movie_title}' (Query: '{search_query}')...")

        download_ddg_image_for_query(
            query=search_query,
            filename_prefix_base=filename_prefix,
            num_images_to_fetch=ddg_num_images_per_relationship_search,
            save_path=save_path_base,
            sleep_between_downloads=ddg_sleep_between_individual_downloads, # Pass through
            logger=logger
        )
        processed_count += 1

        # Sleep after processing a relationship's image group
        # Don't sleep after the very last relationship processed or if it's the last in the list
        if processed_count < max_relationships_to_process and rel_idx < len(relationships) -1 :
             if logger: logger.debug(f"    Sleeping for {ddg_sleep_after_relationship_group}s after processing images for relationship '{source_name}' & '{target_name}'...")
             time.sleep(ddg_sleep_after_relationship_group)

    if logger: logger.info(f"  Finished DDG downloads for relationships. Processed {processed_count} relationships for image search.")


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
    seen_pairs = set()

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

        pair_key = tuple(sorted((source_norm.lower(), target_norm.lower())))

        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            rel_dict = rel_model.model_dump()
            rel_dict['source'] = source_norm
            rel_dict['target'] = target_norm
            try:
                unique_relationships.append(Relationship.model_validate(rel_dict))
            except Exception as e:
                 log_msg = f"Error validating normalized relationship after deduplication: {e}. Data: {rel_dict}"
                 if logger: logger.error(f"    {log_msg}")

    if logger and (len(unique_relationships) < len(relationships_data_from_llm)):
        logger.debug(f"  Normalized/deduplicated relationships from {len(relationships_data_from_llm)} to {len(unique_relationships)}.")
    return unique_relationships