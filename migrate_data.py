import yaml
import os
from typing import List, Dict, Any, Optional

# Assuming your Pydantic models are in models.movie_models
# Adjust the import path if your structure is different for this standalone script
try:
    from models.movie_models import (
        MovieEntry, BigFiveTrait, CharacterProfileBigFive,
        CharacterProfileMyersBriggs, GenreMix, MatchingTags,
        Recommendation, RelatedMovie, CharacterListItem, Relationship
    )
except ImportError:
    print("CRITICAL: Could not import Pydantic models from models.movie_models.")
    print("Ensure this script is run from the project root or adjust PYTHONPATH.")
    exit(1)

OLD_YAML_PATH = "output/clean_movie_database.yaml"
MIGRATED_YAML_PATH = "output/clean_movie_database_migrated.yaml" # Save to a new file first

def transform_big_five(old_big_five_data: Optional[Dict[str, List[Any]]]) -> Optional[CharacterProfileBigFive]:
    if not old_big_five_data or not isinstance(old_big_five_data, dict):
        return None

    new_big_five_dict = {}
    valid_traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    all_traits_valid = True

    for trait_name in valid_traits:
        trait_value = old_big_five_data.get(trait_name)
        if isinstance(trait_value, list) and len(trait_value) == 2:
            try:
                score = int(trait_value[0])
                explanation = str(trait_value[1])
                new_big_five_dict[trait_name] = BigFiveTrait(score=score, explanation=explanation)
            except (ValueError, TypeError, IndexError):
                print(f"  Warning: Malformed BigFive trait '{trait_name}': {trait_value}. Skipping trait.")
                all_traits_valid = False # Mark this entry as potentially incomplete for this field
                new_big_five_dict[trait_name] = None # Or some default valid BigFiveTrait if required non-optional
        else:
            # If a trait is entirely missing or malformed, we might need a default
            # Pydantic model CharacterProfileBigFive expects all 5 traits.
            # For migration, if one is bad, we might skip the whole big5 or fill with defaults.
            # Let's try to create with what we have and let Pydantic catch if a whole trait is missing.
            # Or, more robustly, ensure a default/placeholder if a specific trait is problematic.
            print(f"  Warning: Missing or malformed BigFive trait '{trait_name}' in old data: {trait_value}. Will attempt to use None if model allows.")
            # If BigFiveTrait itself is optional within CharacterProfileBigFive, this would be fine.
            # But CharacterProfileBigFive *requires* Openness, Conscientiousness etc. as BigFiveTrait.
            # So we must provide a valid BigFiveTrait or make them Optional in CharacterProfileBigFive (not ideal).
            # For now, let's assume if a trait is bad, we can't form CharacterProfileBigFive.
            # This means if any old trait list is bad, new_big_five_dict might not validate.
            # Let's build what we can and see.
            # A better strategy if a trait is bad: new_big_five_dict[trait_name] = BigFiveTrait(score=0, explanation="Migration default - original data problematic")
            # For simplicity now, we'll just build what's parsable from old.
            if trait_name not in new_big_five_dict : # If it wasn't even a list of 2
                all_traits_valid = False # Mark as problematic
                new_big_five_dict[trait_name] = None # This will fail Pydantic if trait is required non-optional
                # To make it pass Pydantic, you'd need a default BigFiveTrait here if one is broken
                # e.g. new_big_five_dict[trait_name] = BigFiveTrait(score=1, explanation="Data missing/invalid from old format")

    # Check if all required traits were somewhat processable into dicts for BigFiveTrait
    # This current transform relies on Pydantic to catch if a None was placed where an object is needed
    try:
        # Filter out None values before passing to Pydantic model if some traits were skipped
        valid_trait_data_for_pydantic = {k: v for k, v in new_big_five_dict.items() if v is not None}
        if len(valid_trait_data_for_pydantic) == 5: # All traits must be present for CharacterProfileBigFive
             return CharacterProfileBigFive(**valid_trait_data_for_pydantic)
        else:
            print(f"  Warning: Could not form complete CharacterProfileBigFive due to missing/invalid traits. Original: {old_big_five_data}")
            # Create a default/empty valid one if MovieEntry requires it
            # return CharacterProfileBigFive( # Example default
            #     Openness=BigFiveTrait(score=0, explanation="Default"), ... etc for all 5
            # )
            return None # Or handle as error / skip movie if this field is critical
    except Exception as e:
        print(f"  Error transforming BigFive data: {e}. Original: {old_big_five_data}")
        return None


def transform_myers_briggs(old_mb_data: Optional[Dict[str, str]]) -> Optional[CharacterProfileMyersBriggs]:
    if not old_mb_data or not isinstance(old_mb_data, dict) or len(old_mb_data) != 1:
        if old_mb_data: print(f"  Warning: Malformed Myers-Briggs data (not a single key-value pair): {old_mb_data}")
        return None
    try:
        mbti_type, explanation = list(old_mb_data.items())[0]
        return CharacterProfileMyersBriggs(type=str(mbti_type), explanation=str(explanation))
    except Exception as e:
        print(f"  Error transforming Myers-Briggs data: {e}. Original: {old_mb_data}")
        return None

def transform_genre_mix(old_genre_data: Optional[Dict[str, Any]]) -> Optional[GenreMix]:
    if old_genre_data is None: # Can be explicitly null in old YAML
        return GenreMix(genres={}) # Or None if GenreMix itself is Optional in MovieEntry
    if not isinstance(old_genre_data, dict):
        print(f"  Warning: Malformed genre_mix data (not a dict): {old_genre_data}")
        return GenreMix(genres={}) # Default to empty
    try:
        # Ensure all values are integers
        cleaned_genres = {str(k): int(v) for k, v in old_genre_data.items()}
        return GenreMix(genres=cleaned_genres)
    except (ValueError, TypeError) as e:
        print(f"  Error transforming genre_mix data (values not int?): {e}. Original: {old_genre_data}")
        return GenreMix(genres={}) # Default to empty

def transform_complex_search_queries(old_query_data: Optional[str]) -> List[str]:
    if not old_query_data or not isinstance(old_query_data, str):
        return [] # Default to empty list if missing or not a string
    return [old_query_data.strip()] # New format is a list of strings

def transform_recommendations(old_recs_data: Optional[List[Any]]) -> Optional[List[Recommendation]]:
    if not old_recs_data or not isinstance(old_recs_data, list):
        return None # Or [] if MovieEntry.recommendations cannot be None

    new_recs_list = []
    for old_rec_item in old_recs_data:
        title, year, explanation, imdb_id = None, None, None, None
        if isinstance(old_rec_item, list) and len(old_rec_item) == 3: # Old format: [title, year, explanation]
            title, year, explanation = old_rec_item[0], old_rec_item[1], old_rec_item[2]
        elif isinstance(old_rec_item, dict): # Might already be in new-ish format from a partial prior run
            title = old_rec_item.get("title")
            year = old_rec_item.get("year")
            explanation = old_rec_item.get("explanation")
            imdb_id = old_rec_item.get("imdb_id") # Preserve if already there
        else:
            print(f"  Warning: Skipping malformed recommendation item: {old_rec_item}")
            continue

        try:
            new_recs_list.append(Recommendation(
                title=str(title) if title else "Unknown Title",
                year=str(year) if year is not None else "N/A", # Pydantic model handles Union[int, str]
                explanation=str(explanation) if explanation else "No explanation.",
                imdb_id=str(imdb_id) if imdb_id else None
            ))
        except Exception as e: # PydanticValidationError during construction
            print(f"  Warning: Could not transform recommendation item '{title}': {e}. Skipping.")

    return new_recs_list if new_recs_list else None


def transform_related_movie(old_related_data: Any) -> Optional[RelatedMovie]:
    if not old_related_data:
        return None
    if isinstance(old_related_data, str): # Old format was just title string
        return RelatedMovie(title=old_related_data, imdb_id=None)
    if isinstance(old_related_data, dict): # Might already have title/imdb_id
        try:
            return RelatedMovie.model_validate(old_related_data)
        except Exception: # PydanticValidationError
             # If dict doesn't match RelatedMovie, try to extract title
            title = old_related_data.get("title")
            if title:
                return RelatedMovie(title=str(title), imdb_id=old_related_data.get("imdb_id"))
    print(f"  Warning: Could not transform related movie data: {old_related_data}")
    return None


def transform_character_list(old_char_list: Optional[List[Dict[str, Any]]]) -> Optional[List[CharacterListItem]]:
    if not old_char_list or not isinstance(old_char_list, list):
        return None

    new_char_list = []
    for old_char in old_char_list:
        if not isinstance(old_char, dict):
            print(f"  Warning: Skipping malformed character item (not a dict): {old_char}")
            continue
        try:
            # Map old keys to new keys if necessary, ensure types
            # Your old CHARACTER_LIST_ITEM_KEYS seems to match CharacterListItem fields mostly.
            # Main concern: tmdb_person_id type.
            person_id = old_char.get("tmdb_person_id")
            if person_id is not None and not isinstance(person_id, int):
                try:
                    person_id = int(person_id)
                except (ValueError, TypeError):
                    print(f"  Warning: Invalid tmdb_person_id '{person_id}' for char '{old_char.get('name')}'. Setting to None.")
                    person_id = None

            # Ensure aliases is a list or None
            aliases = old_char.get("aliases")
            if aliases is not None and not isinstance(aliases, list):
                if isinstance(aliases, str): aliases = [aliases] # Handle single string alias
                else: aliases = None # Invalid format

            new_char_list.append(CharacterListItem(
                name=str(old_char.get("name", "Unknown Name")),
                actor_name=str(old_char.get("actor_name", "Unknown Actor")),
                tmdb_person_id=person_id,
                description=str(old_char.get("description", "")),
                group=str(old_char.get("group", "Unknown Group")),
                aliases=aliases,
                image_file=old_char.get("image_file") # Keep if exists
            ))
        except Exception as e: # PydanticValidationError
            print(f"  Warning: Could not transform character item '{old_char.get('name')}': {e}. Skipping.")
    return new_char_list if new_char_list else None


def transform_relationships(old_rels_list: Optional[List[Dict[str, Any]]]) -> Optional[List[Relationship]]:
    if not old_rels_list or not isinstance(old_rels_list, list):
        return None # Or []

    new_rels_list = []
    for old_rel in old_rels_list:
        if not isinstance(old_rel, dict):
            print(f"  Warning: Skipping malformed relationship item (not a dict): {old_rel}")
            continue
        try:
            # Ensure 'directed' is boolean
            directed_val = old_rel.get("directed", True) # Default to True if missing
            if isinstance(directed_val, str):
                if directed_val.lower() == 'true': directed_val = True
                elif directed_val.lower() == 'false': directed_val = False
                else: directed_val = True # Default for unrecognized string
            elif not isinstance(directed_val, bool):
                directed_val = True

            # Validate sentiment and tense if they exist in old data
            # Your Pydantic model for Relationship has validators for these.
            # We just need to pass the string through.

            new_rels_list.append(Relationship(
                source=str(old_rel.get("source", "Unknown Source")),
                target=str(old_rel.get("target", "Unknown Target")),
                type=str(old_rel.get("type", "Unknown Type")), # This field was missing in "Parasite" error
                directed=directed_val,
                description=str(old_rel.get("description", "")),
                sentiment=str(old_rel.get("sentiment", "neutral")), # Default if missing
                strength=int(old_rel.get("strength", 1)), # Default if missing
                tense=str(old_rel.get("tense", "present")) # Default if missing
            ))
        except Exception as e: # PydanticValidationError
            print(f"  Warning: Could not transform relationship item for source '{old_rel.get('source')}': {e}. Skipping.")
    return new_rels_list if new_rels_list else None


def migrate():
    if not os.path.exists(OLD_YAML_PATH):
        print(f"Old YAML file not found at: {OLD_YAML_PATH}. Nothing to migrate.")
        return

    print(f"Loading old data from: {OLD_YAML_PATH}")
    with open(OLD_YAML_PATH, 'r', encoding='utf-8') as f:
        old_data_list = yaml.safe_load(f)

    if not isinstance(old_data_list, list):
        print(f"Old data is not a list. Cannot migrate. Data type: {type(old_data_list)}")
        return

    migrated_movie_entries: List[MovieEntry] = []
    skipped_count = 0

    print(f"Starting migration of {len(old_data_list)} entries...")

    for i, old_entry_dict in enumerate(old_data_list):
        if not isinstance(old_entry_dict, dict):
            print(f"Skipping entry {i+1} as it's not a dictionary.")
            skipped_count +=1
            continue

        movie_title = old_entry_dict.get("movie_title", f"Unknown Movie {i+1}")
        print(f"\nProcessing entry {i+1}/{len(old_data_list)}: '{movie_title}'")

        try:
            # Prepare data for MovieEntry Pydantic model
            transformed_entry_data = {
                "movie_title": movie_title,
                "movie_year": str(old_entry_dict.get("movie_year", "N/A")),
                "tmdb_movie_id": old_entry_dict.get("tmdb_movie_id"), # Keep as is, should be int or None
                "imdb_id": old_entry_dict.get("imdb_id"), # Keep as is

                "character_profile": str(old_entry_dict.get("character_profile", "")),
                "critical_reception": str(old_entry_dict.get("critical_reception", "")),
                "visual_style": str(old_entry_dict.get("visual_style", "")),
                "most_talked_about_related_topic": str(old_entry_dict.get("most_talked_about_related_topic", "")),

                "character_profile_big5": transform_big_five(old_entry_dict.get("character_profile_big5")),
                "character_profile_myersbriggs": transform_myers_briggs(old_entry_dict.get("character_profile_myersbriggs")),
                "genre_mix": transform_genre_mix(old_entry_dict.get("genre_mix")),
                "matching_tags": MatchingTags(tags=old_entry_dict.get("matching_tags")), # Old matching_tags was already a dict of tag:explanation or null

                "complex_search_queries": transform_complex_search_queries(old_entry_dict.get("complex_search_queries")),

                "sequel": transform_related_movie(old_entry_dict.get("sequel")),
                "prequel": transform_related_movie(old_entry_dict.get("prequel")),
                "spin_off_of": transform_related_movie(old_entry_dict.get("spin_off_of")),
                "spin_off": transform_related_movie(old_entry_dict.get("spin_off")),
                "remake_of": transform_related_movie(old_entry_dict.get("remake_of")),
                "remake": transform_related_movie(old_entry_dict.get("remake")),

                "recommendations": transform_recommendations(old_entry_dict.get("recommendations")),
                "character_list": transform_character_list(old_entry_dict.get("character_list")),
                "relationships": transform_relationships(old_entry_dict.get("relationships")),
            }

            # If a transformation returned None for a required Pydantic sub-model,
            # this validation will fail unless that field in MovieEntry is Optional.
            # Example: if transform_big_five returns None, but MovieEntry.character_profile_big5 is not Optional.
            # You need to ensure your MovieEntry Pydantic model correctly defines what's truly optional.
            # For now, let's assume if a transform fails and returns None, and the field is required, Pydantic will error.
            # We should filter out Nones for fields that are Optional in MovieEntry to avoid passing `field_name: None`
            # if Pydantic expects the key to be absent for optionality, but usually `Optional[Model] = None` means `None` is valid.

            # Filter out None values for top-level keys if the field is Optional in MovieEntry
            # and you prefer the key to be absent rather than explicitly null (Pydantic usually handles null for Optional fine)
            # final_data_for_pydantic = {k: v for k, v in transformed_entry_data.items() if v is not None or not MovieEntry.model_fields[k].is_required()}
            # This is complex, Pydantic's `model_validate` should handle `None` for `Optional` fields correctly.

            validated_entry = MovieEntry.model_validate(transformed_entry_data)
            migrated_movie_entries.append(validated_entry)
            print(f"  Successfully migrated and validated '{movie_title}'.")

        except Exception as e: # Catch Pydantic ValidationErrors or others
            print(f"  CRITICAL ERROR migrating '{movie_title}': {e}")
            print(f"    Problematic old entry data (excerpt): {{'title': '{movie_title}', ...}}") # Log more if needed
            # print(f"    Transformed data before Pydantic: {transformed_entry_data}") # Can be very verbose
            skipped_count += 1

    print(f"\nMigration finished.")
    print(f"Successfully migrated: {len(migrated_movie_entries)} entries.")
    print(f"Skipped due to errors: {skipped_count} entries.")

    if migrated_movie_entries:
        print(f"Saving migrated data to: {MIGRATED_YAML_PATH}")
        # Convert Pydantic models to dicts for saving
        data_to_save = [entry.model_dump(exclude_none=True, by_alias=True) for entry in migrated_movie_entries]

        # Ensure output directory exists
        migrated_dir = os.path.dirname(MIGRATED_YAML_PATH)
        if migrated_dir and not os.path.exists(migrated_dir):
            os.makedirs(migrated_dir, exist_ok=True)

        with open(MIGRATED_YAML_PATH, 'w', encoding='utf-8') as f_out:
            yaml.dump(data_to_save, f_out, sort_keys=False, indent=2, allow_unicode=True, Dumper=yaml.SafeDumper)
        print("Migration complete. New file saved.")
    else:
        print("No entries were successfully migrated.")

if __name__ == "__main__":
    # Before running, ensure your Pydantic models in models/movie_models.py are defined.
    # Also ensure OLD_YAML_PATH points to your existing data file.
    # Backup your OLD_YAML_PATH before running if you plan to overwrite it later.
    migrate()
    print(f"\nCheck the output file: {MIGRATED_YAML_PATH}")
    print(f"If satisfied, you can rename '{MIGRATED_YAML_PATH}' to '{OLD_YAML_PATH.replace('.yaml', '_original.yaml')}' and then rename the migrated one to replace the old file for the main orchestrator.")