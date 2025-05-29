## Movie Enrichment Project

**Automated data gathering and enrichment pipeline for movies, leveraging TMDB, OMDB, and various LLM providers (via OpenAI-compatible APIs) to create a comprehensive movie database.**

This project is designed to build a rich dataset of movie information. It starts by fetching top-rated movies from The Movie Database (TMDB), then enriches this data with details from the Open Movie Database (OMDB) for IMDb IDs, and finally uses a configured Large Language Model (LLM) for generating qualitative and analytical insights.

The goal is to create a structured YAML database containing detailed profiles, character lists, relationships, analytical data (like personality profiles, genre mixes), and related movie information. The system is designed to be flexible in its choice of LLM providers through a dedicated configuration file.

### Features

*   **Multi-Source Data Aggregation:**
    *   Fetches initial movie data (title, year, TMDB ID) from TMDB's top-rated list.
    *   Retrieves IMDb IDs using a master fetching function that queries both TMDB and OMDB.
    *   Pulls raw character/actor lists from TMDB.
*   **Flexible LLM Provider Configuration:**
    *   Supports multiple LLM providers (e.g., local LM Studio, Google Gemini via compatible endpoints, official OpenAI) through `configs/llm_providers_config.yaml`.
    *   Easily switch the active LLM provider via `main_config.yaml`.
*   **LLM-Powered Enrichment:**
    *   **Call 1 (Initial Data):** Generates plot summaries, critical reception, visual style descriptions, related topics, and potential sequels/prequels using the configured LLM.
    *   **Call 2 (Characters & Relationships):** Enriches TMDB character data with descriptions, group affiliations, aliases, and generates inter-character relationships.
    *   **Call 3 (Analytical Data):** Generates Big Five & Myers-Briggs personality profiles, genre mix percentages, thematic tags, and movie recommendations.
*   **Image Fetching:** Downloads character images from TMDB (if available and enabled).
*   **Data Persistence:**
    *   Saves all enriched data into a structured YAML file (`output/clean_movie_database.yaml`).
    *   Maintains a raw log file (`output/generated_movie_data_raw_log.txt`) for debugging and transparency.
*   **Configurability:**
    *   Main application settings managed via `configs/main_config.yaml`.
    *   LLM provider details (base URLs, API key environment variable names, model IDs) managed in `configs/llm_providers_config.yaml`.
    *   LLM prompts are externalized in the `prompts/` directory.
    *   API keys for TMDB, OMDB, and selected LLM providers managed via a `.env` file.
*   **Modularity:** Code is organized into data providers, enrichers, models, and utility helpers.
*   **Pydantic Validation:** Uses Pydantic models for robust data validation at various stages, ensuring data integrity.
*   **Update Existing Entries:** Optionally, the pipeline can update existing movies in the database based on specified keys or active enrichers.

### Project Structure

```
movie_enrichment_project/
├── configs/
│   ├── main_config.yaml                # Main application configuration
│   └── llm_providers_config.yaml       # Configuration for different LLM providers
├── data_providers/
│   ├── __init__.py
│   ├── llm_clients.py                  # LLM interaction logic
│   ├── omdb_api.py                     # OMDB API interaction
│   └── tmdb_api.py                     # TMDB API interaction
├── enrichers/
│   ├── __init__.py
│   ├── analytical_enricher.py          # LLM Call 3 logic
│   ├── character_enricher.py           # LLM Call 2 logic & image fetching
│   └── movie_data_enricher.py          # LLM Call 1 logic
├── models/
│   ├── __init__.py
│   └── movie_models.py                 # Pydantic models for data structures
├── output/                             # Generated files (add to .gitignore if large/private)
│   ├── character_images/               # Downloaded character images
│   ├── clean_movie_database.yaml       # The final structured data
│   └── generated_movie_data_raw_log.txt # Raw session log
├── prompts/
│   ├── movie_analytical_data_prompt.txt
│   ├── movie_enrich_chars_relationships_prompt.txt
│   └── movie_initial_data_prompt.txt
├── utils/
│   ├── __init__.py
│   └── helpers.py                      # Utility functions (YAML, logging, tokens)
├── .env.example                        # Example environment variables
├── .gitignore
├── main_orchestrator.py                # Main script to run the pipeline
├── poetry.lock
├── pyproject.toml
└── README.md
```

### Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rurounigit/movie_data_builder.git
    cd movie_data_builder
    ```
    *(Note: Your prompt showed `cd movie_enrichment_project`, but typical structure might be cloning `movie_data_builder` and `movie_enrichment_project` being the source root within that. Adjust `cd` command as per your actual local structure after cloning.)*

2.  **Install Dependencies:**
    This project uses [Poetry](https://python-poetry.org/) for dependency management.
    ```bash
    poetry install
    ```

3.  **Set up Environment Variables (`.env`):**
    *   Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your API keys:
        ```dotenv
        OMDB_API_KEY="your_omdb_api_key"
        TMDB_API_KEY="your_tmdb_api_key"

        # API Keys for LLM Providers (as referenced in llm_providers_config.yaml)
        LM_STUDIO_API_KEY="lm-studio" # Or your specific LM Studio key
        GOOGLE_GEMINI_API_KEY="your_google_gemini_api_key"
        OPENAI_API_KEY="sk-your_openai_api_key"
        # Add other keys if you configure more providers
        ```
        *   Get an OMDB API key from [omdbapi.com](http://www.omdbapi.com/apikey.aspx).
        *   Get a TMDB API key by signing up at [themoviedb.org](https://www.themoviedb.org/documentation/api) (use an "API Read Access Token v4 Auth").
        *   Obtain API keys for any cloud-based LLM providers you intend to use (e.g., Google AI Studio for Gemini, OpenAI platform).

4.  **Configure LLM Providers (`configs/llm_providers_config.yaml`):**
    *   This file defines the connection details for each LLM service you might want to use.
    *   Review and update the example entries or add new ones. Each provider needs:
        *   `description`: A human-readable description.
        *   `base_url`: The base API endpoint URL for the LLM service (for OpenAI-compatible APIs). For official OpenAI, this can be omitted to use the library default.
        *   `api_key_env_var`: The name of the environment variable (in your `.env` file) that holds the API key for this provider.
        *   `model_id`: The specific model identifier string that the provider's API expects (e.g., `gemma-3-12b-it-qat`, `models/gemini-1.5-flash-latest`, `gpt-4-turbo`).
        *   `type`: Currently supports `openai_compatible`. (Future extensions could add other types for different SDKs).
    *   Example entry (already in the file):
        ```yaml
        google_gemini_2_0_flash_lite: # This is an example ID
          description: "Google Gemini 2.0 Flash Lite via OpenAI-compatible endpoint"
          base_url: "https://generativelanguage.googleapis.com/v1beta"
          api_key_env_var: "GOOGLE_GEMINI_API_KEY"
          model_id: "models/gemini-2.0-flash-lite" # Ensure this is a valid model ID for the endpoint
          type: "openai_compatible"
        ```
    *   **Local LLM Server (e.g., LM Studio):** If using a local server like LM Studio:
        *   Ensure LM Studio (or your chosen server) is running.
        *   Load the desired model in LM Studio.
        *   Start the local server (usually on `http://localhost:1234/v1`).
        *   Configure an entry in `llm_providers_config.yaml` pointing to this local server (e.g., the `lm_studio_gemma_3_12b` example).

5.  **Configure Main Application (`configs/main_config.yaml`):**
    *   **Crucially, set `active_llm_provider_id`** to one of the keys you defined in `configs/llm_providers_config.yaml`. This tells the application which LLM configuration to use for the session.
        ```yaml
        active_llm_provider_id: "google_gemini_2_0_flash_lite" # Or "lm_studio_gemma_3_12b", etc.
        ```
    *   Review other settings:
        *   `output_file`, `raw_log_file`, `character_image_save_path`.
        *   `prompts`: Paths to the LLM prompt template files.
        *   `num_new_movies_to_fetch_this_session`, `max_tmdb_top_rated_pages_to_check`, `max_characters_from_tmdb`.
        *   `active_enrichers`: Booleans to toggle different enrichment stages.
        *   `update_existing_movies`, `keys_to_update_for_existing`.
        *   Token calculation ratios and limits.

6.  **Review Prompts (`prompts/` directory):**
    *   The prompts are crucial for the quality of LLM-generated data. You may want to customize them for your chosen LLM, model, or desired output style.

### Running the Pipeline

Once set up, run the main orchestrator from the root directory of the project (e.g., from within `movie_data_builder` if `main_orchestrator.py` is at that level, or adjust path if it's inside `movie_enrichment_project`):

```bash
poetry run python main_orchestrator.py
# Or if main_orchestrator.py is inside movie_enrichment_project:
# poetry run python movie_enrichment_project/main_orchestrator.py
```

The script will log its progress to the console and to the `raw_log_file`. The enriched movie data will be saved to the `output_file`.

### How to Add a New Data Field (Data Point) to Movie Entries

Adding a new data field (e.g., "primary_theme", "notable_cinematography_technique") to your movie entries involves a coordinated effort. Assuming the new data point will be generated by one of the LLM calls:

1.  **Define in Pydantic Models (`models/movie_models.py`):**
    *   Add the new field to `MovieEntry` and, if applicable, to the intermediate LLM output model (e.g., `LLMCall1Output`).
        ```python
        # In LLMCall1Output (example)
        # new_field_from_llm: Optional[str] = None

        # In MovieEntry
        # new_field_in_final_yaml: Optional[str] = None
        ```

2.  **Update LLM Prompt(s) (`prompts/` directory):**
    *   Modify the relevant prompt file to instruct the LLM to generate this new field, specifying the key name it should use in its response and the desired format.

3.  **Update Enricher Function (if needed) (`enrichers/` directory):**
    *   Usually, if the field is added to the LLM output Pydantic model and the LLM returns it correctly, no major changes are needed here beyond ensuring the data gets passed through. Transformations can be added if the LLM's output for the new field needs adjustment.

4.  **Integrate into Main Orchestrator (`main_orchestrator.py`):**
    *   Ensure the data from the LLM output model is correctly mapped to the `working_data_dict` for `MovieEntry`.
    *   Add the new field's key to `key_to_enricher_group_map` if it should be part of the selective update logic.

5.  **Testing:**
    *   Test with a single movie to verify the new field is generated, parsed, and saved correctly.

Certainly! Here's the detailed example for adding a "movie_mood" data point, adapted to fit the current structure where LLM provider configuration is more flexible. This example assumes "movie_mood" will be generated by LLM Call 3 (analytical data).

---

**Example: Adding a "Movie Mood" Data Point (LLM-generated, in Call 3)**

Let's say you want to add a field called `movie_mood` to your `MovieEntry` that describes the overall mood or atmosphere of the film (e.g., "Dark and Gritty", "Hopeful and Uplifting", "Whimsical and Lighthearted"). This will be generated by LLM Call 3.

1.  **Define in Pydantic Models (`models/movie_models.py`):**
    *   First, add the `movie_mood` field to the Pydantic model that will receive the direct output from LLM Call 3, which is `LLMCall3Output`.
    *   Then, add the same field to your final `MovieEntry` model.

    ```python
    # models/movie_models.py
    from typing import Optional, List, Dict # etc.
    from pydantic import BaseModel, Field

    # ... other existing models ...

    class LLMCall3Output(BaseModel):
        # ... existing fields like character_profile_big5, recommendations, etc. ...
        movie_mood: Optional[str] = Field(None, description="The overall mood or atmosphere of the movie, e.g., 'Dark and Gritty'.")
        # ... other existing fields ...

    class MovieEntry(BaseModel):
        # ... existing fields like movie_title, character_list, etc. ...
        movie_mood: Optional[str] = Field(None, description="The overall mood or atmosphere of the movie.")
        # ... other existing fields ...
    ```

2.  **Update LLM Prompt for Call 3 (`prompts/movie_analytical_data_prompt.txt`):**
    *   You need to instruct the LLM to generate this new piece of information and tell it what key name to use in its YAML/JSON response.

    Add a new instruction within the `MOVIE_PROMPT_ANALYTICAL_TEMPLATE` text. For example, you could add it to the list of requested keys:

    ```text
    # prompts/movie_analytical_data_prompt.txt
    # ... (existing parts of the prompt) ...
    For the movie titled "{movie_title_from_call_1}" (released around {movie_year_from_call_1}).
    Provide the following analytical information...
    Respond strictly in YAML format. The YAML should be a single object with these {num_analytical_keys} top-level keys:
    # ... (existing key descriptions like character_profile_big5, genre_mix, etc.) ...
    'movie_mood' (string: Describe the overall mood or atmosphere of the movie in a short phrase. Examples: "Dark and Gritty", "Hopeful and Uplifting", "Whimsical and Lighthearted", "Tense and Suspenseful", "Melancholic and Reflective").
    # ... (rest of the prompt, including the example for recommendations) ...

    # You might also add it to an example output structure if you have one in the prompt:
    # Example of how the YAML output might look (snippet):
    # ...
    # genre_mix:
    #   genres:
    #     action: 70
    #     drama: 30
    # movie_mood: "Intense and Action-Packed"
    # recommendations:
    # ...
    ```
    *   **Important:** Remember to update `{num_analytical_keys}` in the prompt if you're adding a new top-level key. If `movie_mood` is a new top-level key requested from the LLM, `len(LLMCall3Output.model_fields.keys())` (which calculates `num_analytical_keys` in `analytical_enricher.py`) will automatically reflect this change once `movie_mood` is added to the `LLMCall3Output` model.

3.  **Update Enricher Function (`enrichers/analytical_enricher.py`):**
    *   The `LLMCall3Output` Pydantic model (which you updated in Step 1) is used to validate the output of `analytical_enricher.generate_analytical_data`.
    *   If the LLM returns the `movie_mood` as a simple string under the key `"movie_mood"` (as instructed in the prompt), the existing parsing logic in `llm_clients.get_llm_response_and_parse` and the subsequent `LLMCall3Output.model_validate(data)` should automatically handle it. No specific transformation logic for `movie_mood` would be needed *within* `analytical_enricher.py` itself, assuming it's a direct string field.

4.  **Integrate into Main Orchestrator (`main_orchestrator.py`):**
    *   When `analytical_enricher.generate_analytical_data` returns the `llm3_output_data_model` (which is an instance of `LLMCall3Output`), the `movie_mood` field will be part of it.
    *   The existing line that updates `working_data_dict` will automatically include it:
        ```python
        # In main_orchestrator.py, within the analytical_data block:
        # ...
        if llm3_output_data_model: # Renamed from llm3_output_data for clarity
            logger.info(f"  Success: Analytical Data for '{movie_title_for_calls}'.")
            working_data_dict.update(llm3_output_data_model.model_dump(exclude_none=False)) # This will include movie_mood
        else:
            logger.warning(f"  Failure or incomplete: Analytical Data for '{movie_title_for_calls}'.")
            if not is_updating_existing_movie:
                logger.error(f"    Critical analytical data missing for new movie '{movie_title_for_calls}'. Skipping final assembly.")
                continue
        # ...
        ```
    *   **Update `key_to_enricher_group_map`:** If you want `movie_mood` to be considered when `update_existing_movies` and `keys_to_update_for_existing` are used, add it to the map:
        ```python
        # In main_orchestrator.py
        key_to_enricher_group_map = {
            # ... existing mappings ...
            "movie_mood": "analytical_data",
            # ... existing mappings ...
        }
        ```

5.  **Testing:**
    *   Set `num_new_movies_to_fetch_this_session: 1` in `configs/main_config.yaml`.
    *   Ensure the `analytical_data` enricher is active: `active_enrichers: analytical_data: true`.
    *   Run `poetry run python main_orchestrator.py`.
    *   **Check the raw log (`output/generated_movie_data_raw_log.txt`):** Look for the LLM Call 3 output to see if the LLM included the `movie_mood` key and value.
    *   **Check the console output:** Look for any parsing or validation errors related to `movie_mood`.
    *   **Check the final `output/clean_movie_database.yaml`:** Verify that the `movie_mood` field is present and correctly populated for the processed movie.
    *   If testing updates, ensure that if `update_existing_movies: true` and `"movie_mood"` is in `keys_to_update_for_existing`, the field gets updated.

This example shows that for adding a relatively simple, LLM-generated string field:
*   The primary work is in defining it in the Pydantic models and crafting the LLM prompt.
*   The enricher and orchestrator often require minimal changes if the Pydantic models are set up correctly and the LLM follows instructions.
*   The `key_to_enricher_group_map` is important for the update strategy.

If the new data point were more complex (e.g., a list of structured objects for "awards"), the enricher might need more transformation logic, and you'd define more Pydantic sub-models.

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bugs, feature requests, or improvements.

### License

Apache 2.0
