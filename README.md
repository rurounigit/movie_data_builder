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
    *   **TMDB Review Summary:** Fetches TMDB user reviews and generates a concise LLM-powered summary.
    *   **Constrained Plot Description:** Generates a plot description strictly using character names from TMDB's initial list, informed by LLM-generated relationships.
*   **Enhanced Image Downloading:**
    *   Downloads **actor profile images** from TMDB (based on TMDB Person ID).
    *   Downloads **character-specific images** using DuckDuckGo search (based on character name and movie title).
    *   Images are saved to the `output/character_images` directory with descriptive filenames (e.g., `[person_id].jpg` for actors, `[person_id]_char_[character_slug].jpg` for characters).
    *   **Note:** The paths to these downloaded images are NOT stored directly within the `character_list` in the `clean_movie_database.yaml` file, keeping the YAML focused on textual data.
*   **Flexible Operation Modes:** The pipeline supports various modes to control which movies are processed and how existing data is handled:
    *   **`fetch_and_add_new`**: Scans TMDB top-rated. Primarily adds *new* movies. Can optionally update *existing* movies if they are encountered during the TMDB scan (controlled by `update_existing_if_encountered_during_fetch`).
    *   **`update_all_existing`**: Processes and updates *ALL* movies currently stored in your `output/clean_movie_database.yaml`.
    *   **`update_by_list`**: Processes and updates *only* specific movies identified in the `target_movies_to_update` list.
    *   **`update_by_range`**: Processes and updates movies from `output_file` based on their 0-based index range, specified in `target_existing_movies_by_index_range`.
*   **Granular Update Control:** The `fields_to_update` setting allows you to specify exactly which fields (e.g., "recommendations", "imdb_id") should be updated for existing movies, applicable across all update scenarios. If empty, all fields relevant to active enrichers will be updated.
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
│   ├── constrained_plot_rel_enricher.py # LLM Call for constrained plot
│   ├── movie_data_enricher.py          # LLM Call 1 logic
│   └── review_summarizer_enricher.py   # LLM Call for TMDB review summary
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
│   ├── movie_initial_data_prompt.txt
│   ├── summarize_tmdb_reviews_prompt.txt
│   └── plot_constrained_with_relations_prompt.txt
├── utils/
│   ├── __init__.py
│   ├── helpers.py                      # Utility functions (YAML, logging, tokens, image download helpers)
│   └── image_downloader.py             # New module for image downloading logic (TMDB and DDG)
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
    *Note: The `duckduckgo_search` library (used for character images) might require `html-parser` or similar dependencies that Poetry should handle. If you encounter issues, refer to its documentation.*

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
    *   **Choose your `operation_mode`**: This is the primary control for what the pipeline will do.
        *   **`fetch_and_add_new`:** (Default) The pipeline scans TMDB top-rated movies. If a movie is *new* to your database, it's added and fully enriched. If a movie *already exists*, its treatment is controlled by `update_existing_if_encountered_during_fetch`.
        *   **`update_all_existing`:** The pipeline loads *all* movies from your `output/clean_movie_database.yaml` and attempts to update them.
        *   **`update_by_list`:** The pipeline updates *only* specific movies listed in `target_movies_to_update`.
        *   **`update_by_range`:** The pipeline updates movies from your `output/clean_movie_database.yaml` based on their 0-based index range specified in `target_existing_movies_by_index_range`.
    *   **Control `fetch_and_add_new` behavior with `update_existing_if_encountered_during_fetch`**:
        *   If `operation_mode` is `fetch_and_add_new` and `update_existing_if_encountered_during_fetch: true`, then existing movies found during the TMDB scan will be updated (according to `fields_to_update`).
        *   If `operation_mode` is `fetch_and_add_new` and `update_existing_if_encountered_during_fetch: false`, then existing movies found during the TMDB scan will be *skipped*, and only new movies will be added. This is the option for "only add new datasets."
    *   **Define `fields_to_update`**: This list controls *which specific top-level fields* (e.g., `recommendations`, `imdb_id`) of an *existing* movie entry will be re-generated/overwritten.
        *   If `fields_to_update` is an **empty list (`[]`)**, then *all* fields generated by currently `active_enrichers` will be updated for any existing movie that's processed for an update.
        *   If `fields_to_update` is **populated** (e.g., `["tmdb_user_review_summary", "character_profile_big5"]`), then *only* those specified fields will be updated, provided their corresponding `active_enrichers` are `true`.
        *   This setting applies universally whenever an existing movie is targeted for an update (e.g., in `update_all_existing` mode, or when `update_existing_if_encountered_during_fetch` is `true`).
    *   Review other settings:
        *   `output_file`, `raw_log_file`, `character_image_save_path`.
        *   `prompts`: Paths to the LLM prompt template files.
        *   `num_new_movies_to_fetch_this_session`, `max_tmdb_top_rated_pages_to_check`, `max_characters_from_tmdb`.
        *   `active_enrichers`: Booleans to toggle different enrichment stages (applies to all operation modes for what to generate/regenerate).
        *   `ddg_num_images_per_search`: New setting for the number of images to attempt downloading per DuckDuckGo search.
        *   Token calculation ratios and limits.

6.  **Review Prompts (`prompts/` directory):**
    *   The prompts are crucial for the quality of LLM-generated data. You may want to customize them for your chosen LLM, model, or desired output style. Note that `movie_enrich_chars_relationships_prompt.txt` has been updated to no longer request the `directed` field for relationships.

### Running the Pipeline

Once set up, run the main orchestrator from the root directory of the project (e.g., from within `movie_data_builder` if `main_orchestrator.py` is at that level, or adjust path if it's inside `movie_enrichment_project`):

```bash
poetry run python main_orchestrator.py
# Or if main_orchestrator.py is inside movie_enrichment_project:
# poetry run python movie_enrichment_project/main_orchestrator.py
```

The script will log its progress to the console and to the `raw_log_file`. The enriched movie data will be saved to the `output_file`. Downloaded character and actor images will be saved to the directory specified by `character_image_save_path`.

### How to Add a New Data Field (Data Point) to Movie Entries

Adding a new data field (e.g., "primary_theme", "notable_cinematography_technique") to your movie entries involves a coordinated effort. Assuming the new data point will be generated by one of the LLM calls:

1.  **Define in Pydantic Models (`models/movie_models.py`):**
    *   Add the new field to `MovieEntry` and, if applicable, to the intermediate LLM output model (e.g., `LLMCall3Output`).
        ```python
        # models/movie_models.py
        from typing import Optional, List, Dict
        from pydantic import BaseModel, Field

        # ... other existing models ...

        class LLMCall3Output(BaseModel):
            # ... existing fields ...
            movie_mood: Optional[str] = Field(None, description="The overall mood or atmosphere of the movie, e.g., 'Dark and Gritty'.")

        class MovieEntry(BaseModel):
            # ... existing fields ...
            movie_mood: Optional[str] = Field(None, description="The overall mood or atmosphere of the movie.")
            # ... other existing fields ...
        ```

2.  **Update LLM Prompt(s) (`prompts/` directory):**
    *   Modify the relevant prompt file (e.g., `prompts/movie_analytical_data_prompt.txt` for Call 3) to instruct the LLM to generate this new field, specifying the key name it should use in its YAML/JSON response and the desired format. Remember to update the `num_analytical_keys` variable in your prompt template if you're adding a new top-level key.

3.  **Update Enricher Function (`enrichers/` directory):**
    *   Typically, if the new field is part of an LLM output Pydantic model (e.g., `LLMCall3Output`) and the LLM correctly returns it in the expected format, no major changes are needed in the enricher function itself. The `get_llm_response_and_parse` function combined with Pydantic's `model_validate` will handle parsing and validation.

4.  **Integrate into Main Orchestrator (`main_orchestrator.py`):**
    *   The common enrichment function `_enrich_and_update_movie_data` handles the merging of LLM output into the `working_data_dict`. If `movie_mood` is part of `LLMCall3Output`, the existing `working_data_dict.update(llm3_output_data.model_dump(exclude_none=False))` will automatically include it, provided `should_update_field_local("movie_mood")` evaluates to `True`.
    *   **Add the new field's key to `key_to_enricher_group_map`:** This is important for the selective update logic. Add `"movie_mood": "analytical_data"` to this dictionary in `main_orchestrator.py` so that `fields_to_update` can correctly target it.

    ```python
    # In main_orchestrator.py (within run_enrichment_pipeline scope)
    key_to_enricher_group_map = {
        # ... existing mappings ...
        "movie_mood": "analytical_data", # Add this line
        # ... existing mappings ...
    }

    # No change needed in the _enrich_and_update_movie_data function itself for simple string fields,
    # as the loop already handles `should_update_field_local(key)` and `working_data_dict[key] = value`.
    ```

5.  **Testing:**
    *   Set `operation_mode` to `fetch_and_add_new` and `num_new_movies_to_fetch_this_session: 1` in `configs/main_config.yaml`.
    *   Ensure `active_enrichers.analytical_data: true`.
    *   Run `poetry run python main_orchestrator.py`.
    *   **Check `output/clean_movie_database.yaml`:** Verify that the `movie_mood` field is present and correctly populated for the processed movie.
    *   If testing updates for existing movies, set `operation_mode` to `update_all_existing` (or `update_by_list`/`range`) and ensure `fields_to_update: []` or `fields_to_update: ["movie_mood"]` is set.

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bugs, feature requests, or improvements.

### License

Apache 2.0