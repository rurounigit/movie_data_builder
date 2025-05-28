## Movie Enrichment Project

**Automated data gathering and enrichment pipeline for movies, leveraging TMDB, OMDB, and local LLMs to create a comprehensive movie database.**

This project is designed to build a rich dataset of movie information. It starts by fetching top-rated movies from The Movie Database (TMDB), then enriches this data with details from the Open Movie Database (OMDB) for IMDb IDs, and finally uses a local Large Language Model (LLM) via an OpenAI-compatible API (like LM Studio) for generating qualitative and analytical insights.

The goal is to create a structured YAML database containing detailed profiles, character lists, relationships, analytical data (like personality profiles, genre mixes), and related movie information.

### Features

*   **Multi-Source Data Aggregation:**
    *   Fetches initial movie data (title, year, TMDB ID) from TMDB's top-rated list.
    *   Retrieves IMDb IDs using a master fetching function that queries both TMDB and OMDB.
    *   Pulls raw character/actor lists from TMDB.
*   **LLM-Powered Enrichment:**
    *   **Call 1 (Initial Data):** Generates plot summaries, critical reception, visual style descriptions, related topics, and potential sequels/prequels.
    *   **Call 2 (Characters & Relationships):** Enriches TMDB character data with descriptions, group affiliations, aliases, and generates inter-character relationships with types, sentiments, and strengths.
    *   **Call 3 (Analytical Data):** Generates Big Five & Myers-Briggs personality profiles for the movie's overall character, genre mix percentages, thematic tags, and movie recommendations.
*   **Image Fetching:** Downloads character images from TMDB (if available and enabled).
*   **Data Persistence:**
    *   Saves all enriched data into a structured YAML file (`output/clean_movie_database.yaml`).
    *   Maintains a raw log file (`output/generated_movie_data_raw_log.txt`) for debugging and transparency.
*   **Configurability:**
    *   Main settings managed via `configs/main_config.yaml`.
    *   LLM prompts are externalized in the `prompts/` directory.
    *   API keys and LLM endpoint managed via a `.env` file.
*   **Modularity:** Code is organized into data providers, enrichers, models, and utility helpers.
*   **Pydantic Validation:** Uses Pydantic models for robust data validation at various stages, ensuring data integrity.
*   **Update Existing Entries:** Optionally, the pipeline can update existing movies in the database based on specified keys or active enrichers.

### Project Structure

```
movie_enrichment_project/
├── configs/
│   └── main_config.yaml        # Main application configuration
├── data_providers/
│   ├── __init__.py
│   ├── llm_clients.py          # LLM interaction logic
│   ├── omdb_api.py             # OMDB API interaction
│   └── tmdb_api.py             # TMDB API interaction
├── enrichers/
│   ├── __init__.py
│   ├── analytical_enricher.py  # LLM Call 3 logic
│   ├── character_enricher.py   # LLM Call 2 logic & image fetching
│   └── movie_data_enricher.py  # LLM Call 1 logic
├── models/
│   ├── __init__.py
│   └── movie_models.py         # Pydantic models for data structures
├── output/                     # Generated files (add to .gitignore if large/private)
│   ├── character_images/       # Downloaded character images
│   ├── clean_movie_database.yaml # The final structured data
│   └── generated_movie_data_raw_log.txt # Raw session log
├── prompts/
│   ├── movie_analytical_data_prompt.txt
│   ├── movie_enrich_chars_relationships_prompt.txt
│   └── movie_initial_data_prompt.txt
├── utils/
│   ├── __init__.py
│   └── helpers.py              # Utility functions (YAML, logging, tokens)
├── .env.example                # Example environment variables
├── .gitignore
├── main_orchestrator.py        # Main script to run the pipeline
├── poetry.lock
├── pyproject.toml
└── README.md
```

### Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rurounigit/movie_data_builder.git
    cd movie_enrichment_project
    ```

2.  **Install Dependencies:**
    This project uses [Poetry](https://python-poetry.org/) for dependency management.
    ```bash
    poetry install
    ```

3.  **Set up Environment Variables:**
    *   Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your API keys and LLM server details:
        ```
        OMDB_API_KEY="your_omdb_api_key"
        TMDB_API_KEY="your_tmdb_api_key"
        LLM_BASE_URL="http://localhost:1234/v1" # Default for LM Studio
        LLM_API_KEY="lm-studio" # Or your actual API key if required by the server
        ```
        *   Get an OMDB API key from [omdbapi.com](http://www.omdbapi.com/apikey.aspx).
        *   Get a TMDB API key by signing up at [themoviedb.org](https://www.themoviedb.org/documentation/api) (use an "API Read Access Token v4 Auth").

4.  **Set up Local LLM Server:**
    *   This project is designed to work with a local LLM server that exposes an OpenAI-compatible API (e.g., [LM Studio](https://lmstudio.ai/), Ollama with a compatible frontend, etc.).
    *   Download and run LM Studio (or your preferred server).
    *   Download a model compatible with your tasks (e.g., a good instruction-tuned model like a Mixtral variant, Llama, Gemma, etc.).
    *   Start the server in LM Studio (usually on `http://localhost:1234/v1`). Make sure the model you want to use is loaded.

5.  **Configure `main_config.yaml`:**
    *   Review and adjust settings in `configs/main_config.yaml` as needed:
        *   `output_file`: Path to the main YAML database.
        *   `raw_log_file`: Path to the raw log.
        *   `character_image_save_path`: Directory for character images.
        *   `default_llm_model_id`: The model identifier your LLM server uses (e.g., `gemma-3-12b-it-qat`, `NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO`, etc. – this must match what your local LLM server expects).
        *   `num_new_movies_to_fetch_this_session`: How many new movies to process.
        *   `max_tmdb_top_rated_pages_to_check`: How many pages of TMDB top-rated movies to iterate through.
        *   `update_existing_movies`: Boolean to enable/disable updating existing movie entries.
        *   `keys_to_update_for_existing`: Specific keys to update if `update_existing_movies` is true (empty list means update all active enricher fields).
        *   `active_enrichers`: Booleans to toggle different enrichment stages.
        *   Token and character limits for API calls and LLM prompts.

6.  **Review Prompts:**
    *   The prompts in the `prompts/` directory are crucial for the quality of LLM-generated data. You may want to customize them for your chosen LLM or desired output style.

### Running the Pipeline

Once set up, run the main orchestrator:

```bash
poetry run python main_orchestrator.py
```

The script will log its progress to the console and to the `raw_log_file` specified in the config. The enriched movie data will be saved to the `output_file`.

### How to Add a New Data Point

Adding a new data point to the system involves several steps, touching different parts of the project:

1.  **Define in Pydantic Model (`models/movie_models.py`):**
    *   Decide which Pydantic model the new data point belongs to (e.g., `MovieEntry`, `LLMCall1Output`, `CharacterListItem`, etc.).
    *   Add the new field to the chosen Pydantic model with its type hint (e.g., `new_data_point: Optional[str] = None`).
    *   If it's a complex type (like a list of objects or a nested object), define a new Pydantic model for it first and then use that new model as the type.

2.  **Update LLM Prompt(s):**
    *   If the new data point is to be generated by an LLM:
        *   Identify which LLM call (1, 2, or 3) is most appropriate for generating this data.
        *   Modify the corresponding prompt file in the `prompts/` directory.
        *   Clearly instruct the LLM on what the new data point is, what format it should be in, and provide an example if necessary.
        *   Remember to escape any literal curly braces `{{ }}` in your examples if they are not `str.format()` placeholders.

3.  **Update Enricher Function (`enrichers/`):**
    *   If generated by an LLM, the enricher function that makes the relevant LLM call needs to be aware of this new data point.
    *   The Pydantic model used to validate the LLM's output for that call (e.g., `LLMCall1Output`, `LLMCall2Output`, `LLMCall3Output`) must now include the new field (as done in step 1).
    *   If the LLM returns the data directly, no further change might be needed in the enricher if the Pydantic model handles it.
    *   If the LLM's output for this new field needs transformation before Pydantic validation (e.g., converting a string to a list), add that transformation logic within the enricher function (similar to how `recommendations` or `genre_mix` are handled in `analytical_enricher.py`).

4.  **Update Orchestrator (`main_orchestrator.py`):**
    *   Ensure the new data point is correctly passed from the enricher's output model (e.g., `llm1_data_generated`) to the `working_data_dict`.
        ```python
        if llmX_data_generated:
            # Ensure the new field is part of the model_dump or handle it specifically
            working_data_dict.update(llmX_data_generated.model_dump(exclude_none=False))
        ```
    *   If the new data point is part of `MovieEntry` itself (and not just an intermediate LLM output model), ensure `MovieEntry` in `movie_models.py` has the field.
    *   If the new data point affects IMDb ID fetching (e.g., a new type of related movie), update the IMDb ID fetching logic for it.

5.  **Update Configuration (`configs/main_config.yaml`) (Optional):**
    *   If the new data point requires new configuration parameters (e.g., max tokens specifically for it, a toggle to enable/disable its generation), add these to `main_config.yaml`.
    *   Update `key_to_enricher_group_map` in `main_orchestrator.py` if the new field belongs to a specific enricher group and you want it to be controllable via `keys_to_update_for_existing`.

6.  **Testing:**
    *   Thoroughly test with a single new movie to ensure the data is generated, parsed, validated, and saved correctly.
    *   Check the logs for any errors or warnings related to the new data point.

**Example: Adding a "Mood" data point (LLM-generated, in Call 3)**

1.  **`models/movie_models.py`:**
    ```python
    class LLMCall3Output(BaseModel):
        # ... existing fields ...
        movie_mood: Optional[str] = Field(None, description="The overall mood or atmosphere of the movie, e.g., 'Dark and Gritty', 'Hopeful and Uplifting'.")

    class MovieEntry(BaseModel):
        # ... existing fields ...
        movie_mood: Optional[str] = None
    ```

2.  **`prompts/movie_analytical_data_prompt.txt`:**
    Add instructions for `movie_mood`:
    ```
    ...
    - movie_mood: (string) Describe the overall mood or atmosphere of the movie in a short phrase (e.g., "Dark and Gritty", "Hopeful and Uplifting", "Whimsical and Lighthearted").
    ...
    Example output snippet:
    ...
    movie_mood: "Tense and Suspenseful"
    ...
    ```

3.  **`enrichers/analytical_enricher.py`:**
    *   The `LLMCall3Output` model already includes `movie_mood`. If the LLM returns it correctly as a string under the key `"movie_mood"`, `get_llm_response_and_parse` and subsequent `LLMCall3Output.model_validate(data)` should handle it without extra transformation logic specifically for this field.

4.  **`main_orchestrator.py`:**
    *   In the section for LLM Call 3:
        ```python
        if llm3_output:
            logger.info(f"  Success: Analytical Data for '{movie_title_for_calls}'.")
            working_data_dict.update(llm3_output.model_dump(exclude_none=False)) # This will include movie_mood
        ```
    *   Update `key_to_enricher_group_map` if needed:
        ```python
        key_to_enricher_group_map = {
            # ... existing ...
            "movie_mood": "analytical_data",
            # ... existing ...
        }
        ```

This systematic approach will help you expand the dataset cleanly.

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bugs, feature requests, or improvements.

### License

Apache 2.0


