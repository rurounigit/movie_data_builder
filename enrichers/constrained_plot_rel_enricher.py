# enrichers/constrained_plot_rel_enricher.py
from typing import Optional, List, Dict, Any
import openai
import yaml # To dump relationships list to YAML string for the prompt
from models.movie_models import LLMConstrainedPlotWithRelationsOutput, Relationship # Assuming Relationship is model from call 2
from data_providers.llm_clients import get_llm_response_and_parse

def generate_constrained_plot_with_relations(
    llm_client: openai.OpenAI,
    llm_model_id: str,
    movie_title: str,
    movie_year: str,
    tmdb_character_names: List[str],       # List of raw TMDB names
    relationships: List[Relationship],    # List of Relationship Pydantic models from LLM Call 2
    prompt_template: str,
    max_tokens: int,
    logger: Optional[Any] = None
) -> Optional[LLMConstrainedPlotWithRelationsOutput]:

    if not tmdb_character_names:
        if logger: logger.info(f"No TMDB character names for constrained plot (w/ relations) of '{movie_title}'.")
        return LLMConstrainedPlotWithRelationsOutput(plot_with_character_constraints_and_relations=None)

    # Convert relationship models to dicts then to YAML string for the prompt
    relationships_dict_list = [rel.model_dump(exclude_none=True) for rel in relationships]
    relationships_yaml_str = yaml.dump(relationships_dict_list, sort_keys=False, allow_unicode=True, indent=2) if relationships_dict_list else "No specific relationships provided."

    character_list_str_for_prompt = "- " + "\n- ".join(tmdb_character_names)

    prompt_user_content = prompt_template.format(
        movie_title=movie_title,
        movie_year=movie_year,
        tmdb_character_name_list_str=character_list_str_for_prompt,
        relationships_yaml_str=relationships_yaml_str
    )

    messages = [
        {"role": "system", "content": "You write plot descriptions strictly adhering to character naming constraints, using provided relationship context."},
        {"role": "user", "content": prompt_user_content}
    ]

    parsing_context = f"LLM Constrained Plot (w/ Relations) for '{movie_title}'"
    parsed_data = get_llm_response_and_parse(
        client=llm_client,
        model_id_for_call=llm_model_id,
        messages_history=messages,
        max_tokens=max_tokens,
        temperature=0.6,
        logger=logger,
        parsing_context=parsing_context
    )

    if parsed_data and "plot_with_character_constraints_and_relations" in parsed_data:
        try:
            return LLMConstrainedPlotWithRelationsOutput.model_validate(parsed_data)
        except Exception as e:
            if logger: logger.error(f"Pydantic validation for LLMConstrainedPlotWithRelationsOutput for '{movie_title}' failed: {e}. Data: {parsed_data}")
            return None
    elif parsed_data:
         if logger: logger.warning(f"LLM for constrained plot (w/ rel) for '{movie_title}' response missing key. Data: {parsed_data}")
         return LLMConstrainedPlotWithRelationsOutput(plot_with_character_constraints_and_relations=None)
    else:
        if logger: logger.error(f"Failed to get/parse LLM response for constrained plot (w/ rel) for '{movie_title}'.")
        return LLMConstrainedPlotWithRelationsOutput(plot_with_character_constraints_and_relations=None)