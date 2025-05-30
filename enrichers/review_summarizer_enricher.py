# enrichers/review_summarizer_enricher.py
from typing import Optional, List, Dict, Any
import openai
from models.movie_models import LLMReviewSummaryOutput
from data_providers.llm_clients import get_llm_response_and_parse

def generate_tmdb_review_summary(
    llm_client: openai.OpenAI,
    llm_model_id: str,
    movie_title: str,
    movie_year: str,
    review_snippets: List[str], # List of review content strings
    prompt_template: str,
    max_tokens: int,
    logger: Optional[Any] = None
) -> Optional[LLMReviewSummaryOutput]:
    if not review_snippets:
        if logger: logger.info(f"No review snippets provided for '{movie_title}' to summarize.")
        return LLMReviewSummaryOutput(tmdb_user_review_summary=None) # Return model with None if no reviews

    # Join snippets into a single string for the prompt
    formatted_reviews = "\n\n".join(review_snippets)

    prompt_user_content = prompt_template.format(
        movie_title=movie_title,
        movie_year=movie_year,
        tmdb_review_snippets=formatted_reviews
    )

    messages = [
        {"role": "system", "content": "You are an expert at summarizing movie reviews neutrally and concisely."},
        {"role": "user", "content": prompt_user_content}
    ]

    parsing_context = f"LLM Review Summary for '{movie_title}'"
    parsed_data = get_llm_response_and_parse(
        client=llm_client,
        model_id_for_call=llm_model_id,
        messages_history=messages,
        max_tokens=max_tokens,
        temperature=0.5, # Slightly higher for summarization
        logger=logger,
        parsing_context=parsing_context
    )

    if parsed_data and "tmdb_user_review_summary" in parsed_data:
        try:
            return LLMReviewSummaryOutput.model_validate(parsed_data)
        except Exception as e:
            if logger: logger.error(f"Pydantic validation failed for LLMReviewSummaryOutput for '{movie_title}': {e}. Data: {parsed_data}")
            return None # Or return model with None summary
    elif parsed_data: # LLM responded with a dict, but not the expected key
         if logger: logger.warning(f"LLM for review summary for '{movie_title}' responded with a dict, but missing 'tmdb_user_review_summary' key. Data: {parsed_data}")
         return LLMReviewSummaryOutput(tmdb_user_review_summary=None)
    else: # LLM did not respond or parsing failed
        if logger: logger.error(f"Failed to get or parse LLM response for review summary for '{movie_title}'.")
        return LLMReviewSummaryOutput(tmdb_user_review_summary=None)