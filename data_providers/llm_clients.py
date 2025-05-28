# movie_enrichment_project/data_providers/llm_clients.py
import openai
from typing import List, Dict, Optional, Any
import re
import json
import yaml

def strip_code_fences(raw_text: str) -> str:
    """
    Aggressively strips common code block fences (e.g., ```yaml, ```json, ```)
    and optional language prefixes from the beginning and end of a string.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Iteratively strip fences and language tags in case of nesting or multiple layers
    # This is a bit more aggressive.
    previous_text = ""
    while text != previous_text: # Keep stripping as long as changes are made
        previous_text = text

        # Try to match markdown code blocks ```[lang]\ncontent\n```
        # This regex is designed to capture the content within the outermost fences.
        # It handles optional language specifiers and various spacing.
        fence_match = re.match(r"^\s*```(?:[a-zA-Z0-9\-_]+)?\s*(.*?)\s*```\s*$", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
            continue # Restart stripping on the inner content

        # Fallback: if it starts with ``` but doesn't match the full pattern (e.g. missing closing ```)
        if text.startswith("```"):
            text = text[3:] # Remove opening ```
            # Attempt to remove language specifier if present on the same line or next
            first_line_end = text.find('\n')
            if first_line_end != -1:
                first_line = text[:first_line_end].strip()
                # Check if first_line is a common language specifier
                if re.match(r"^[a-zA-Z0-9\-_]+$", first_line) and len(first_line) < 10: # Heuristic for lang specifier
                    text = text[first_line_end+1:]
            text = text.strip() # Strip leading whitespace from content

        # Fallback: if it ends with ```
        if text.endswith("```"):
            text = text[:-3].strip()

        # Fallback: remove potential leading "json" or "yaml" if they are on their own line or followed by a space
        # This helps if there were no fences but the LLM still prefixed the language.
        if text.lower().startswith("json\n") or text.lower().startswith("yaml\n"):
            text = text.split("\n", 1)[1].strip()
        elif text.lower().startswith("json ") or text.lower().startswith("yaml "):
            text = text[5:].strip()

        text = text.strip() # Ensure stripped at each iteration

    return text


def parse_llm_output_to_dict(
    response_content: str,
    logger: Optional[Any] = None,
    context: str = "LLM Response"
) -> Optional[Dict[str, Any]]:
    if not response_content:
        return None

    cleaned_content = strip_code_fences(response_content)

    if not cleaned_content:
        if logger: logger.warning(f"{context}: Content was empty after stripping fences. Original: '{response_content[:100]}...'")
        return None

    # Attempt 1: Parse as JSON
    try:
        data = json.loads(cleaned_content)
        if isinstance(data, dict):
            if logger: logger.debug(f"{context}: Successfully parsed as JSON.")
            return data
        else:
            if logger: logger.warning(f"{context}: Parsed as JSON, but result is not a dictionary (type: {type(data)}). Content: '{cleaned_content[:200]}...'")
    except json.JSONDecodeError as je:
        if logger: logger.debug(f"{context}: Failed to parse as JSON: {je}. Will attempt YAML. Content preview: '{cleaned_content[:200]}...'")
        pass

    # Attempt 2: Parse as YAML
    try:
        data = yaml.safe_load(cleaned_content)
        if isinstance(data, dict):
            if logger: logger.debug(f"{context}: Successfully parsed as YAML.")
            return data
        else:
            if logger: logger.error(f"{context}: Parsed as YAML, but result is not a dictionary (type: {type(data)}). Content: '{cleaned_content[:200]}...'")
            return None
    except yaml.YAMLError as ye:
        # This is where the "found character '`'" error would be caught if stripping failed and '```json' was passed to yaml.safe_load
        if logger: logger.error(f"{context}: YAML parsing error: {ye}. This often means code fences were not stripped or content is not valid YAML/JSON. Cleaned content preview: '{cleaned_content[:200]}...'")
        return None
    except Exception as e:
        if logger: logger.error(f"{context}: Unexpected error during YAML parsing: {e}. Cleaned content: '{cleaned_content[:200]}...'")
        return None

    if logger: logger.error(f"{context}: Could not parse content into a dictionary using JSON or YAML. Cleaned content: '{cleaned_content[:200]}...'")
    return None


def get_llm_response_and_parse(
    client: openai.OpenAI,
    model_id_for_call: str,
    messages_history: List[Dict[str, str]],
    max_tokens: int,
    temperature: float = 0.3,
    logger: Optional[Any] = None,
    parsing_context: str = "LLM Response"
) -> Optional[Dict[str, Any]]:
    if logger:
        logger.debug(f"Sending request to LLM (model: {model_id_for_call}, max_tokens: {max_tokens}, temp: {temperature}). For: {parsing_context}")

    try:
        # IMPORTANT: Check for response_format parameter here.
        # If your OpenAI client or server supports strict JSON mode via response_format,
        # and you've enabled it, it might cause the "JSON schema is missing" error
        # if the schema isn't also provided in the request.
        # For now, ensure response_format is NOT set, or set to None.
        completion_params = {
            "model": model_id_for_call,
            "messages": messages_history,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        # Example: if you had this, remove it or ensure your model and prompt support it:
        # if model_is_known_to_support_json_mode_strictly:
        # completion_params["response_format"] = {"type": "json_object"}

        completion = client.chat.completions.create(**completion_params)

        raw_response_content = completion.choices[0].message.content

        if not raw_response_content or not raw_response_content.strip():
            log_msg = f"{parsing_context}: LLM returned empty or whitespace-only content for model {model_id_for_call}."
            if logger: logger.warning(log_msg)
            return None

        if logger: logger.debug(f"{parsing_context}: Raw LLM response before parsing: '{raw_response_content[:500]}...'")

        return parse_llm_output_to_dict(raw_response_content, logger, context=parsing_context)

    except openai.APIError as e:
        # Check if the error message is about JSON schema
        if e.body and isinstance(e.body, dict) and 'error' in e.body:
            error_detail = e.body['error']
            if isinstance(error_detail, str) and 'JSON schema is missing' in error_detail:
                 message = f"{parsing_context}: OpenAI APIError (model: {model_id_for_call}): {error_detail}. This indicates the model/server is in a strict JSON mode but no schema was provided in the request. Check for `response_format` in API call."
            elif isinstance(error_detail, dict) and 'message' in error_detail and 'JSON schema is missing' in error_detail['message']:
                 message = f"{parsing_context}: OpenAI APIError (model: {model_id_for_call}): {error_detail['message']}. This indicates the model/server is in a strict JSON mode but no schema was provided in the request. Check for `response_format` in API call."
            else:
                message = f"{parsing_context}: OpenAI APIError communicating with LLM (model: {model_id_for_call}): {e}"
        else:
            message = f"{parsing_context}: OpenAI APIError communicating with LLM (model: {model_id_for_call}): {e}"

        if logger: logger.error(message)
    except Exception as e:
        message = f"{parsing_context}: Error communicating with LLM (model: {model_id_for_call}): {e}"
        if logger: logger.error(message)
    return None

# Deprecated get_llm_response remains unchanged but its fence stripping could also use the improved strip_code_fences
def get_llm_response(
    client: openai.OpenAI,
    model_id_for_call: str,
    messages_history: List[Dict[str, str]],
    max_tokens: int,
    temperature: float = 0.3,
    attempt_yaml_cleanup: bool = True,
    logger: Optional[Any] = None
) -> Optional[str]:
    if logger:
        logger.debug(f"Sending request to LLM (model: {model_id_for_call}, max_tokens: {max_tokens}, temp: {temperature})... (Using DEPRECATED get_llm_response)")

    try:
        completion_params = {
            "model": model_id_for_call,
            "messages": messages_history,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        # Ensure no response_format forcing JSON mode here either
        completion = client.chat.completions.create(**completion_params)

        response_content = completion.choices[0].message.content

        if response_content:
             response_content = response_content.strip()
        else:
            log_msg = f"LLM returned empty content for model {model_id_for_call} (via deprecated get_llm_response)."
            if logger: logger.warning(log_msg)
            return None

        if attempt_yaml_cleanup and response_content:
            response_content = strip_code_fences(response_content) # Use the improved stripper

        return response_content

    except openai.APIError as e:
        message = f"OpenAI APIError communicating with LLM (model: {model_id_for_call}, via deprecated get_llm_response): {e}"
        if logger: logger.error(message)
    except Exception as e:
        message = f"Error communicating with LLM (model: {model_id_for_call}, via deprecated get_llm_response): {e}"
        if logger: logger.error(message)
    return None