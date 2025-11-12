import os
from typing import Any, Dict, Optional, Type

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

# Load environment variables from .env if present
load_dotenv()

# Default model can be overridden via env var
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")



def _get_api_key() -> str:
    """
    Resolve the Gemini API key.

    Priority:
    1. GEMINI_API_KEY (recommended; matches official docs)
    2. GOOGLE_API_KEY (fallback if someone uses old naming)
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment. "
            "Set it in your .env or export it before running."
        )
    return api_key


def _get_client() -> genai.Client:
    """
    Create a Google GenAI client using the configured API key.
    """
    api_key = _get_api_key()
    return genai.Client(api_key=api_key)


def generate_text(
    prompt: str,
    model: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a plain text response from Gemini using the new Google GenAI SDK.

    This is the primary helper for benchmark runs where we only need the
    model's final answer as text.

    :param prompt: Input prompt string.
    :param model: Optional model name (defaults to DEFAULT_GEMINI_MODEL).
    :param extra_config: Optional dict merged into the request `config`.
    :return: Response text from the model.
    """
    client = _get_client()
    model_name = model or DEFAULT_GEMINI_MODEL

    config: Dict[str, Any] = {}
    if extra_config:
        config.update(extra_config)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config or None,
    )

    # The SDK attaches the combined text on .text; fall back defensively.
    if getattr(response, "text", None):
        return response.text

    # Fallback: reconstruct from candidates if needed
    try:
        candidate = response.candidates[0]
        parts = getattr(candidate, "content", candidate).parts
        return "".join(getattr(p, "text", "") for p in parts if hasattr(p, "text"))
    except Exception:
        return str(response)


def generate_json(
    prompt: str,
    schema: Type[BaseModel],
    model: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    """
    Generate a **structured JSON** response parsed into a Pydantic model.

    This is useful for consistent benchmark outputs (e.g., MCQ choice, rationale).

    Example usage:
        class Answer(BaseModel):
            answer: str
            reasoning: str

        result = generate_json(prompt, Answer)
        print(result.answer, result.reasoning)

    :param prompt: Input prompt.
    :param schema: Pydantic BaseModel subclass representing expected JSON shape.
    :param model: Optional model name.
    :param extra_config: Optional config overrides.
    :return: An instance of `schema` parsed from the model's response.
    """
    client = _get_client()
    model_name = model or DEFAULT_GEMINI_MODEL

    config: Dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_schema": schema,
    }
    if extra_config:
        config.update(extra_config)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    # With response_schema set, the SDK exposes .parsed as the Pydantic object.
    if hasattr(response, "parsed") and isinstance(response.parsed, schema):
        return response.parsed

    # Fallback if something changes upstream: try to coerce manually.
    if hasattr(response, "parsed"):
        return schema.parse_obj(response.parsed)  # type: ignore[arg-type]

    # Last resort: try to parse from .text
    return schema.parse_raw(getattr(response, "text", "{}"))
