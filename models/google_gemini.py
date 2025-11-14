import os
from typing import Any, Dict, Optional, Type
from .rate_limit import build_gemini_limiter
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

# Load environment variables from .env if present
load_dotenv()

# Default model can be overridden via env var
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
GEMINI_LIMITER = build_gemini_limiter()

def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError()
    return api_key


def _get_client() -> genai.Client:
    api_key = _get_api_key()
    return genai.Client(api_key=api_key)


def generate_text(
    prompt: str,
    model: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    **gen_kwargs: Any,
) -> str:
    client = _get_client()
    model_name = model or DEFAULT_GEMINI_MODEL

    # 🔒 Respect Gemini RPM (e.g., 15 RPM for Flash-Lite)
    # Uses GEMINI_MAX_RPM env var if set, otherwise defaults to 15.
    GEMINI_LIMITER.for_key(model_name).wait()

    # Base config
    config: Dict[str, Any] = {}
    if extra_config:
        config.update(extra_config)

    # Map generic kwargs into Gemini config
    # e.g. max_tokens (our generic name) -> max_output_tokens (Gemini)
    if "max_tokens" in gen_kwargs and "max_output_tokens" not in config:
        config["max_output_tokens"] = gen_kwargs["max_tokens"]

    # You can pass other knobs like temperature / top_p via gen_kwargs
    # and they'll just land in config:
    # generate_text(prompt, temperature=0.2, top_p=0.8)
    for k, v in gen_kwargs.items():
        # don't overwrite anything the caller explicitly set in extra_config
        config.setdefault(k, v)

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
