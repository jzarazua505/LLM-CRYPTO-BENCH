# models/openrouter_gptoss.py
from .openrouter_client import generate_text_openrouter

MODEL_ID = "openai/gpt-oss-20b:free"

def generate_text(prompt: str, **gen_kwargs) -> str:
    default_params = {
        "temperature": 0.8,
        "top_p": 1.0,
    }
    default_params.update(gen_kwargs)

    return generate_text_openrouter(
        MODEL_ID,
        prompt,
        extra_params=default_params,
        max_retries=10,
        min_backoff=2.0,
        max_backoff=60.0,
    )

