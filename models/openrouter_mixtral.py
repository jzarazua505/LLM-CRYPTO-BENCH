# models/openrouter_mixtral.py
from .openrouter_client import generate_text_openrouter

MODEL_ID = "mistralai/mistral-7b-instruct:free"

def generate_text(prompt: str, **gen_kwargs) -> str:
    return generate_text_openrouter(
        MODEL_ID,
        prompt,
        extra_params=gen_kwargs,
    )
