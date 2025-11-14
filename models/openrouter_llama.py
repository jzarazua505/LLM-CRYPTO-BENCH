# models/openrouter_llama.py
from .openrouter_client import generate_text_openrouter

MODEL_ID = "meta-llama/llama-3.3-70b-instruct:free"

def generate_text(prompt: str, **gen_kwargs) -> str:
    return generate_text_openrouter(
        MODEL_ID,
        prompt,
        extra_params=gen_kwargs,
    )
