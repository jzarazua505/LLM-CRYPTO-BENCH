# models/openrouter_client.py

import os
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_openrouter(model: str, prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional:
        # "HTTP-Referer": "https://github.com/your-org/llm-crypto-bench",
        # "X-Title": "llm-crypto-bench",
    }

    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "top_p": 1.0,
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected OpenRouter response: {data}")

    return text.strip()
