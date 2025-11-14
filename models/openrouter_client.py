# models/openrouter_client.py
from __future__ import annotations

import os
import json
import time
import random
from typing import Any, Dict, Optional, Type, List, Union

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from .rate_limit import build_global_limiter

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Global/per-provider/per-model rate limiter
_RATES = build_global_limiter()  # looks at OPENROUTER_MAX_RPM, OPENROUTER_RPM_PROVIDER_*, OPENROUTER_RPM_MODEL_*

def _get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY in environment. "
            "Set it in your .env or export it before running."
        )
    return api_key

def _build_headers() -> Dict[str, str]:
    """
    OpenRouter recommends including Referer and X-Title headers for app attribution.
    """
    api_key = _get_openrouter_api_key()
    referer = os.getenv("OPENROUTER_SITE_URL", "https://github.com/jzarazua505/LLM-CRYPTO-BENCH")
    app_name = os.getenv("OPENROUTER_APP_NAME", "LLM-CRYPTO-BENCH")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": app_name,
    }

def _extract_text(data: Dict[str, Any]) -> str:
    """
    Robustly extract assistant content from OpenRouter chat.completions response.
    """
    try:
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content", "")

        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts: List[str] = []
            for seg in content:
                if isinstance(seg, dict):
                    if isinstance(seg.get("text"), str):
                        parts.append(seg["text"])
                    elif isinstance(seg.get("content"), str):
                        parts.append(seg["content"])
                elif isinstance(seg, str):
                    parts.append(seg)
            text = "".join(parts).strip()
        else:
            text = str(content).strip()

        if not text:
            # Fallbacks seen in some providers
            alt = choice.get("text") or msg.get("reasoning") or ""
            if isinstance(alt, str) and alt.strip():
                text = alt.strip()

        return text if text else f"[Empty content] Raw: {json.dumps(data)[:500]}..."
    except Exception:
        return f"[Unexpected format] Raw: {json.dumps(data)[:500]}..."

def _norm_key(s: str) -> str:
    """Normalize model id like 'deepseek/deepseek-chat-v3-0324:free' → 'deepseek_deepseek_chat_v3_0324_free'."""
    return "".join(ch if ch.isalnum() else "_" for ch in s)

def _rate_for(model_id: str):
    """Return a token bucket for this model id (model override > provider override > global)."""
    # build_global_limiter handles env precedence internally, keyed by normalized model/provider
    return _RATES.for_key(_norm_key(model_id))

def generate_text_openrouter(
    model: str,
    prompt_or_messages: Union[str, List[Dict[str, Any]]],
    extra_params: Optional[Dict[str, Any]] = None,
    *,
    timeout: int = 60,
    max_retries: int = 8,
    min_backoff: float = 2.0,
    max_backoff: float = 32.0,
    return_on_429: bool = True,
) -> str:

    headers = _build_headers()

    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = prompt_or_messages

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.1,
        "max_tokens": 256,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    if extra_params:
        body.update(extra_params)

    attempt = 0
    backoff = min_backoff

    while True:
        _rate_for(model).wait()  # throttle for this model/provider/global

        resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=body, timeout=timeout)

        # Success
        if resp.status_code < 400:
            return _extract_text(resp.json())

        # 429 → try to sleep smartly & retry
        if resp.status_code == 429 and attempt < max_retries:
            attempt += 1

            # default backoff baseline
            base = backoff

            # Prefer server-provided guidance if present
            retry_after = resp.headers.get("Retry-After")
            rl_reset = resp.headers.get("x-ratelimit-reset")

            if retry_after:
                try:
                    base = max(base, float(retry_after))
                except ValueError:
                    pass
            elif rl_reset:
                try:
                    reset_ts = float(rl_reset)
                    now = time.time()
                    if reset_ts > now:
                        base = max(base, reset_ts - now)
                except ValueError:
                    pass
            else:
                # No headers → likely provider DDOS/burst protection.
                # Enforce a stronger floor from env (default 20s).
                headerless_floor = float(os.getenv("OPENROUTER_429_FLOOR_SECONDS", "20"))
                base = max(base, headerless_floor)

            # jitter to avoid synchronized retries
            sleep_s = max(0.0, base + random.uniform(-0.2 * base, 0.2 * base))

            # After a few strikes, extend cooldown aggressively (e.g., ≥60s)
            if attempt >= 3:
                sleep_s = max(sleep_s, float(os.getenv("OPENROUTER_429_COOLDOWN_SECONDS", "60")))

            # escalate backoff for next time
            backoff = min(backoff * 1.8, max_backoff)

            try:
                hint = (resp.json().get("error", {}) or {}).get("message", "")
            except Exception:
                hint = resp.text[:200]
            print(f"[OpenRouter 429] Retry {attempt}/{max_retries} in {sleep_s:.1f}s — {hint or 'rate limited (no headers)'}")
            time.sleep(sleep_s)
            continue

        # 429 but exhausted retries
        if resp.status_code == 429 and attempt >= max_retries:
            if return_on_429:
                return "[Rate-limited: exhausted retries on free model]"
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            resp.raise_for_status()

        # Other HTTP errors → raise
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        resp.raise_for_status()
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {detail}")

def generate_json_openrouter(
    model: str,
    prompt: str,
    schema: Type[BaseModel],
    extra_params: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    """
    Ask the model to return valid JSON matching a given Pydantic schema.
    """
    json_prompt = (
        "You are a scoring assistant. Respond ONLY with valid JSON matching this schema:\n"
        f"{schema.model_json_schema()}\n"
        "Now follow the task:\n"
        f"{prompt}"
    )

    text = generate_text_openrouter(model, json_prompt, extra_params=extra_params)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json", "", 1).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON:\n{text}") from e

    return schema.model_validate(data)
