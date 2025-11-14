# models/rate_limit.py
import os, time, threading
from typing import Dict

class TokenBucket:
    def __init__(self, rpm: float):
        self.capacity = max(float(rpm), 1.0)
        self.tokens = self.capacity
        self.refill_rate = self.capacity / 60.0  # tokens per second
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            # need to wait until 1 full token available
            need = 1.0 - self.tokens
            delay = need / self.refill_rate
        time.sleep(delay)

class RateLimiter:
    def __init__(self, default_rpm: float):
        self.default = TokenBucket(default_rpm)
        self.by_key: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    @staticmethod
    def _norm(s: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in s)

    def for_key(self, key: str) -> TokenBucket:
        k = self._norm(key)
        with self.lock:
            if k in self.by_key:
                return self.by_key[k]
            # per-model override
            per_model = os.getenv(f"OPENROUTER_RPM_MODEL_{k}")
            if per_model:
                tb = TokenBucket(float(per_model))
                self.by_key[k] = tb
                return tb
            # per-provider override (prefix before first slash)
            provider = k.split("_", 1)[0] if "_" in k else k
            per_provider = os.getenv(f"OPENROUTER_RPM_PROVIDER_{provider}")
            if per_provider:
                tb = TokenBucket(float(per_provider))
                self.by_key[k] = tb
                return tb
            # default
            self.by_key[k] = self.default
            return self.default

def build_global_limiter() -> RateLimiter:
    rpm = float(os.getenv("OPENROUTER_MAX_RPM", "12"))
    return RateLimiter(rpm)

def build_gemini_limiter() -> RateLimiter:
    rpm = float(os.getenv("GEMINI_MAX_RPM", "15"))
    return RateLimiter(rpm)