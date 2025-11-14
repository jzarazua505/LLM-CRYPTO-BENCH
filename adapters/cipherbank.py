import json
from typing import Dict, Any, Iterable
from .base import DatasetAdapter

class CipherBankAdapter(DatasetAdapter):
    """
    Decrypt known-classical ciphers with explicit algorithm.
    datasets/cipherbank/cipherbank.jsonl
    """

    def iter_items(self) -> Iterable[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield {
                    "id": rec["id"],
                    "dataset": rec.get("dataset", "cipherbank"),
                    "algorithm": rec["algorithm"],
                    "ciphertext": rec["ciphertext"],
                    "plaintext": rec["plaintext"],
                }

    def build_prompt(self, item: Dict[str, Any]) -> str:
        """
        Force the model into the same strict ANSWER: format used in CipherBench,
        but with explicit algorithm + ciphertext instead of a puzzle-style prompt.
        """
        algo = item["algorithm"]
        ctxt = item["ciphertext"]

        return (
            "You are a strict decoder.\n"
            "You MUST follow these rules exactly:\n"
            "1. DO NOT show your reasoning.\n"
            "2. DO NOT explain what you are doing.\n"
            "3. Your ENTIRE response must be exactly ONE line.\n"
            "4. That line must start with 'ANSWER:' followed by a space and the decoded plaintext.\n"
            "5. If you output anything other than this single ANSWER line, your answer will be considered WRONG.\n"
            "6. Do not add quotes around the plaintext. Do not add extra punctuation.\n\n"
            f"The following text has been encrypted using the {algo} cipher.\n\n"
            "Ciphertext:\n"
            f"{ctxt}\n\n"
            "Now respond in exactly this format:\n"
            "ANSWER: <decoded plaintext>\n"
        )

    def score(self, item: Dict[str, Any], model_output: str) -> int:
        """
        Same two-stage scoring logic as CipherBench:

        1. Try to extract an ANSWER: line and compare strictly to the gold plaintext.
        2. If that fails, but the gold plaintext appears anywhere in the full output,
           still count it as correct (model decoded, just bad formatting).
        """
        text = (model_output or "").strip()
        if not text:
            return 0

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        answer = None

        # Step 1: look for an ANSWER: line (from bottom up)
        for line in reversed(lines):
            lowered = line.lower()
            if lowered.startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
                break

        if answer is None and lines:
            # Fallback candidate = last non-empty line
            answer = lines[-1]

        gold = item["plaintext"].strip()

        # Strict check
        if answer == gold:
            return 1

        # Step 2: lenient substring fallback
        if gold in text:
            return 1

        return 0
