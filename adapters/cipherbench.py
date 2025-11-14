import json
from typing import Dict, Any, Iterable
from .base import DatasetAdapter


class CipherBenchAdapter(DatasetAdapter):
    """
    Instruction-following / puzzle-style decryption.
    Expects expanded JSONL:
      { id, dataset, algorithm, prompt_text, plaintext }
    datasets/cipherbench/cipherbench.jsonl
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
                    "dataset": rec.get("dataset", "cipherbench"),
                    "algorithm": rec["algorithm"],
                    "prompt_text": rec["prompt_text"],
                    "plaintext": rec["plaintext"],
                }

    def build_prompt(self, item: Dict[str, Any]) -> str:
        """
        Force the model into a single-line ANSWER: format.
        We explicitly forbid explanations to reduce 'chatty' behavior,
        which open-source models tend to default to.
        """
        return (
            "You are a strict decoder.\n"
            "You MUST follow these rules exactly:\n"
            "1. DO NOT show your reasoning.\n"
            "2. DO NOT explain what you are doing.\n"
            "3. Your ENTIRE response must be exactly ONE line.\n"
            "4. That line must start with 'ANSWER:' followed by a space and the decoded plaintext.\n"
            "5. If you output anything other than this single ANSWER line, your answer will be considered WRONG.\n"
            "6. Do not add quotes around the plaintext. Do not add extra punctuation.\n\n"
            f"{item['prompt_text'].rstrip()}\n\n"
            "Now respond in exactly this format:\n"
            "ANSWER: <decoded plaintext>\n"
        )


    def score(self, item: Dict[str, Any], model_output: str) -> int:
        """
        Two-stage scoring:

        1. Strict: look for an ANSWER: line and require exact match.
        2. Lenient fallback: if the gold plaintext appears anywhere in the output
        (e.g., inside a rambly explanation), still count it as correct.

        This keeps us strict on *content* but not on *format*.
        """
        text = (model_output or "").strip()
        if not text:
            return 0

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        answer = None

        # ---------- Step 1: strict ANSWER: line ----------
        for line in reversed(lines):
            lowered = line.lower()
            if lowered.startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
                break

        if answer is None and lines:
            # Fallback candidate = last non-empty line
            answer = lines[-1]

        gold = item["plaintext"].strip()

        # Strict check first
        if answer == gold:
            return 1

        # ---------- Step 2: lenient substring fallback ----------
        # If the model's full text contains the exact gold plaintext anywhere,
        # give it credit (it *did* decode correctly, just failed the format).
        if gold in text:
            return 1

        return 0
