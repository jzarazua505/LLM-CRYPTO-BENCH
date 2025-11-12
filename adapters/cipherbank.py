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
        return (
            "You are a cipher decoder.\n\n"
            f"Algorithm: {item['algorithm']}\n"
            "Ciphertext:\n"
            f"{item['ciphertext']}\n\n"
            "Decrypt it. Output only the decrypted plaintext."
        )

    def score(self, item: Dict[str, Any], model_output: str) -> int:
        pred = (model_output or "").strip()
        gold = item["plaintext"].strip()
        return int(pred == gold)
