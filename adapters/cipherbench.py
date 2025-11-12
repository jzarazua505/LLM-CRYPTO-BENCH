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
        return (
            f"{item['prompt_text'].rstrip()}\n\n"
            "Output only the decoded plaintext. Do not add explanations."
        )

    def score(self, item: Dict[str, Any], model_output: str) -> int:
        pred = (model_output or "").strip()
        gold = item["plaintext"].strip()
        return int(pred == gold)
