import json
import re
from typing import Dict, Any, Iterable
from .base import DatasetAdapter

class CyberMetricAdapter(DatasetAdapter):
    """
    Multiple-choice crypto questions.
    datasets/cybermetric/cybermetric.json
    """

    def iter_items(self) -> Iterable[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for q in data.get("questions", []):
            qid = q.get("id")
            if not qid:
                # Fallback if some entries somehow miss id
                idx = id(q)  # won't really happen if you wrote ids properly
                qid = f"cybermetric/{idx:06d}"

            yield {
                "id": qid,
                "question": q["question"],
                "answers": q["answers"],
                "solution": q["solution"].strip().upper(),
            }

    def build_prompt(self, item: Dict[str, Any]) -> str:
        lines = [
            "You are a cryptography expert.",
            "",
            item["question"],
            "",
            "Choices:",
        ]
        for letter, text in item["answers"].items():
            lines.append(f"{letter}: {text}")
        lines.append("")
        lines.append("Answer with only one letter: A, B, C, or D.")
        return "\n".join(lines)

    def score(self, item: Dict[str, Any], model_output: str) -> int:
        text = (model_output or "").strip().upper()
        m = re.search(r"\b([ABCD])\b", text)
        pred = m.group(1) if m else ""
        return int(pred == item["solution"])
