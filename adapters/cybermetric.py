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

        questions = data.get("questions", [])
        for idx, q in enumerate(questions):
            qid = q.get("id")
            if not qid:
                # Fallback if some entries somehow miss id
                qid = f"cybermetric/{idx:06d}"

            yield {
                "id": qid,
                "dataset": q.get("dataset", "cybermetric"),
                "question": q["question"],
                "answers": q["answers"],
                "solution": q["solution"].strip().upper(),
            }

    def build_prompt(self, item: Dict[str, Any]) -> str:
        """
        Encourage a strict single-letter ANSWER format, similar in spirit
        to the decryption adapters, but for multiple-choice.
        """
        lines = [
            "You are a cryptography expert.",
            "",
            "You MUST follow these rules exactly:",
            "1. Carefully read the question and all answer choices.",
            "2. DO NOT show your reasoning.",
            "3. DO NOT explain your answer.",
            "4. Your ENTIRE response must be exactly ONE line.",
            "5. That line must start with 'ANSWER:' followed by a space and a single letter.",
            "6. The letter must be one of: A, B, C, or D.",
            "7. If you output anything other than this single ANSWER line, your answer will be considered WRONG.",
            "",
            "Question:",
            item["question"],
            "",
            "Choices:",
        ]
        for letter, text in item["answers"].items():
            lines.append(f"{letter}: {text}")

        lines.append("")
        lines.append("Now respond in exactly this format:")
        lines.append("ANSWER: <one letter: A, B, C, or D>")
        return "\n".join(lines)

    def score(self, item: Dict[str, Any], model_output: str) -> int:
        """
        Two-stage scoring:

        1. Prefer an ANSWER: line (mirroring the decryption adapters).
        2. If none is found, fall back to regex for a bare A/B/C/D anywhere.

        This lets us be strict in *intent* but robust to minor formatting issues.
        """
        text = (model_output or "").strip().upper()
        if not text:
            return 0

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        answer_letter = None

        # ----- Step 1: try to extract from an ANSWER: line -----
        for line in reversed(lines):
            lowered = line.lower()
            if lowered.startswith("answer:"):
                # Grab everything after "ANSWER:" and look for A/B/C/D
                tail = line.split(":", 1)[1].strip().upper()
                m = re.search(r"\b([ABCD])\b", tail)
                if m:
                    answer_letter = m.group(1)
                break

        # ----- Step 2: fallback to regex anywhere in the text -----
        if answer_letter is None:
            m = re.search(r"\b([ABCD])\b", text)
            answer_letter = m.group(1) if m else ""

        gold = item["solution"]
        return int(answer_letter == gold)
