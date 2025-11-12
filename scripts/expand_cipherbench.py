import json
import os

# Run this script from the project root:
#   python scripts/expand_cipherbench.py

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH = os.path.join(ROOT_DIR, "datasets", "cipherbench", "cipherbench.jsonl")
OUT_PATH = os.path.join(ROOT_DIR, "datasets", "cipherbench", "cipherbench_expanded.jsonl")

ALGO_KEYS = [
    "base_64",
    "rot_13",
    "pig_latin",
    "leetspeak",
    "keyboard",
    "upside_down",
    "word_reversal",
    "word_substitution",
    "grid_encoding",
    "art_ascii",
]

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

if not os.path.exists(IN_PATH):
    raise FileNotFoundError(f"Input file not found at: {IN_PATH}")

count_total = 0

with open(IN_PATH, "r", encoding="utf-8") as fin, \
     open(OUT_PATH, "w", encoding="utf-8") as fout:

    for idx, line in enumerate(fin):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)

        base_sentence = rec["sentence"]

        for algo in ALGO_KEYS:
            if algo not in rec:
                continue
            prompt_text = rec[algo]
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                continue

            out = {
                "id": f"cipherbench/{algo}/{idx:06d}",
                "dataset": "cipherbench",
                "algorithm": algo,
                "prompt_text": prompt_text,
                "plaintext": base_sentence,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count_total += 1

print(f"[OK] Wrote {count_total} expanded items to {OUT_PATH}")
