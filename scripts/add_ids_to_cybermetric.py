import json
import os

# Path setup (adjusted to your structure)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH  = os.path.join(ROOT_DIR, "datasets", "cybermetric_mcq", "CyberMetric-crypto.json")
OUT_PATH = os.path.join(ROOT_DIR, "datasets", "cybermetric_mcq", "CyberMetric-crypto_with_ids.json")

with open(IN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = data.get("questions", [])
for idx, q in enumerate(questions):
    q["id"] = f"cybermetric/{idx:06d}"

data["questions"] = questions

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[OK] Added IDs to {len(questions)} questions.")
print(f"[OK] New file saved to {OUT_PATH}")
