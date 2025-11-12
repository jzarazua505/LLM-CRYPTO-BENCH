import os
import json
import argparse
from typing import Tuple, Type

from adapters.base import DatasetAdapter
from adapters.cybermetric import CyberMetricAdapter
from adapters.cipherbank import CipherBankAdapter
from adapters.cipherbench import CipherBenchAdapter
from models.base import get_model

# Dataset registry: name -> (AdapterClass, path)
DATASETS = {
    "cybermetric": (
        CyberMetricAdapter,
        os.path.join("datasets", "cybermetric", "cybermetric.json"),
    ),
    "cipherbank": (
        CipherBankAdapter,
        os.path.join("datasets", "cipherbank", "cipherbank.jsonl"),
    ),
    "cipherbench": (
        CipherBenchAdapter,
        os.path.join("datasets", "cipherbench", "cipherbench.jsonl"),
    ),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        required=True,
        help="Which dataset to evaluate on."
    )
    p.add_argument(
        "--model",
        required=True,
        help='Model name (e.g. "echo", "gemini-1.5-flash", "deepseek-v3", etc.)'
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: limit number of items for quick testing."
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional: custom output path."
    )
    return p.parse_args()

def main():
    args = parse_args()

    AdapterCls, ds_path = DATASETS[args.dataset]
    adapter: DatasetAdapter = AdapterCls(ds_path)

    model = get_model(args.model)

    os.makedirs("results", exist_ok=True)
    out_path = args.out or os.path.join(
        "results",
        f"{args.dataset}__{model.name}.jsonl"
    )

    total = 0
    correct = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, item in enumerate(adapter.iter_items()):
            if args.limit is not None and i >= args.limit:
                break

            prompt = adapter.build_prompt(item)
            # Call model
            output = model.generate(prompt)

            label = adapter.score(item, output)
            correct += label
            total += 1

            rec = {
                "id": item["id"],
                "dataset": args.dataset,
                "model": model.name,
                "prompt": prompt,
                "output": output,
                "correct": label,
            }
            # Include extra metadata if present
            for key in ("algorithm", "ciphertext", "prompt_text", "question"):
                if key in item:
                    rec[key] = item[key]

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    acc = correct / total if total else 0.0
    print(f"Evaluated {total} items on {args.dataset} with {model.name}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
