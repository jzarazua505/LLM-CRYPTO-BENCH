import os
import json
import time
import argparse
from typing import Tuple, Type, Dict, Any

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
        help='Model name (e.g. "gemini-2.5-pro", "gpt-oss-20b", "llama-3.3-70b-instruct", "mistral-7b-instruct")'
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
    p.add_argument(
        "--progress_every",
        type=int,
        default=25,
        help="Print progress every N items."
    )
    return p.parse_args()

def classify_output(raw: str) -> Dict[str, Any]:
    """
    Normalize model output into a status + text payload without throwing.
    """
    if not isinstance(raw, str):
        return {"status": "error", "output": "", "error_message": f"Non-string output: {type(raw)}"}
    s = raw.strip()
    if s.startswith("[Rate-limited"):
        return {"status": "rate_limited", "output": ""}
    if s.startswith("[Empty content]") or s.startswith("[Unexpected format]"):
        return {"status": "empty", "output": ""}
    return {"status": "ok", "output": s}


def _is_blank(s: str) -> bool:
    return not isinstance(s, str) or not s.strip()

CONTENT_RETRIES = int(os.getenv("EVAL_CONTENT_RETRIES", "2"))
CONTENT_BACKOFF = float(os.getenv("EVAL_CONTENT_BACKOFF", "1.5"))  # seconds
ANSWER_CUE = "\n\nPlaintext:"  # nudges models that return whitespace
FALLBACK_OUTPUT = "[NO_ANSWER]"  # ensures every record has an output


def safe_generate(model, prompt: str) -> dict:
    """
    Call the model with content-level retries.
    Guarantees a non-empty 'output' (falls back to FALLBACK_OUTPUT).
    Returns: {status, output, error_message?, attempts, used_cue}
    """
    attempts = 0
    used_cue = False

    # attempt 0: original prompt
    attempts += 1
    try:
        out = model.generate(prompt)
        if _is_blank(out) or out.startswith("[Rate-limited"):
            raise ValueError("blank_or_rate_limited")
        return {"status": "ok", "output": out.strip(), "attempts": attempts, "used_cue": used_cue}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        last_err = str(e)

    # retries with cue
    retry_prompt = prompt + ANSWER_CUE
    used_cue = True
    for _ in range(CONTENT_RETRIES):
        time.sleep(CONTENT_BACKOFF)
        attempts += 1
        try:
            out = model.generate(retry_prompt)
            if _is_blank(out) or out.startswith("[Rate-limited"):
                raise ValueError("blank_or_rate_limited")
            return {"status": "ok", "output": out.strip(), "attempts": attempts, "used_cue": used_cue}
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = str(e)

    # hard fallback to ensure an answer is present
    return {
        "status": "empty",
        "output": FALLBACK_OUTPUT,
        "error_message": last_err,
        "attempts": attempts,
        "used_cue": used_cue,
    }

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

    # Optional throttle between items (seconds). Good for free-tier models.
    # Example: EVAL_MIN_SECS_BETWEEN_CALLS=0.5  (or leave unset for no extra sleep)
    per_item_sleep = float(os.getenv("EVAL_MIN_SECS_BETWEEN_CALLS", "0"))

    total = 0
    correct = 0

    try:
        with open(out_path, "w", encoding="utf-8") as fout:
            for i, item in enumerate(adapter.iter_items()):
                if args.limit is not None and i >= args.limit:
                    break

                prompt = adapter.build_prompt(item)

                # Call model safely
                result = safe_generate(model, prompt)
                status = result["status"]
                output = result.get("output", "")
                error_message = result.get("error_message", "")
                attempts = result.get("attempts", 1)
                used_cue = result.get("used_cue", False)

                # Only score "ok" outputs; others count as incorrect
                if status == "ok":
                    label = adapter.score(item, output)
                else:
                    label = 0

                correct += label
                total += 1

                rec = {
                    "id": item.get("id", i),
                    "dataset": args.dataset,
                    "model": model.name,
                    "status": status,          # ok | rate_limited | empty | error
                    "prompt": prompt,
                    "output": output,
                    "correct": label,
                    "attempts": attempts,
                    "used_cue": used_cue,
                }
                if error_message:
                    rec["error_message"] = error_message

                # Include extra metadata if present
                for key in ("algorithm", "ciphertext", "prompt_text", "question"):
                    if key in item:
                        rec[key] = item[key]

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Optional progress logging
                if args.progress_every and total % args.progress_every == 0:
                    acc_so_far = correct / total if total else 0.0
                    print(f"[{total}] running acc={acc_so_far:.4f} (last status={status})")

                # Optional gentle pacing (useful for free-tier models)
                if per_item_sleep > 0:
                    time.sleep(per_item_sleep)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Finalizing...")

    acc = correct / total if total else 0.0
    print(f"Evaluated {total} items on {args.dataset} with {model.name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Wrote results to: {out_path}")

if __name__ == "__main__":
    main()
