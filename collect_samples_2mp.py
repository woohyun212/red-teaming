import argparse
import csv
import json
import math
import os
from multiprocessing import Pool, cpu_count

import pandas as pd
from w00.utils import *
from tqdm import tqdm

_classifier = None
_threshold_lm = None

def read_csv_robust(path: str) -> pd.DataFrame:
    """Read a possibly malformed CSV robustly.
    1) Prefer the Python engine (required for on_bad_lines) with QUOTE_NONE.
    2) Fallback: standard quoting with escapechar and ignoring encoding errors.
    """
    # Try: python engine + QUOTE_NONE (treat quotes as normal chars)
    try:
        return pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",
            quoting=csv.QUOTE_NONE,
        )
    except Exception:
        # Fallback: allow quoted fields; ignore bad unicode & escape sequences
        try:
            return pd.read_csv(
                path,
                engine="python",
                on_bad_lines="skip",
                sep=",",
                quotechar='"',
                escapechar='\\',
                encoding="utf-8",
                encoding_errors="ignore",
            )
        except TypeError:
            # Older pandas without encoding_errors parameter
            return pd.read_csv(
                path,
                engine="python",
                on_bad_lines="skip",
                sep=",",
                quotechar='"',
                escapechar='\\',
                encoding="utf-8",
            )

def check_filename(exp_name, file_name):
    candidates = [f"{exp_name}_0{i}.csv" for i in range(10)]
    if file_name == f"{exp_name}.csv" or file_name in candidates:
        return True
    else:
        return False


def _init_worker(device, target_fields, threshold, threshold_lm):
    """Initializer for each worker process: create a PresidioClassifier once per process."""
    global _classifier, _threshold_lm
    _classifier = PresidioClassifier(
        device=device,
        target_fields=target_fields,
        threshold=threshold,
    )
    _threshold_lm = threshold_lm


def _process_row(row):
    """Worker function to process a single row and return an item dict or None."""
    global _classifier, _threshold_lm
    try:
        atk_prompt = row["atk_output"]
        c_log_reward = float(row["c_log_reward"])
        lm_log_reward = float(row["lm_log_reward"])
        vtm_response = row["vtm_output"]
    except Exception:
        return None

    try:
        extracted = _classifier.extract_pii(vtm_response)
    except Exception:
        return None

    # Check if any extracted PII is in the classifier's union set
    is_pii = any(pii["value"] in _classifier.union_set for pii in extracted)

    # lm_log_reward >= args.threshold_lm 조건으로 변경
    if is_pii and lm_log_reward >= _threshold_lm:
        return None

    return {
        "instruction": atk_prompt.replace('"', "").strip(),
        "victim_response": vtm_response.replace('"', "\"").strip(),
        "c_log_reward": c_log_reward,
        "lm_log_reward": lm_log_reward,
    }

def run(args):
    """Collect samples from log CSVs using multiprocessing for per-row processing."""
    items_by_instruction = {}
    check_count = 0
    seen_prompts = set()

    # Pre-compute list of relevant log files
    log_dir = "logs"
    input_files = [
        f for f in os.listdir(log_dir)
        if args.exp_name in f and check_filename(args.exp_name, f)
    ]

    if not input_files:
        print("No matching log files found.")

    # Collect all rows from all matching files
    all_rows = []
    for input_file in input_files:
        print(input_file)
        full_path = os.path.join(log_dir, input_file)
        df = read_csv_robust(full_path)  # robust CSV reader
        df = df.reset_index()  # make sure indexes pair with number of rows

        rows = df.to_dict(orient="records")
        all_rows.extend(rows)

    # Create a single worker pool for all rows so that PresidioClassifier
    # instances are created only once per worker process.
    with Pool(
        processes=args.num_workers or cpu_count(),
        initializer=_init_worker,
        initargs=(
            torch.cuda.current_device(),
            ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"],
            0.7,
            args.threshold_lm,
        ),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_row, all_rows),
            desc="Processing rows",
            total=len(all_rows),
        ):
            check_count += 1
            if result is None:
                continue

            instruction = result["instruction"]
            # Deduplicate by instruction text across all files
            if instruction in seen_prompts:
                continue
            seen_prompts.add(instruction)
            items_by_instruction[instruction] = result

    items = list(items_by_instruction.values())
    corpus_sentences = [item["instruction"] for item in items]
    print("total number of exampes: ", len(corpus_sentences))
    print("Encode the corpus. This might take a while")

    output_dir = os.path.join("offline_dataset", args.exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "dataset.json")
    with open(output_file, "w") as f:
        json.dump(items, f, indent=2)

    print(f"checked {check_count} and got {len(items)} items!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--threshold_toxic", type=float, default=0.7)
    parser.add_argument("--threshold_lm", type=float, default=-100)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Number of worker processes for multiprocessing")
    args = parser.parse_args()
    run(args)
