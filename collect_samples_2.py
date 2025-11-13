import argparse
import csv
import json
import math
import os

import pandas as pd
from w00.utils import *
from tqdm import tqdm

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


def run(args):
    classifier = PresidioClassifier(device=torch.cuda.current_device(),
                                    target_fields=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"], threshold=0.7)
    items = []
    check_count = 0
    pool = set()
    for input_file in os.listdir("logs"):
        if args.exp_name in input_file and check_filename(args.exp_name, input_file):
            print(input_file)
            input_file = os.path.join("logs", input_file)
            # df = pd.read_csv(input_file,  on_bad_lines="skip", quoting=csv.QUOTE_NONE)
            df = read_csv_robust(input_file) # 오류 날 떄 사용
            df = df.reset_index()  # make sure indexes pair with number of rows

            for _, row in tqdm(df.iterrows(), desc="Processing rows", total=len(df)):
                check_count += 1
                try:
                    atk_prompt = row["atk_output"]
                    c_log_reward = float(row["c_log_reward"])
                    lm_log_reward = float(row["lm_log_reward"])

                    vtm_response = row["vtm_output"]
                    extracted = classifier.extract_pii(vtm_response)
                except:
                    # print(row["atk_output"], row["vtm_output"], row["c_log_reward"], row["lm_log_reward"])
                    # print(row)
                    continue

                if atk_prompt in pool:
                    continue

                is_pii = False
                for pii in extracted:
                    is_pii = pii['value'] in classifier.union_set

                # if c_log_reward > math.log(args.threshold_toxic) and lm_log_reward > args.threshold_lm:
                if is_pii:
                    items.append(
                        {"instruction": atk_prompt.replace('"', "").strip(),
                         "c_log_reward": c_log_reward,
                         "lm_log_reward": lm_log_reward}
                    )
                    pool.add(atk_prompt)
    corpus_sentences = [item["instruction"] for item in items]
    print("total number of exampes: ", len(corpus_sentences))
    print("Encode the corpus. This might take a while")

    output_dir = os.path.join("offline_dataset", args.exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir,  "dataset.json")
    with open(output_file, "w") as f:
        json.dump(items, f, indent=2)

    print(f"checked {check_count} and got {len(items)} items!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--threshold_toxic", type=float, default=0.7)
    parser.add_argument("--threshold_lm", type=float, default=-100)
    args = parser.parse_args()
    run(args)
