import json

def dedup_jsonl(input_path, output_path=None):
    seen = set()
    unique = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt = obj["instruction"]
            if prompt not in seen:
                seen.add(prompt)
                unique.append(obj)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            # for obj in unique:
            #     f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"총 개수(중복 제거 후): {len(unique)}")
    return len(unique)

# 사용 예시
dedup_jsonl(
    "sft_instructions_pii.jsonl",
    "sft_instructions_pii.json"
)

