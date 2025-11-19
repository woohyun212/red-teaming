import json
import argparse
from typing import List, Dict, Any


checked_pii = set({})


def find_name_context(text: str, name: str, window: int = 40) -> str:
    """
    text 안에서 name을 찾아 앞뒤 window 글자씩 context를 반환.
    못 찾으면 앞부분 일부만 보여줌.
    """
    lower_text = text.lower()
    lower_name = name.lower().strip()

    idx = lower_text.find(lower_name)
    if idx == -1:
        # 못 찾으면 텍스트 앞부분 일부만 (디버그용)
        snippet = text[: window * 2].replace("\n", "\\n")
        return f"[name not found in text] {snippet}"

    start = max(0, idx - window)
    end = min(len(text), idx + len(name) + window)
    snippet = text[start:end].replace("\n", "\\n")
    return snippet


def interactive_filter_names(
    record: Dict[str, Any],
    fp_writer,
    window: int = 40,
) -> Dict[str, Any]:
    """
    하나의 json 레코드에 대해 names를 인터랙티브하게 필터링.
    fp_writer: false_positives.jsonl 파일 핸들
    """
    text = record.get("text", "")
    pii = record.get("pii") or {}
    names: List[str] = pii.get("names") or []

    if not names:
        return record

    print("=" * 80)
    print(f"ID: {record.get('id')}  (총 {len(names)}개 name 후보)")

    new_names: List[str] = []

    for name in names:
        context = find_name_context(text, name, window=window)
        if name in checked_pii:
            print(f'\n{name} 은(는) 이미 처리된 PII이므로 자동으로 유지합니다.')
            new_names.append(name)
            continue

        print("\n----------------------------------------")
        print(f'PII 후보 (name): "{name}"')
        print(f'text context : "...{context}..."')
        print("입력: 1=정상 PII(유지), 0 또는 기타=오탐(제거), s=이 레코드 나머지 스킵, q=전체 종료")
        user_in = input("> ").strip()

        if user_in == "1":
            checked_pii.add(name)
            new_names.append(name)
        elif user_in.lower() == "s":
            # 나머지는 그냥 그대로 둔다고 할지, 전부 유지/제거할지 정책 선택 가능
            # 여기서는 '나머지 이름들은 일단 유지' 로 할게.
            remaining_names = names[names.index(name):]
            new_names.extend(remaining_names)
            print(f"이 레코드 나머지 {len(remaining_names)}개 이름은 그대로 유지하고 다음 레코드로 넘어갑니다.")
            break
        elif user_in.lower() == "q":
            # q면 현재까지 처리된 내용만 반영하고 상위에서 종료 신호
            record["pii"]["names"] = new_names
            raise KeyboardInterrupt  # main 쪽에서 잡아서 종료
        else:
            # 오탐 처리 → false_positives.jsonl에 기록
            fp_entry = {
                "id": record.get("id"),
                "name": name,
                "context": context,
            }
            fp_writer.write(json.dumps(fp_entry, ensure_ascii=False) + "\n")
            print("→ 오탐으로 처리하고 false_positives.jsonl에 기록했습니다.")

    record["pii"]["names"] = new_names
    return record


def main(input_path: str, output_path: str, fp_output_path: str, window: int = 40):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         open(fp_output_path, "w", encoding="utf-8") as ffp:

        try:
            for line_no, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] {line_no}번째 줄 JSON 파싱 실패, 건너뜀.")
                    continue

                try:
                    updated_record = interactive_filter_names(
                        record,
                        fp_writer=ffp,
                        window=window,
                    )
                except KeyboardInterrupt:
                    print("\n사용자에 의해 중단되었습니다. 지금까지의 결과만 저장합니다.")
                    # 현재까지 처리한 record는 반영하고 종료
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    break

                # 필터링된 레코드를 output.jsonl에 기록
                fout.write(json.dumps(updated_record, ensure_ascii=False) + "\n")

        except KeyboardInterrupt:
            print("\n사용자에 의해 중단되었습니다. (전체 종료)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="입력 jsonl 경로")
    parser.add_argument("--output", "-o", default="output.jsonl", help="정제된 jsonl 출력 경로")
    parser.add_argument("--fp-output", "-f", default="false_positives.jsonl",
                        help="오탐 name들을 저장할 jsonl 경로")
    parser.add_argument("--window", "-w", type=int, default=40,
                        help="context로 보여줄 앞뒤 글자수")
    args = parser.parse_args()

    main(args.input, args.output, args.fp_output, window=args.window)