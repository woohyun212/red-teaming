import json
import logging
from pathlib import Path
from typing import Dict, Set, Any, Optional


def setup_logging() -> None:
    """전역 로거 설정."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def read_jsonl_line(line: str) -> Optional[Dict[str, Any]]:
    """한 줄의 JSONL 문자열을 파싱한다. 실패 시 None을 반환."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        logging.error("JSON decode error: %s", e)
        return None


def count_pii_from_jsonl(jsonl_path: Path) -> Dict[str, int]:
    """
    JSONL(enron_emails.jsonl)을 한 줄씩 읽으면서
    names/emails/phones를 전역 dedup(set)으로 모은 뒤, 개수를 반환한다.
    """
    if not jsonl_path.exists():
        logging.error("JSONL file does not exist: %s", jsonl_path)
        return {
            "unique_names": 0,
            "unique_emails": 0,
            "unique_phones": 0,
            "total_unique_pii": 0,
            "num_lines": 0,
            "num_valid_records": 0,
            "num_failed_lines": 0,
        }

    unique_names: Set[str] = set()
    unique_emails: Set[str] = set()
    unique_phones: Set[str] = set()

    num_lines = 0
    num_valid_records = 0
    num_failed_lines = 0

    logging.info("Start reading JSONL: %s", jsonl_path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            num_lines += 1
            record = read_jsonl_line(line)
            if record is None:
                num_failed_lines += 1
                continue

            # 기본 구조: { "id": ..., "text": ..., "pii": { "names": [...], "emails": [...], "phones": [...] } }
            pii = record.get("pii", {})
            names = pii.get("names", []) or []
            emails = pii.get("emails", []) or []
            phones = pii.get("phones", []) or []

            # 각 리스트의 요소가 문자열이라고 가정하고, 전역 set에 추가
            for name in names:
                if isinstance(name, str):
                    unique_names.add(name)
            for email in emails:
                if isinstance(email, str):
                    unique_emails.add(email)
            for phone in phones:
                if isinstance(phone, str):
                    unique_phones.add(phone)

            num_valid_records += 1

    stats = {
        "unique_names": len(unique_names),
        "unique_emails": len(unique_emails),
        "unique_phones": len(unique_phones),
        "total_unique_pii": len(unique_names) + len(unique_emails) + len(unique_phones),
        "num_lines": num_lines,
        "num_valid_records": num_valid_records,
        "num_failed_lines": num_failed_lines,
    }

    logging.info(
        "Finished counting PII. lines=%d, valid=%d, failed=%d, "
        "unique_names=%d, unique_emails=%d, unique_phones=%d, total_unique_pii=%d",
        stats["num_lines"],
        stats["num_valid_records"],
        stats["num_failed_lines"],
        stats["unique_names"],
        stats["unique_emails"],
        stats["unique_phones"],
        stats["total_unique_pii"],
    )

    return stats


def main() -> None:
    """
    enron_emails.jsonl을 읽어서 전체 PII(dedup) 개수를 카운트하고,
    결과를 로그와 stdout에 출력하는 엔트리 포인트.
    """
    setup_logging()

    # 기본 경로: 현재 작업 디렉토리의 enron_emails.jsonl
    jsonl_path = Path.cwd() / "enron_emails.jsonl"
    logging.info("Target JSONL file: %s", jsonl_path)

    stats = count_pii_from_jsonl(jsonl_path)

    # 사람이 보기 좋게 stdout에도 요약 출력
    print("==== Global PII Statistics (deduplicated across dataset) ====")
    print(f"JSONL path          : {jsonl_path}")
    print(f"Total lines         : {stats['num_lines']}")
    print(f"Valid records       : {stats['num_valid_records']}")
    print(f"Failed lines        : {stats['num_failed_lines']}")
    print(f"Unique names        : {stats['unique_names']}")
    print(f"Unique emails       : {stats['unique_emails']}")
    print(f"Unique phones       : {stats['unique_phones']}")
    print(f"Total unique PII    : {stats['total_unique_pii']}")


if __name__ == "__main__":
    main()
