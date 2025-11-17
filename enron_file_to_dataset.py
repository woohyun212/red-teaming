# enron_file_to_dataset.py
# Enron 메일 데이터셋에서 PII(이름, 이메일, 전화번호)를 추출하여
# 각 메일당 한 줄씩 가지는 JSONL(enron_emails.jsonl) 파일로 저장하는 스크립트.
# - 루트 디렉토리: ../maildir/
# - Presidio + 정규표현식으로 PII를 감지
# - 메일 하나를 처리할 때마다 곧바로 JSONL에 한 줄을 append하는 스트리밍 방식

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

# Microsoft Presidio 기반 PII 탐지를 위해 AnalyzerEngine과 spaCy NLP 엔진 provider를 사용
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


# 이메일을 보조적으로 검출하기 위한 정규표현식 (Presidio 탐지 결과를 보완)
EMAIL_REGEX = r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9.\-]+"

# 미국식 전화번호 포맷을 느슨하게 잡는 정규표현식
# - 국가코드(+1 등), 괄호, 공백, 하이픈, 점(.) 등을 허용
PHONE_REGEX = r"""
    (?:
        (?:\+?\d{1,2}[\s\-\.]?)?      # Optional country code
        (?:\(?\d{3}\)?[\s\-\.]?)      # Area code
        \d{3}[\s\-\.]?\d{4}           # Local number
    )
"""


def setup_logging() -> None:
    # 전역 로거 설정: INFO 레벨 이상의 로그를 시간/레벨/메시지 형식으로 출력
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def setup_presidio_analyzer() -> AnalyzerEngine:
    # spaCy 기반 NLP 엔진 설정
    # en_core_web_lg 모델을 Presidio가 내부적으로 사용하도록 구성
    """
    Initialize Presidio AnalyzerEngine with a spaCy NLP engine.

    Note:
        Requires an English spaCy model installed, e.g.:
        python -m spacy download en_core_web_lg
    """
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    return analyzer


def normalize_email(email: str) -> str:
    # 이메일 문자열 양 끝의 공백과 구두점을 제거하고, 소문자로 통일
    email = email.strip().strip(".,;:<>\"'()[]")
    email = email.lower()
    return email


def normalize_phone(phone: str) -> str:
    # 전화번호 문자열에서 숫자만 남기고 나머지 문자(공백, -, (), . 등)를 제거
    digits = re.sub(r"[^\d]", "", phone)
    # 너무 짧은 숫자 시퀀스(예: 3~4자리 내선 번호 등)는 전화번호로 보지 않고 무시
    if len(digits) < 7:
        return ""
    return digits


def normalize_name(name: str) -> str:
    # 이름 문자열에서 중복 공백을 제거하고 strip
    name = " ".join(name.strip().split())
    if not name:
        return ""
    # 이메일 주소나 URL처럼 보이는 문자열은 이름으로 취급하지 않고 제외
    if ":" in name or "@" in name or "." in name and " " not in name:
        return ""
    return name.lower()


def extract_pii_with_presidio(
    text: str, analyzer: AnalyzerEngine
) -> Dict[str, Set[str]]:
    # Presidio AnalyzerEngine을 사용하여 PERSON, EMAIL_ADDRESS, PHONE_NUMBER 엔티티를 감지
    # 감지된 텍스트 span을 각 카테고리별 set에 넣고, 후에 정규화를 거쳐 반환
    names: Set[str] = set()
    emails: Set[str] = set()
    phones: Set[str] = set()

    # Presidio로 한 번에 세 종류의 엔티티를 분석
    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"],
        )
    except Exception as e:
        logging.error("Presidio analysis failed: %s", e)
        return {"names": names, "emails": emails, "phones": phones}

    for res in results:
        # Presidio 결과 객체에서 원본 텍스트의 start/end 인덱스로 실제 문자열을 추출
        start, end = res.start, res.end
        chunk = text[start:end]
        if not chunk:
            continue

        if res.entity_type == "PERSON":
            norm = normalize_name(chunk)
            if norm:
                names.add(norm)
        elif res.entity_type == "EMAIL_ADDRESS":
            norm = normalize_email(chunk)
            if norm:
                emails.add(norm)
        elif res.entity_type == "PHONE_NUMBER":
            norm = normalize_phone(chunk)
            if norm:
                phones.add(norm)

    return {"names": names, "emails": emails, "phones": phones}


def extract_pii_with_regex(text: str) -> Dict[str, Set[str]]:
    # 정규표현식을 사용해 이메일/전화번호를 추가로 검출
    # Presidio가 놓친 패턴을 보완하고, 간단한 포맷에도 대응하기 위함
    emails: Set[str] = set()
    phones: Set[str] = set()

    # 이메일 패턴에 매칭되는 모든 문자열을 찾아 정규화 후 set에 추가
    for match in re.findall(EMAIL_REGEX, text, flags=re.IGNORECASE):
        norm = normalize_email(match)
        if norm:
            emails.add(norm)

    # 전화번호 패턴에 매칭되는 문자열을 찾아 숫자만 남기는 방식으로 정규화
    for match in re.finditer(PHONE_REGEX, text, flags=re.VERBOSE):
        raw = match.group(0)
        norm = normalize_phone(raw)
        if norm:
            phones.add(norm)

    return {"emails": emails, "phones": phones}


def merge_pii(
    presidio_pii: Dict[str, Set[str]], regex_pii: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    # Presidio와 Regex 결과를 merge하여 최종적으로 사용할 PII 집합을 만든다.
    # - 이름은 Presidio 결과만 사용 (정규표현식 기반 이름 검출은 구현하지 않음)
    # - 이메일/전화번호는 Presidio + Regex union
    names = presidio_pii.get("names", set())
    emails = presidio_pii.get("emails", set()).union(regex_pii.get("emails", set()))
    phones = presidio_pii.get("phones", set()).union(regex_pii.get("phones", set()))

    # set을 정렬된 리스트로 변환하여 JSON 직렬화가 가능하도록 함
    return {
        "names": sorted(names),
        "emails": sorted(emails),
        "phones": sorted(phones),
    }


# 메일 디렉토리(root_dir) 아래의 모든 파일을 재귀적으로 순회하는 제너레이터
def iter_email_files(root_dir: Path):
    """
    Yield all file paths under root_dir recursively.
    """
    # os.walk를 사용해 디렉토리 트리를 순회하면서 파일 경로들을 yield
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            yield file_path


def read_file_text(path: Path) -> Optional[str]:
    # 주어진 경로의 파일 내용을 UTF-8로 읽되, 오류가 있으면 로그만 남기고 None 반환
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logging.error("Failed to read file %s: %s", path, e)
        return None


def process_single_file(
    file_path: Path,
    root_dir: Path,
    analyzer: AnalyzerEngine,
    out_f,
) -> Optional[bool]:
    # 개별 메일 파일을 처리하는 핵심 함수
    # 1) 파일 내용 읽기
    # 2) Presidio + Regex로 PII 추출
    # 3) 상대 경로 id 구성
    # 4) JSONL 한 줄을 출력 파일에 기록
    logging.info("Processing file: %s", file_path)

    # Presidio와 Regex 두 가지 방법으로 PII 감지
    text = read_file_text(file_path)
    if text is None:
        return None

    presidio_pii = extract_pii_with_presidio(text, analyzer)
    regex_pii = extract_pii_with_regex(text)
    pii = merge_pii(presidio_pii, regex_pii)

    # ../maildir/ 기준 상대 경로를 id로 사용 (예: "allen-p/all_documents/1.")
    rel_id = file_path.relative_to(root_dir).as_posix()

    # JSONL 한 줄로 직렬화할 레코드(dict) 구성
    record = {
        "id": rel_id,
        "text": text,
        "pii": pii,
    }

    try:
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error("Failed to write JSONL line for %s: %s", file_path, e)
        return None

    # 이 메일에 PII가 하나라도 있는지 여부를 bool로 계산 (통계용)
    has_pii = any(pii[key] for key in ("names", "emails", "phones"))
    return has_pii


def main() -> None:
    # 스크립트 진입점: 디렉토리 경로 설정, Presidio 초기화, 파일 순회 및 JSONL 작성
    setup_logging()
    logging.info("Starting Enron maildir -> JSONL PII extraction")

    # 현재 스크립트 파일 경로 기준으로 ../maildir/ 위치를 계산
    script_dir = Path(__file__).resolve().parent
    root_dir = (script_dir / ".." / "maildir").resolve()

    if not root_dir.exists() or not root_dir.is_dir():
        logging.error("Maildir root directory does not exist: %s", root_dir)
        return

    try:
        analyzer = setup_presidio_analyzer()
    except Exception as e:
        logging.error("Failed to initialize Presidio AnalyzerEngine: %s", e)
        return

    output_path = Path.cwd() / "enron_emails.jsonl"
    logging.info("Output JSONL file will be written to: %s", output_path)

    total_files = 0
    processed_files = 0
    pii_files = 0
    skipped_files = 0

    # 출력 파일을 한 번 열고, 모든 메일을 순회하며 한 메일당 한 줄씩 JSONL을 append
    with output_path.open("w", encoding="utf-8") as out_f:
        for file_path in iter_email_files(root_dir):
            # 모든 메일 파일에 대해 개별 처리; 에러가 발생해도 다른 파일 처리는 계속 진행
            total_files += 1
            result = process_single_file(file_path, root_dir, analyzer, out_f)
            if result is None:
                skipped_files += 1
                continue
            processed_files += 1
            if result:
                pii_files += 1

    # 최종 통계 로그 출력
    logging.info("Processing finished.")
    logging.info("Total files discovered: %d", total_files)
    logging.info("Successfully processed files: %d", processed_files)
    logging.info("Files with at least one PII entity: %d", pii_files)
    logging.info("Skipped files (errors): %d", skipped_files)


if __name__ == "__main__":
    main()