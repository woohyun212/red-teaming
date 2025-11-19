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
import string
from pathlib import Path
from typing import Dict, List, Optional, Set
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
# Microsoft Presidio 기반 PII 탐지를 위해 AnalyzerEngine과 spaCy NLP 엔진 provider를 사용
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
import mailparser


# 멀티프로세싱 환경에서 각 워커 프로세스마다 Presidio AnalyzerEngine을 초기화하기 위한 전역 변수
ANALYZER: Optional[AnalyzerEngine] = None


def init_worker() -> None:
    """
    멀티프로세싱 풀의 각 워커 프로세스에서 호출되는 초기화 함수.
    각 프로세스마다 독립적인 Presidio AnalyzerEngine 인스턴스를 생성한다.
    """
    global ANALYZER
    # 이미 초기화된 경우 다시 만들 필요 없음
    if ANALYZER is None:
        ANALYZER = setup_presidio_analyzer()

def get_mail_body(raw_mail: str) -> str:
    """
    메일 원본 문자열에서 바디(text/plain 기준)를 추출한다.
    - mailparser가 설치되어 있으면 mailparser.parse_from_string(raw_mail).body 사용
    - mailparser가 없거나 실패하면, 첫 번째 빈 줄(헤더/바디 구분) 이후를 바디로 간주
    """
    # mailparser가 사용 가능한 경우 우선 시도
    # if mailparser is not None:
    try:
        mail = mailparser.parse_from_string(raw_mail)
        if mail and mail.body:
            return mail.body
    except Exception as e:
        logging.error("mailparser failed to parse mail body: %s", e)

    # mailparser가 없거나 실패한 경우: 단순 헤더/바디 분리 (첫 번째 빈 줄 기준)
    parts = raw_mail.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1]
    return raw_mail

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
    # Presidio 관련 로거는 에러만 보이게 줄이기
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    logging.getLogger("presidio-anonymizer").setLevel(logging.ERROR)

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
        "model_to_presidio_entity_mapping": {
            "en": {  # spaCy의 레이블 -> Presidio 엔티티명
                "PERSON": "PERSON",
                "EMAIL": "EMAIL_ADDRESS",
                "PHONE": "PHONE_NUMBER",
            }
        },
        "low_score_entity_names": [],  # 필요 없으면 빈 리스트
        "labels_to_ignore": [],  # 무시할 레이블 없으면 빈 리스트
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # spaCy nlp 객체를 직접 꺼내서 max_length 조정 (주의!)
    try:
        spacy_nlp = nlp_engine.nlp["en"]
        spacy_nlp.max_length = 2_000_000  # or whatever
    except Exception as e:
        logging.warning("Failed to set spaCy max_length: %s", e)

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
    name = " ".join(name.lower().strip().split())
    if not name:
        return ""
    # 이메일 주소나 URL처럼 보이는 문자열은 이름으로 취급하지 않고 제외
    if ":" in name or "@" in name or "." in name and " " not in name:
        return ""
    allowed_characters = string.ascii_lowercase+' `' # 소문자, 공백, 백틱 허용
    for char in name.lower():
        if char not in allowed_characters:
            return ""

    return name

def extract_names_from_headers(full_text: str) -> Set[str]:
    """
    메일 원문에서 X-From, X-To 헤더를 파싱해서 이름 후보를 추출한다.
    - 이메일 주소(<...>)는 제거
    - @ 도메인 붙은 토큰은 제거
    - ',' / ';' 로 1차 split, 공백으로 2차 split
    - normalize_name을 거친 full name + 토큰(3글자 이상)을 names로 추가
    """
    header_lines: List[str] = []

    # X-From / X-To 헤더 라인만 뽑기 (대소문자 무시)
    for header in ("X-From", "X-To"):
        pattern = rf"^{header}:\s*(.+)$"
        matches = re.findall(pattern, full_text, flags=re.MULTILINE | re.IGNORECASE)
        header_lines.extend(matches)

    candidates: Set[str] = set()

    for raw in header_lines:
        # 1) 이메일 주소(<...>) 제거
        tmp = re.sub(r"<[^>]+>", " ", raw)
        # 2) @가 포함된 토큰 제거 (도메인/아이디 등)
        tokens = tmp.split()
        tokens = [t for t in tokens if "@" not in t]
        tmp = " ".join(tokens)
        # 3) 따옴표 제거
        tmp = tmp.replace('"', " ").replace("'", " ")

        # 4) ',' / ';' 기준으로 1차 분리 (여러 명이 있을 수 있으니까)
        parts = re.split(r"[;,]", tmp)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 먼저 전체를 하나의 이름으로 normalization 시도
            full_norm = normalize_name(part)
            if full_norm:
                candidates.add(full_norm)

                # full name을 공백 기준으로 쪼개서 3글자 이상 토큰도 추가
                for tok in full_norm.split():
                    if len(tok) >= 3:
                        candidates.add(tok)

    return candidates

def extract_pii_with_presidio(
    full_text: str, body_text: str, analyzer: AnalyzerEngine
) -> Dict[str, Set[str]]:
    """
    Presidio AnalyzerEngine을 사용하여 PII를 감지한다.
    - PERSON, EMAIL_ADDRESS는 전체 메일 텍스트(full_text, 헤더+바디)에서 검출
    - PHONE_NUMBER는 메일 바디(body_text)에서만 검출하여 Message-ID 등의 노이즈를 줄인다.
    """
    names: Set[str] = set()
    emails: Set[str] = set()
    phones: Set[str] = set()

    try:
        # 전체 텍스트에서 PERSON / EMAIL_ADDRESS 탐지
        full_results = analyzer.analyze(
            text=full_text,
            language="en",
            entities=["EMAIL_ADDRESS", "PERSON"],
        )

        # 바디 텍스트에서 PHONE_NUMBER만 별도로 탐지
        body_results = analyzer.analyze(
            text=body_text,
            language="en",
            entities=["PHONE_NUMBER"],
        )
    except Exception as e:
        logging.error("Presidio analysis failed: %s", e)
        return {"names": names, "emails": emails, "phones": phones}

    # 전체 텍스트 기반 결과 처리 (이름/이메일)
    for res in full_results:
        start, end = res.start, res.end
        chunk = full_text[start:end]
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

    # 바디 텍스트 기반 결과 처리 (전화번호)
    for res in body_results:
        start, end = res.start, res.end
        chunk = body_text[start:end]
        if not chunk:
            continue

        if res.entity_type == "PHONE_NUMBER":
            norm = normalize_phone(chunk)
            if norm:
                phones.add(norm)

    # --- 추가: X-From / X-To 헤더에서 이름 후보 추출 ---
    header_names = extract_names_from_headers(full_text)
    names.update(header_names)

    return {"names": names, "emails": emails, "phones": phones}


def extract_pii_with_regex(full_text: str, body_text: str) -> Dict[str, Set[str]]:
    """
    정규표현식을 사용해 이메일/전화번호를 추가로 검출한다.
    - 이메일: 헤더/바디 전체(full_text)에서 검출
    - 전화번호: 메일 바디(body_text)에서만 검출
    """
    emails: Set[str] = set()
    phones: Set[str] = set()

    # 이메일 패턴에 매칭되는 모든 문자열을 찾아 정규화 후 set에 추가
    for match in re.findall(EMAIL_REGEX, full_text, flags=re.IGNORECASE):
        norm = normalize_email(match)
        if norm:
            emails.add(norm)

    # 전화번호 패턴에 매칭되는 문자열을 찾아 숫자만 남기는 방식으로 정규화
    for match in re.finditer(PHONE_REGEX, body_text, flags=re.VERBOSE):
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

def worker_process_file(args):
    """
    멀티프로세싱 풀에서 사용할 워커 함수.
    입력:
        args: (file_path_str, root_dir_str) 튜플
    출력:
        {
            "status": "ok" 또는 "skip",
            "json": 직렬화된 JSON 문자열 또는 None,
            "has_pii": bool (status == "ok"일 때만 의미 있음)
        }
    """
    global ANALYZER
    file_path_str, root_dir_str = args
    file_path = Path(file_path_str)
    root_dir = Path(root_dir_str)

    # 워커 프로세스에서 ANALYZER가 초기화되지 않았다면 방어적으로 초기화
    if ANALYZER is None:
        ANALYZER = setup_presidio_analyzer()

    text = read_file_text(file_path)
    if text is None:
        # 읽기 실패 등으로 스킵
        return {"status": "skip", "json": None, "has_pii": False}

    body_text = get_mail_body(text)

    presidio_pii = extract_pii_with_presidio(text, body_text, ANALYZER)
    regex_pii = extract_pii_with_regex(text, body_text)
    pii = merge_pii(presidio_pii, regex_pii)

    rel_id = file_path.relative_to(root_dir).as_posix()
    record = {
        "id": rel_id,
        "text": text,
        "pii": pii,
    }
    json_line = json.dumps(record, ensure_ascii=False)
    has_pii = any(pii[key] for key in ("names", "emails", "phones"))

    return {"status": "ok", "json": json_line, "has_pii": has_pii}

def process_single_file(
    file_path: Path,
    root_dir: Path,
    analyzer: AnalyzerEngine,
    out_f,
) -> Optional[bool]:
    # 개별 메일 파일을 처리하는 단일 프로세스 버전 함수
    # (현재 메인 로직은 멀티프로세싱 worker_process_file을 사용하며,
    #  이 함수는 디버깅/단일 프로세스 실행 시에만 사용할 수 있다.)
    # 1) 파일 내용 읽기
    # 2) Presidio + Regex로 PII 추출
    # 3) 상대 경로 id 구성
    # 4) JSONL 한 줄을 출력 파일에 기록
    # logging.info("Processing file: %s", file_path)

    # Presidio와 Regex 두 가지 방법으로 PII 감지
    text = read_file_text(file_path)
    if text is None:
        return None

    body_text = get_mail_body(text)

    presidio_pii = extract_pii_with_presidio(text, body_text, analyzer)
    regex_pii = extract_pii_with_regex(text, body_text)
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

def count_email_files(root_dir: Path) -> int:
    count = 0
    for _ in iter_email_files(root_dir):
        count += 1
    return count

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
        # 한 번 테스트로 AnalyzerEngine을 초기화해 보고, 환경(모델 설치 등)에 문제가 없는지 확인
        _ = setup_presidio_analyzer()
    except Exception as e:
        logging.error("Failed to initialize Presidio AnalyzerEngine: %s", e)
        return

    output_path = Path.cwd() / "enron_emails_mp.jsonl"
    logging.info("Output JSONL file will be written to: %s", output_path)

    processed_files = 0
    pii_files = 0
    skipped_files = 0

    # 먼저 전체 파일 목록을 리스트로 만들어 개수를 파악하고, tqdm에 total로 넘긴다.
    file_paths = list(iter_email_files(root_dir))
    total_files = len(file_paths)
    logging.info("Total email files to process: %d", total_files)

    # 출력 파일을 한 번 열고, 멀티프로세싱 풀을 사용해 병렬로 메일을 처리
    with output_path.open("w", encoding="utf-8") as out_f:
        num_workers = max(1, cpu_count() - 1)
        logging.info("Using %d worker processes", num_workers)

        # (file_path_str, root_dir_str) 튜플 시퀀스를 만들어 풀에 전달
        args_iter = ((str(p), str(root_dir)) for p in file_paths)

        with Pool(processes=num_workers, initializer=init_worker) as pool:
            for result in tqdm(
                pool.imap_unordered(worker_process_file, args_iter),
                total=total_files,
                desc="Processing emails",
            ):
                # 에러/스킵된 파일은 카운트만 증가시키고 넘어감
                if result is None or result.get("status") != "ok":
                    skipped_files += 1
                    continue

                processed_files += 1
                if result.get("has_pii"):
                    pii_files += 1

                # 메인 프로세스에서만 JSONL 파일에 쓰기
                out_f.write(result["json"] + "\n")

    # 최종 통계 로그 출력
    logging.info("Processing finished.")
    logging.info("Total files discovered: %d", total_files)
    logging.info("Successfully processed files: %d", processed_files)
    logging.info("Files with at least one PII entity: %d", pii_files)
    logging.info("Skipped files (errors): %d", skipped_files)


if __name__ == "__main__":
    # main()
    print(extract_names_from_headers("""Message-ID: <23754218.1075856162542.JavaMail.evans@thyme>\nDate: Thu, 7 Dec 2000 02:37:00 -0800 (PST)\nFrom: tori.kuykendall@enron.com\nTo: ppope01@coair.com\nSubject: Re: Coair Reply\nMime-Version: 1.0\nContent-Type: text/plain; charset=us-ascii\nContent-Transfer-Encoding: 7bit\nX-From: Tori Kuykendall\nX-To: \"Pope, Pamela\" <PPope01@coair.com> @ ENRON\nX-cc: \nX-bcc: \nX-Folder: \\Tori_Kuykendall_Dec2000\\Notes Folders\\'sent mail\nX-Origin: Kuykendall-T\nX-FileName: tkuyken.nsf\n\nHere is the information that you needed:  My tickets were electronic and the \nconfirmation number was NGP6MP.  Thank You."""))