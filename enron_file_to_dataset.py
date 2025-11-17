import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


EMAIL_REGEX = r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9.\-]+"
PHONE_REGEX = r"""
    (?:
        (?:\+?\d{1,2}[\s\-\.]?)?      # Optional country code
        (?:\(?\d{3}\)?[\s\-\.]?)      # Area code
        \d{3}[\s\-\.]?\d{4}           # Local number
    )
"""


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def setup_presidio_analyzer() -> AnalyzerEngine:
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
    email = email.strip().strip(".,;:<>\"'()[]")
    email = email.lower()
    return email


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"[^\d]", "", phone)
    # Heuristic: skip too-short sequences
    if len(digits) < 7:
        return ""
    return digits


def normalize_name(name: str) -> str:
    name = " ".join(name.strip().split())
    if not name:
        return ""
    # Skip if looks like email or URL
    if "@" in name or "." in name and " " not in name:
        return ""
    return name.lower()


def extract_pii_with_presidio(
    text: str, analyzer: AnalyzerEngine
) -> Dict[str, Set[str]]:
    names: Set[str] = set()
    emails: Set[str] = set()
    phones: Set[str] = set()

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
    emails: Set[str] = set()
    phones: Set[str] = set()

    for match in re.findall(EMAIL_REGEX, text, flags=re.IGNORECASE):
        norm = normalize_email(match)
        if norm:
            emails.add(norm)

    for match in re.finditer(PHONE_REGEX, text, flags=re.VERBOSE):
        raw = match.group(0)
        norm = normalize_phone(raw)
        if norm:
            phones.add(norm)

    return {"emails": emails, "phones": phones}


def merge_pii(
    presidio_pii: Dict[str, Set[str]], regex_pii: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    names = presidio_pii.get("names", set())
    emails = presidio_pii.get("emails", set()).union(regex_pii.get("emails", set()))
    phones = presidio_pii.get("phones", set()).union(regex_pii.get("phones", set()))

    return {
        "names": sorted(names),
        "emails": sorted(emails),
        "phones": sorted(phones),
    }


def iter_email_files(root_dir: Path):
    """
    Yield all file paths under root_dir recursively.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            yield file_path


def read_file_text(path: Path) -> Optional[str]:
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
    """
    Process one email file:
      - read text
      - extract PII
      - write JSONL line

    Returns:
        True if processed and has PII,
        False if processed and no PII,
        None if skipped (read error, etc.).
    """
    logging.info("Processing file: %s", file_path)

    text = read_file_text(file_path)
    if text is None:
        return None

    presidio_pii = extract_pii_with_presidio(text, analyzer)
    regex_pii = extract_pii_with_regex(text)
    pii = merge_pii(presidio_pii, regex_pii)

    rel_id = file_path.relative_to(root_dir).as_posix()

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

    has_pii = any(pii[key] for key in ("names", "emails", "phones"))
    return has_pii


def main() -> None:
    setup_logging()
    logging.info("Starting Enron maildir -> JSONL PII extraction")

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

    with output_path.open("w", encoding="utf-8") as out_f:
        for file_path in iter_email_files(root_dir):
            total_files += 1
            result = process_single_file(file_path, root_dir, analyzer, out_f)
            if result is None:
                skipped_files += 1
                continue
            processed_files += 1
            if result:
                pii_files += 1

    logging.info("Processing finished.")
    logging.info("Total files discovered: %d", total_files)
    logging.info("Successfully processed files: %d", processed_files)
    logging.info("Files with at least one PII entity: %d", pii_files)
    logging.info("Skipped files (errors): %d", skipped_files)


if __name__ == "__main__":
    main()