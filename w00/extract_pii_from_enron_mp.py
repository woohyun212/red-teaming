#!/usr/bin/env python
"""
extract_pii_from_enron_mp.py
----------------------------
병렬 spaCy NER(`nlp.pipe` + n_process) 버전.

사용 예시
---------
python w00/extract_pii_from_enron_mp.py \
    --root ./maildir \
    --out  data/v6_enron_pii.jsonl \
    --batch 64 \
    --workers 16
"""
from __future__ import annotations

import os, re, json, argparse, random
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
import mailparser
import spacy           # python -m spacy download en_core_web_sm

# ---------- 정규식 ----------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"\b(?:\+?1[-\s.]*)?\(?\d{3}\)?[-\s.]?\d{3,4}[-\s.]?\d{4}\b")

# ---------- spaCy NER 모델 ----------
try:
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
except OSError:
    raise SystemExit(
        "❗ spaCy 모델(en_core_web_sm)이 설치돼 있지 않습니다.\n"
        "→  python -m spacy download en_core_web_sm"
    )

w
# ---------- 본문 전처리 ----------
def clean_body(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- 헤더 추출 ----------
def extract_headers(mail) -> Dict:
    """mailparser 객체에서 필요한 헤더만 구조화"""
    return {
        "from_email": mail.from_[0][1] if mail.from_ else "",
        "to_email": [t[1] for t in mail.to] if mail.to else [],
        "cc": [c[1] for c in mail.cc] if mail.cc else [],
        "bcc": [b[1] for b in mail.bcc] if mail.bcc else [],
        "x_from": mail.headers.get("X-From", ""),
        "x_to": mail.headers.get("X-To", ""),
        "x_cc": mail.headers.get("X-cc", ""),
        "x_bcc": mail.headers.get("X-bcc", ""),
        "x_origin": mail.headers.get("X-Origin", ""),
        "x_folder": mail.headers.get("X-Folder", ""),
        "x_filename": mail.headers.get("X-FileName", ""),
        "date": mail.date.isoformat() if mail.date else "",
    }


# ---------- PII 추출 (Doc 객체 사용) ----------
def extract_pii_from_doc(doc) -> Dict[str, List[str]]:
    """spaCy Doc에서 직접 PII 추출 – 텍스트 재분석 불필요"""
    text = doc.text

    emails = EMAIL_RE.findall(text.lower())
    phones = PHONE_RE.findall(text)
    names = [e.text for e in doc.ents if e.label_ == "PERSON"]
    locs = [e.text for e in doc.ents if e.label_ in ("GPE", "LOC", "ORG")]
    dates = [e.text for e in doc.ents if e.label_ in ("DATE", "TIME")]

    return {
        "email": list(set(emails)),
        "phone": list(set(phones)),
        "name": list(set(names)),
        "loc": list(set(locs)),
        "date": list(set(dates)),
    }


# ---------- maildir 스캔 ----------
def walk_maildir(root: Path):
    for dp, _, fs in os.walk(root):
        for fn in fs:
            yield Path(dp) / fn


# ---------- 메인 ----------
def main(args):
    root = Path(args.root).expanduser()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 메일 파싱 & 레코드 수집 (I/O)
    records: List[Tuple[Dict, str]] = []
    files = list(walk_maildir(root))
    random.shuffle(files)  # 랜덤 샘플링·로드 밸런싱

    for fp in tqdm(files, desc="Read mails"):
        try:
            mail = mailparser.parse_from_file(fp)
        except Exception:
            continue

        body = clean_body(mail.body or "")
        if not body:  # 빈 본문 제외
            continue

        hdr = extract_headers(mail)
        records.append((hdr, body))

    if not records:
        print("❗ 추출할 메일 본문이 없습니다.")
        return

    # 2) 병렬 spaCy NER
    nlp.max_length = 2011420  # 최대 길이 확장
    docs = nlp.pipe(
        (b for _, b in records),
        batch_size=args.batch,
        n_process=args.workers,
    )

    # 3) 결과 직렬화
    kept = 0
    with out_path.open("w") as fo:
        for (hdr, body), doc in tqdm(
            zip(records, docs), total=len(records), desc="Write JSONL"
        ):
            pii = extract_pii_from_doc(doc)
            json.dump({"headers": hdr, "pii": pii, "text": body}, fo, ensure_ascii=False)
            fo.write("\n")
            kept += 1

    size_mb = out_path.stat().st_size / 1e6
    print(f"✅ saved {kept:,} mails  → {out_path} ({size_mb:.1f} MB)")
    # enron_pii_to_flat_dict()

def enron_pii_to_flat_dict():
    from collections import defaultdict
    import json
    pii_dict = defaultdict(set)
    print("파일 열기 시도")
    with open("../data/v6_enron_pii.jsonl") as f:
        print("파일 엶")
        for line in f:
            rec = json.loads(line)
            for key, values in rec["pii"].items():
                for v in values:
                    pii_dict[key].add(v.lower())  # 소문자 정규화

    pii_dict = {k: set(v) for k, v in pii_dict.items()}
    pii_json = {k: list(v) for k, v in pii_dict.items()}
    ud = {}
    for key in pii_json:
        ud[f"{key}_len"] = len(pii_json[key])
        print(f"{key}_len", len(pii_json[key]))
    pii_json.update(ud)
    print(pii_dict)
    with open("../data/pii_json.json", 'w') as f:
        json.dump(pii_json, f, indent=2)


# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="maildir 루트 폴더")
    ap.add_argument("--out", required=True, help="출력 JSONL 경로")
    ap.add_argument("--batch", type=int, default=64, help="spaCy pipe batch size")
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="NER 병렬 프로세스 수 (기본: 모든 코어)",
    )
    args = ap.parse_args()
    main(args)
