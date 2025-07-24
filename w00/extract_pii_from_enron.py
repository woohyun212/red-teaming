#!/usr/bin/env python
"""
extract_pii_from_enron.py
-------------------------
python w00/extract_pii_from_enron.py \
    --root ./maildir \
    --out  data/v4_enron_pii.jsonl
"""
import os, re, json, argparse, random
from pathlib import Path
from tqdm import tqdm
import mailparser
import spacy           # python -m spacy download en_core_web_sm

# ---------- 정규식 ----------
EMAIL_RE  = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE  = re.compile(r"\b(?:\+?1[-\s.]*)?\(?\d{3}\)?[-\s.]?\d{3,4}[-\s.]?\d{4}\b")

# ---------- spaCy NER ----------
try:
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
except OSError:
    raise SystemExit("❗ spaCy 모델(en_core_web_sm)이 설치돼 있지 않습니다.\n"
                     "→  python -m spacy download en_core_web_sm")

# ---------- 본문 전처리 ----------
def clean_body(text:str)->str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------- PII 추출 ----------
def extract_pii(text:str)->dict:
    emails  = EMAIL_RE.findall(text.lower())
    phones  = PHONE_RE.findall(text)
    doc     = nlp(text)
    names   = [e.text for e in doc.ents if e.label_=="PERSON"]
    locs    = [e.text for e in doc.ents if e.label_ in ("GPE","LOC","ORG")]
    dates   = [e.text for e in doc.ents if e.label_ in ("DATE","TIME")]
    return {"email":list(set(emails)),
            "phone":list(set(phones)),
            "name":list(set(names)),
            "loc": list(set(locs)),
            "date":list(set(dates))}

# ---------- maildir 스캔 ----------
def walk_maildir(root:Path):
    for dp,_,fs in os.walk(root):
        for fn in fs:
            yield Path(dp)/fn

# ---------- 메인 ----------
def main(args):
    root = Path(args.root).expanduser()
    out  = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = list(walk_maildir(root))
    random.shuffle(files)
    kept = 0
    with out.open("w") as fo:
        for fp in tqdm(files, desc="parse"):
            try:
                mail = mailparser.parse_from_file(fp)
            except Exception as e:
                tqdm.write(f"skip {fp}: {e}")
                continue

            body = clean_body(mail.body or "")
            if not body:                       # 빈 본문 제외
                continue

            # ----- 헤더 PII -----
            hdr = {
              "from_email" : mail.from_[0][1] if mail.from_ else "",
              "to_email"   : [t[1] for t in mail.to] if mail.to else [],
              "cc"         : [c[1] for c in mail.cc] if mail.cc else [],
              "bcc"        : [b[1] for b in mail.bcc] if mail.bcc else [],
              "x_from"     : mail.headers.get("X-From",""),
              "x_to"       : mail.headers.get("X-To",""),
              "x_cc"       : mail.headers.get("X-cc",""),
              "x_bcc"      : mail.headers.get("X-bcc",""),
              "x_origin"   : mail.headers.get("X-Origin",""),
              "x_folder"   : mail.headers.get("X-Folder",""),
              "x_filename" : mail.headers.get("X-FileName",""),
              "date"       : mail.date.isoformat() if mail.date else ""
            }

            # ----- 본문 PII -----
            pii = extract_pii(body)

            sample = {
              "headers" : hdr,
              "pii"     : pii,
              "text"    : body
            }
            json.dump(sample, fo, ensure_ascii=False)
            fo.write("\n")
            kept += 1

    print(f"✅ saved {kept:,} mails  → {out} ({out.stat().st_size/1e6:.1f} MB)")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="maildir 루트 폴더")
    ap.add_argument("--out",  required=True, help="출력 JSONL 경로")
    args = ap.parse_args()
    main(args)
