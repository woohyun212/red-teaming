"""
maildir_to_jsonl.py  (v2: PII-friendly)

변경 내역
---------
1. `strip_forward()`             : "----- Forwarded by ..." 블록 제거
2. 본문 전처리 시 헤더·본문을 한 줄(text)로 합치기
3. 길이 필터링(토큰 20~4096) + 중복 제거(md5)
4. 출력 JSONL은 `{"text": "..."} ` 단일 필드 (모델 학습에 바로 사용)
   └ 필요 시 --keep_meta 플래그로 (path/from/to/subject)도 보존

python ./w00/maildir_to_jsonl.py --root ./maildir --out ./w00/data/email_pii_meta.jsonl --keep_meta
"""
from pprint import pprint
import argparse, json, os, re, mailparser, hashlib, random
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

MAX_TOK = 4096


def strip_forward(text: str) -> str:
    """Forwarded 메일 체인 이하를 잘라낸다."""
    return re.split(r'-{5,}\s*Forwarded by', text, 1, flags=re.IGNORECASE)[0]


def preprocess(body: str) -> str:
    # body = strip_forward(body)
    body = re.sub(r"\r\n", "\n", body)  # CRLF → LF
    body = re.sub(r"\n{3,}", "\n\n", body)  # 연속 공백 줄 축소
    return body.strip()


def scan_maildir(root: Path):
    return [Path(dp) / fn
            for dp, _, fs in os.walk(root)
            for fn in fs]


def main(root: str, out: str, keep_meta: bool):
    root, out = Path(root).expanduser(), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    eml_files = scan_maildir(root)
    random.shuffle(eml_files)  # 중복 필터 편향 방지
    print(f"Found {len(eml_files):,} files under {root}")

    seen = set()
    kept = 0
    with out.open("w") as fo:
        for fp in tqdm(eml_files, desc="parsing"):
            try:
                mail = mailparser.parse_from_file(fp)
            except Exception as e:
                tqdm.write(f"⚠️  skip {fp}: {e}")
                continue

            body = preprocess(mail.body or "")
            # if not body:
            #     continue
            # pprint(mail.headers)
            mail_headers = mail.headers
            text = (f"From: {mail.from_[0][1] if mail.from_ else ''}\n"
                    f"To: {mail.to[0][1] if mail.to else ''}\n"
                    f"X-From: {mail_headers.get('X-From', '')}\n"
                    f"X-To: {mail_headers.get('X-To', '')}\n"
                    f"X-cc: {mail_headers.get('X-cc', '')}\n"
                    f"X-bcc: {mail_headers.get('X-bcc', '')}\n"
                    f"X-Origin: {mail_headers.get('X-Origin', '')}\n"
                    f"X-Folder: {mail_headers.get('X-Folder', '')}\n"
                    f"X-FileName: {mail_headers.get('X-FileName', '')}\n"
                    f"Subject: {mail.subject or ''}\n\n{body}")
            # print(text)
            # 길이 & 중복 필터
            # ntok = len(text.split())
            ntok = len(tok(text).input_ids)
            if ntok < 20 or ntok > MAX_TOK:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            sample = {"text": text}
            if keep_meta:
                sample.update({
                    "path": str(fp.relative_to(root)),
                    "from": mail.from_[0][1] if mail.from_ else "",
                    "to": mail.to[0][1] if mail.to else "",
                    "date": mail.date.isoformat() if mail.date else "",
                    "subject": mail.subject or "",
                })

            json.dump(sample, fo, ensure_ascii=False)
            fo.write("\n")
            kept += 1

    size_mb = out.stat().st_size / 1e6
    print(f"✅ Saved {kept:,} samples → {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="maildir 루트 경로")
    p.add_argument("--out", required=True, help="출력 JSONL 경로")
    p.add_argument("--keep_meta", action="store_true",
                   help="메타데이터(path/from/to/subject) 필드 유지")
    args = p.parse_args()
    main(args.root, args.out, args.keep_meta)
