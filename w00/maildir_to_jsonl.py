"""
maildir_to_jsonl.py
-------------------
`python maildir_to_jsonl.py --root ./maildir --out data/email_corpus.jsonl`
"""

import argparse, json, os, re, mailparser
from pathlib import Path
from tqdm import tqdm

def preprocess(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)            # CRLF → LF
    text = re.sub(r"\n{3,}", "\n\n", text)        # 연속 공백 줄 축소
    return text.strip()

def scan_maildir(root: Path):
    eml_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            # 숫자 파일·.txt·.eml 전부 포함
            eml_paths.append(Path(dirpath) / fn)
    return eml_paths

def main(root: str, out: str):
    root = Path(root).expanduser()
    out  = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    eml_files = scan_maildir(root)
    print(f"Found {len(eml_files):,} files under {root}")

    with out.open("w") as f_out:
        for fp in tqdm(eml_files, desc="parsing"):
            try:
                mail = mailparser.parse_from_file(fp)
            except Exception as e:
                # 손상된 파일 건너뜀
                tqdm.write(f"⚠️  skip {fp}: {e}")
                continue

            body = preprocess(mail.body or "")
            sample = {
                "path" : str(fp.relative_to(root)),
                "from" : mail.from_[0][1] if mail.from_ else "",
                "to"   : mail.to[0][1] if mail.to else "",
                "date" : mail.date.isoformat() if mail.date else "",
                "subject": mail.subject or "",
                "text" : body
            }
            json.dump(sample, f_out, ensure_ascii=False, indent=2)
            f_out.write("\n")

    print(f"✅ Saved JSONL → {out} ({out.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="maildir 루트 경로")
    p.add_argument("--out", required=True, help="출력 JSONL 경로")
    args = p.parse_args()
    main(args.root, args.out)
