"""Download PDFs from knowledge_db metadata."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT.parent / "knowledge_db"
META = KB / "metadata" / "selected_documents.jsonl"
PDF_ROOT = KB / "pdfs"
LOG = ROOT / "data" / "download_log.txt"


def download_url(url: str, dest: Path, retries: int = 2) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as e:
            if attempt >= retries:
                LOG.open("a", encoding="utf-8").write(f"FAIL {url} -> {dest}: {e}\n")
                return False
            time.sleep(2)
    return False


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not META.exists():
        raise FileNotFoundError(META)
    ok, skip, fail = 0, 0, 0
    for line in META.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        doc_id = doc.get("doc_id", "")
        url = doc.get("pdf_url", "")
        local = doc.get("local_pdf_path", "")
        if not url:
            skip += 1
            continue
        if local:
            dest = PDF_ROOT / Path(local.replace("pdfs/", "").replace("pdfs\\", ""))
        else:
            dest = PDF_ROOT / "misc" / f"{doc_id}.pdf"
        if dest.exists() and dest.stat().st_size > 1000:
            skip += 1
            continue
        if download_url(url, dest):
            ok += 1
            LOG.open("a", encoding="utf-8").write(f"OK {doc_id} -> {dest}\n")
            print(f"Downloaded {doc_id}")
        else:
            fail += 1
            print(f"Failed {doc_id}")
        time.sleep(0.5)
    print(f"Done: {ok} downloaded, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
