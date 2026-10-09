#!/usr/bin/env python3
"""Download literature candidate PDFs from literature_inventory.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT / "literature_inventory.csv"
PDF_DIR = ROOT / "literature" / "pdfs"
TIMEOUT = 120
HEADERS = {"User-Agent": "xai-rag-depression/1.0 (research; pdf download)"}


def download_pdf(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  exists: {dest.name}")
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        content = r.content
        if not content.startswith(b"%PDF"):
            print(f"  SKIP {dest.name}: not a PDF ({len(content)} bytes)")
            return False
        dest.write_bytes(content)
        print(f"  saved: {dest.name} ({len(content):,} bytes)")
        return True
    except Exception as e:
        print(f"  FAIL {dest.name}: {e}")
        return False


def main() -> int:
    if not INVENTORY.exists():
        print(f"Missing {INVENTORY}")
        return 1

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(INVENTORY)
    inv = inv.drop_duplicates(subset=["file_name"], keep="first")

    ok = fail = 0
    print(f"Downloading {len(inv)} PDFs -> {PDF_DIR.resolve()}\n")
    for _, row in inv.iterrows():
        dest = PDF_DIR / row["file_name"]
        if download_pdf(row["PDF_URL"], dest):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} ok, {fail} failed -> {PDF_DIR}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
