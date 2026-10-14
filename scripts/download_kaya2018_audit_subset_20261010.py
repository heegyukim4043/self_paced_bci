"""Download three fixed, original Kaya2018 CLA examples for a metadata/raw audit.

No overwrite: a pre-existing file must match the Figshare byte count.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "kaya2018_audit_20261010"
MANIFEST = ROOT / "data" / "processed" / "kaya2018_audit_subset_manifest_20261010.json"
ARTICLES = [5554504, 5554531, 6818792, 5579680]  # CLA A/F/J + author acquisition scripts


def main():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    records = []
    for article_id in ARTICLES:
        source = f"https://api.figshare.com/v2/articles/{article_id}"
        response = session.get(source, timeout=30)
        response.raise_for_status()
        article = response.json()
        if len(article["files"]) != 1:
            raise AssertionError(f"expected one source file: {article_id}")
        meta = article["files"][0]
        path = OUT / meta["name"]
        if path.exists():
            if path.stat().st_size != meta["size"]:
                raise FileExistsError(f"existing file has a different size: {path}")
        else:
            # A failed transfer remains a visible .part file and is never
            # silently treated as a complete raw dataset.
            part = path.with_name(path.name + ".part")
            if part.exists():
                raise FileExistsError(part)
            with session.get(meta["download_url"], stream=True, timeout=120) as download:
                download.raise_for_status()
                with part.open("xb") as fh:
                    for chunk in download.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            if part.stat().st_size != meta["size"]:
                raise AssertionError(f"truncated download: {part}")
            part.replace(path)
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        record = {"article_id": article_id, "article_title": article["title"],
                  "source": source, "download_url": meta["download_url"],
                  "file": str(path.relative_to(ROOT)), "bytes": meta["size"],
                  "sha256": digest.hexdigest()}
        print(f"verified {record['file']} ({record['bytes']} bytes)", flush=True)
        records.append(record)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"subset": "CLA A standard; CLA F standard; CLA J Inter; author scripts",
                                    "selection_before_marker_inspection": True,
                                    "records": records}, indent=2), encoding="utf-8")
    print(MANIFEST)


if __name__ == "__main__":
    main()
