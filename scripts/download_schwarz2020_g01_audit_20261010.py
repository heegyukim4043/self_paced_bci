"""Download fixed BNCI 001-2020 sample files for compatibility audit.

This is a raw-data preflight, not a detector experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://lampx.tugraz.at/~bci/database/001-2020/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", choices=("G01", "V01", "H01"), default="G01")
    participant = parser.parse_args().participant
    url = f"{BASE_URL}{participant}.mat"
    target = ROOT / "data" / "raw" / "schwarz2020_audit_20261010" / f"{participant}.mat"
    manifest = ROOT / "data" / "processed" / f"schwarz2020_{participant.lower()}_manifest_20261010.json"
    if target.exists() or manifest.exists():
        raise FileExistsError("Raw audit target or manifest already exists; preserve prior download")
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".mat.partial")
    if partial.exists():
        raise FileExistsError(partial)
    hasher = hashlib.sha256()
    nbytes = 0
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            declared = int(response.headers["Content-Length"])
            with partial.open("xb") as dest:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    dest.write(chunk)
                    hasher.update(chunk)
                    nbytes += len(chunk)
        if nbytes != declared or nbytes < 50_000_000:
            raise AssertionError(f"Incomplete {participant}: {nbytes} bytes vs {declared}")
        partial.replace(target)
    except Exception:
        # Keep the partial file for diagnosis; do not silently reuse it.
        raise
    record = {
        "status": f"single original BNCI {participant} compatibility sample; not cohort audit",
        "url": url,
        "file": str(target.relative_to(ROOT)),
        "bytes": nbytes,
        "sha256": hasher.hexdigest(),
    }
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
