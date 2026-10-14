"""Download evaluable Cho2017 subjects directly from GigaDB."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

import requests

BASE_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/"
    "pub/10.5524/100001_101000/100295/mat_data"
)
EXCLUDED = {29, 34}


def download_one(output_root: Path, subject: int) -> str:
    name = f"s{subject:02d}.mat"
    target = output_root / name
    partial = target.with_suffix(".mat.part")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return f"exists {name}"

    start = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with requests.get(
        f"{BASE_URL}/{name}",
        headers=headers,
        stream=True,
        timeout=(30, 180),
    ) as response:
        response.raise_for_status()
        if start and response.status_code != 206:
            start = 0
        with partial.open("ab" if start else "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(target)
    return f"complete {name} {target.stat().st_size}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/cho2017"))
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--log", type=Path, default=Path("logs/cho2017_download.log"))
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)

    subjects = [subject for subject in range(1, 53) if subject not in EXCLUDED]
    failures: list[str] = []
    with args.log.open("a", encoding="utf-8") as log:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(download_one, args.output_root, subject): subject
                for subject in subjects
            }
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                subject = futures[future]
                try:
                    message = f"[{index}/{len(subjects)}] {future.result()}"
                except Exception as error:
                    message = f"[{index}/{len(subjects)}] FAILED s{subject:02d}: {error!r}"
                    failures.append(message)
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()

    if failures:
        raise RuntimeError("\n".join(failures))


if __name__ == "__main__":
    main()
