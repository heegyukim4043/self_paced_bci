"""Download all Lee2019 artifact recordings with resumable HTTPS transfers."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

import requests

BASE_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/"
    "pub/10.5524/100001_101000/100542"
)


def download_one(output_root: Path, session: int, subject: int) -> str:
    relative = (
        f"session{session}/s{subject}/"
        f"sess{session:02d}_subj{subject:02d}_EEG_Artifact.mat"
    )
    target = output_root / relative
    partial = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return f"exists {relative}"

    start = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with requests.get(
        f"{BASE_URL}/{relative}",
        headers=headers,
        stream=True,
        timeout=(30, 120),
    ) as response:
        response.raise_for_status()
        if start and response.status_code != 206:
            start = 0
        mode = "ab" if start else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(target)
    return f"complete {relative} {target.stat().st_size}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/lee2019_artifact"))
    parser.add_argument("--parallel", type=int, default=6)
    args = parser.parse_args()

    jobs = [(session, subject) for session in (1, 2) for subject in range(1, 55)]
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(download_one, args.output_root, session, subject): (session, subject)
            for session, subject in jobs
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            session, subject = futures[future]
            try:
                print(f"[{index}/{len(jobs)}] {future.result()}", flush=True)
            except Exception as error:
                message = f"session={session} subject={subject}: {error!r}"
                failures.append(message)
                print(f"[{index}/{len(jobs)}] FAILED {message}", flush=True)

    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Lee2019 Artifact download complete: {len(jobs)}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
