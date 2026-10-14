"""Download public Lee2019 MI MAT files directly onto Linux/Ubuntu.

Uses curl's resumable .part transfer; a file is promoted to its final name
only after the expected MATLAB variables can be read.  No data enter Git.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from scipy.io import whosmat


BASE_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
    "10.5524/100001_101000/100542"
)


def relative_path(session: int, subject: int) -> Path:
    if session not in (1, 2) or subject not in range(1, 55):
        raise ValueError("expected session 1/2 and subject 1..54")
    return Path(f"session{session}") / f"s{subject}" / (
        f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    )


def validate_mat(path: Path) -> None:
    names = {name for name, _, _ in whosmat(path)}
    missing = {"EEG_MI_train", "EEG_MI_test"} - names
    if missing:
        raise ValueError(f"{path}: missing MATLAB variables {sorted(missing)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=Path("data/raw/lee2019_mi"))
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--sha256", action="store_true")
    args = ap.parse_args()
    if len(set(args.subjects)) != len(args.subjects) or len(set(args.sessions)) != len(args.sessions):
        raise ValueError("duplicate subject or session")
    paths = [relative_path(session, subject)
             for session in args.sessions for subject in args.subjects]
    if args.dry_run and args.verify_only:
        raise ValueError("choose dry-run or verify-only")
    if args.sha256 and args.dry_run:
        raise ValueError("sha256 needs downloaded files")
    validated = 0
    for relative in paths:
        dest = args.output_root / relative
        url = args.base_url.rstrip("/") + "/" + relative.as_posix()
        if args.dry_run:
            print(f"{url} -> {dest}")
            continue
        if not dest.exists():
            if args.verify_only:
                raise FileNotFoundError(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name + ".part")
            subprocess.run([
                "curl", "--location", "--fail", "--retry", "5",
                "--retry-delay", "5", "--continue-at", "-",
                "--output", str(partial), url,
            ], check=True)
            validate_mat(partial)
            partial.replace(dest)
        else:
            validate_mat(dest)
        validated += 1
        suffix = f" sha256={sha256(dest)}" if args.sha256 else ""
        print(f"[{validated}/{len(paths)}] {relative} bytes={dest.stat().st_size}{suffix}", flush=True)
    if not args.dry_run:
        print(f"validated {validated}/{len(paths)} MI files")


if __name__ == "__main__":
    main()
