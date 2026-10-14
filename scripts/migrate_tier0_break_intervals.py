from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument("--max-task-iti-seconds", type=float, default=6.0)
    args = parser.parse_args()
    migrated = 0
    for path in sorted(args.cache_root.glob("session*/sub-*_tier0_features.npz")):
        with np.load(path, allow_pickle=False) as source:
            payload = {key: source[key] for key in source.files}
        iti = np.asarray(payload["iti_intervals"]).reshape(-1, 2)
        duration = iti[:, 1] - iti[:, 0]
        payload["iti_intervals"] = iti[duration <= args.max_task_iti_seconds]
        payload["break_intervals"] = iti[duration > args.max_task_iti_seconds]
        temporary = path.with_suffix(".migrating.npz")
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
        migrated += 1
    print(f"migrated {migrated} caches")


if __name__ == "__main__":
    main()
