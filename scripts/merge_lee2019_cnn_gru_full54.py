"""Merge chunked Lee2019 CNN-GRU revalidation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_lee2019_cnn_gru import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    rows = []
    seen = set()
    protocols = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocols.append(payload.get("protocol", {}))
        for row in payload["rows"]:
            key = (int(row["subject"]), int(row["session"]))
            if key in seen:
                raise ValueError(f"duplicate row: {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (int(row["session"]), int(row["subject"])))
    sessions = sorted({int(row["session"]) for row in rows})
    expected = {(subject, session) for session in sessions for subject in range(1, 55)}
    missing = expected.difference(seen)
    if missing:
        raise ValueError(f"missing rows: {sorted(missing)[:12]}")

    payload = {
        "protocol": protocols[0] if protocols else {},
        "summary": summarize(rows, sessions, args.bootstrap),
        "rows": rows,
        "source_files": [str(path) for path in args.inputs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
