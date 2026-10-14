from __future__ import annotations

import json
from pathlib import Path


SOURCE_DIR = Path(
    "data/processed/direct_collection_literature_review_20260924/sources"
)

KEYWORDS = (
    "sham",
    "yoked",
    "placebo",
    "feedback",
    "self-paced",
    "asynchronous",
    "false positive",
    "false alarm",
    "intersession",
    "inter-session",
    "day-to-day",
    "longitudinal",
    "movement onset",
    "motor intention",
    "continuous",
)


def main() -> None:
    lines: list[str] = []
    for path in sorted(SOURCE_DIR.glob("europepmc_*_narrow.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = data["reconciliation"]
        lines.append(
            f"\n{path.name} total={rec['expected_total']} "
            f"retrieved={rec['retrieved_total']} complete={rec['complete']}"
        )
        selected = [
            record
            for record in data["records"]
            if any(
                keyword in (record.get("title") or "").lower()
                for keyword in KEYWORDS
            )
        ]
        for record in selected[:60]:
            lines.append(
                "- {title} | {year} | {doi} | cited={cited}".format(
                    title=record.get("title"),
                    year=record.get("pubYear"),
                    doi=record.get("doi"),
                    cited=record.get("citedByCount"),
                )
            )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
