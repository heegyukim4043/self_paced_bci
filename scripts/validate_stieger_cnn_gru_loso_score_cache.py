"""Integrity validation for Stieger subject-LOSO CNN-GRU score caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/stieger2021_cnn_gru_loso_scores"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/stieger2021_cnn_gru_loso_score_cache_validation.json"))
    args = parser.parse_args()
    files = sorted(args.cache_root.glob("session*/sub-*.npz"))
    if len(files) != 30:
        raise ValueError(f"expected 30 caches, found {len(files)}")
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as cache:
            audit = json.loads(str(cache["audit_json"]))
            subject = int(audit["subject"])
            sources = {int(row["subject"]) for row in audit["source_sampling"]["per_source_subject"]}
            checks = {
                "target_excluded": subject not in sources,
                "source_subject_count_19": len(sources) == 19,
                "seed_count_3": cache["seed_scores"].shape[0] == 3,
                "ensemble_exact": bool(np.max(np.abs(cache["scores"] - cache["seed_scores"].mean(axis=0))) <= 1e-7),
                "finite_scores": bool(np.all(np.isfinite(cache["seed_scores"]))),
                "offsets_match": int(cache["trial_offsets"][-1]) == len(cache["scores"]),
                "causality_reset": all(row["passed"] for row in audit["causality_reset_audits"]),
                "loso_regime": "LOSO" in audit["training_regime"],
            }
            if not all(checks.values()):
                raise AssertionError(f"{path}: {checks}")
            rows.append({"path": str(path), "subject": subject, "session": int(audit["session"]), "checks": checks})
    payload = {"passed": True, "cache_count": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "cache_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
