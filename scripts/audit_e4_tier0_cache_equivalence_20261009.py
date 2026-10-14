"""Technical comparison of 15 stale-start tier-0 caches; no labels/outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/interim/lee2019_tier0/session1"
NEW = ROOT / "data/interim/lee2019_tier0_reaudit_20261009/session1"
OUTPUT = ROOT / "data/processed/e4_tier0_cache_equivalence_20261009.json"
SUBJECTS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19)


def main() -> None:
    rows = []
    for subject in SUBJECTS:
        name = f"sub-{subject:02d}_tier0_features.npz"
        with np.load(OLD / name, allow_pickle=False) as old, np.load(
            NEW / name, allow_pickle=False) as current:
            row = {"subject": subject}
            for stem in ("pre", "post", "continuous"):
                previous_times = np.asarray(old[f"{stem}_times"], float)
                current_times = np.asarray(current[f"{stem}_times"], float)
                offset = int(np.searchsorted(previous_times,
                                             current_times[0] - 1e-5))
                if not np.array_equal(previous_times[offset:], current_times):
                    raise AssertionError(f"time mismatch {subject} {stem}")
                left = np.asarray(old[f"{stem}_features"][offset:])
                right = np.asarray(current[f"{stem}_features"])
                if left.shape != right.shape:
                    raise AssertionError(f"shape mismatch {subject} {stem}")
                row[f"{stem}_old_first_end_s"] = float(previous_times[0])
                row[f"{stem}_new_first_end_s"] = float(current_times[0])
                row[f"{stem}_dropped_windows"] = int(offset)
                row[f"{stem}_max_abs_feature_difference"] = float(
                    np.max(np.abs(left.astype(float) - right.astype(float))))
        rows.append(row)
    result = {
        "scope": "cache feature/time equivalence only; no score, threshold, FP or SEN inspected",
        "n_subjects": len(rows),
        "all_post_current_start_exact": all(
            row[f"{stem}_max_abs_feature_difference"] == 0.0
            for row in rows for stem in ("pre", "post", "continuous")),
        "rows": rows,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
