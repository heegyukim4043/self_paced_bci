from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.features import sample_mask


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.r_[positive, negative]
    ranks = rankdata(values, method="average")
    n_positive = len(positive)
    return float(
        (
            np.sum(ranks[:n_positive])
            - n_positive * (n_positive + 1) / 2
        )
        / (n_positive * len(negative))
    )


def second_half(intervals: np.ndarray) -> np.ndarray:
    intervals = np.asarray(intervals).reshape(-1, 2)
    return intervals[len(intervals) // 2 :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument(
        "--null-results",
        type=Path,
        default=Path(
            "data/processed/detection_circular_shift_null_second_half.json"
        ),
    )
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--decision", default="ma_dwell")
    parser.add_argument("--target-fp", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tier0_auc_excess_subset20.json"),
    )
    args = parser.parse_args()
    null_rows = json.loads(args.null_results.read_text(encoding="utf-8"))["rows"]
    excess = {
        int(row["subject"]): float(row["observed_minus_null_median"])
        for row in null_rows
        if row["session"] == args.session
        and row["decision"] == args.decision
        and float(row["target_fp_setting"]) == args.target_fp
    }
    rows = []
    paths = sorted(
        (args.cache_root / f"session{args.session}").glob(
            "sub-*_tier0_features.npz"
        )
    )
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            subject = int(cache["subject"])
            model = LocationScale.fit(cache["pre_features"], axis=0)
            channels = tuple(str(value) for value in cache["channels"])
            scores = tier0_from_features(
                cache["continuous_features"], channels, model
            )
            times = cache["continuous_times"]
            command_mask = sample_mask(
                times, second_half(cache["command_intervals"]).tolist()
            )
            idle_mask = sample_mask(
                times, second_half(cache["iti_intervals"]).tolist()
            )
            rows.append(
                {
                    "subject": subject,
                    "second_half_window_auc": auc(
                        scores[command_mask], scores[idle_mask]
                    ),
                    "fp2_excess_tpr": excess[subject],
                }
            )
    x = np.asarray([row["second_half_window_auc"] for row in rows])
    y = np.asarray([row["fp2_excess_tpr"] for row in rows])
    rho, rho_p = spearmanr(x, y)
    auc_bins = [
        ("<0.60", -np.inf, 0.60),
        ("0.60-0.70", 0.60, 0.70),
        ("0.70-0.80", 0.70, 0.80),
        (">=0.80", 0.80, np.inf),
    ]
    binned = []
    for label, lower, upper in auc_bins:
        selected = y[(x >= lower) & (x < upper)]
        binned.append(
            {
                "auc_bin": label,
                "n": int(len(selected)),
                "median_excess_tpr": (
                    float(np.median(selected)) if len(selected) else None
                ),
                "iqr": (
                    [float(value) for value in np.quantile(selected, [0.25, 0.75])]
                    if len(selected)
                    else None
                ),
            }
        )

    # Fixed stratified random subset: five subjects from each AUC quartile.
    order = np.argsort(x, kind="stable")
    strata = np.array_split(order, 4)
    rng = np.random.default_rng(args.seed)
    selected_indices = np.concatenate(
        [rng.choice(stratum, size=5, replace=False) for stratum in strata]
    )
    selected_subjects = sorted(int(rows[index]["subject"]) for index in selected_indices)
    for row in rows:
        row["tier1_subset20"] = row["subject"] in selected_subjects

    payload = {
        "session": args.session,
        "decision": args.decision,
        "target_fp": args.target_fp,
        "association": {
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "auc_binned_excess_tpr": binned,
            "note": "No linear extrapolation outside the observed AUC range.",
        },
        "subset_selection": {
            "rule": "5 subjects sampled without replacement from each Tier-0 AUC quartile",
            "seed": args.seed,
            "subjects": selected_subjects,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
