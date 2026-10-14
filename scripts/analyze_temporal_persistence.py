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
from uncued_mi.decision import moving_average_dwell
from uncued_mi.features import decision_subsample


def second_half(intervals: np.ndarray) -> np.ndarray:
    intervals = np.asarray(intervals).reshape(-1, 2)
    return intervals[len(intervals) // 2 :]


def partial_spearman(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray
) -> float:
    controls = np.asarray(controls)
    if controls.ndim == 1:
        controls = controls[:, None]
    control_design = np.c_[
        np.ones(len(controls)),
        np.column_stack([rankdata(controls[:, index]) for index in range(controls.shape[1])]),
    ]
    x_residual = rankdata(x) - control_design @ np.linalg.lstsq(
        control_design, rankdata(x), rcond=None
    )[0]
    y_residual = rankdata(y) - control_design @ np.linalg.lstsq(
        control_design, rankdata(y), rcond=None
    )[0]
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def run_features(above: np.ndarray, step: float) -> tuple[float, float, float]:
    above = np.asarray(above, dtype=bool)
    if not len(above):
        return 0.0, 0.0, 0.0
    transitions = np.diff(np.r_[False, above, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    lengths = stops - starts
    return (
        float(np.mean(above)),
        float(np.max(lengths) * step) if len(lengths) else 0.0,
        float(len(lengths)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument(
        "--loso-results",
        type=Path,
        default=Path("data/processed/tier0_revised_session1.json"),
    )
    parser.add_argument(
        "--auc-results",
        type=Path,
        default=Path("data/processed/tier0_auc_excess_subset20.json"),
    )
    parser.add_argument(
        "--null-results",
        type=Path,
        default=Path(
            "data/processed/detection_circular_shift_null_second_half.json"
        ),
    )
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--target-fp", type=float, default=2.0)
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tier0_temporal_persistence.json"),
    )
    args = parser.parse_args()
    results = json.loads(args.loso_results.read_text(encoding="utf-8"))
    thresholds = {
        int(row["subject"]): float(row["operating_point"]["threshold"])
        for row in results
        if row["session"] == args.session
        and row["decision"] == "ma_dwell"
        and row["calibration"] == "idle_feature_alignment"
        and float(row["target_fp_setting"]) == args.target_fp
    }
    auc_rows = json.loads(args.auc_results.read_text(encoding="utf-8"))["rows"]
    auc_lookup = {
        int(row["subject"]): float(row["second_half_window_auc"])
        for row in auc_rows
    }
    null_rows = json.loads(args.null_results.read_text(encoding="utf-8"))["rows"]
    excess_lookup = {
        int(row["subject"]): float(row["observed_minus_null_median"])
        for row in null_rows
        if row["session"] == args.session
        and row["decision"] == "ma_dwell"
        and float(row["target_fp_setting"]) == args.target_fp
    }
    rows = []
    for path in sorted(
        (args.cache_root / f"session{args.session}").glob(
            "sub-*_tier0_features.npz"
        )
    ):
        with np.load(path, allow_pickle=False) as cache:
            subject = int(cache["subject"])
            model = LocationScale.fit(cache["pre_features"], axis=0)
            channels = tuple(str(value) for value in cache["channels"])
            raw = tier0_from_features(
                cache["continuous_features"], channels, model
            )
            times, raw = decision_subsample(
                cache["continuous_times"], raw, args.decision_step
            )
            statistic = moving_average_dwell(
                raw,
                thresholds[subject],
                args.decision_step,
                ma_seconds=0.5,
                dwell_seconds=0.3,
                refractory_seconds=1.0,
            ).statistic
            event_features = []
            for start, stop in second_half(cache["command_intervals"]):
                keep = (times >= start) & (times < stop)
                event_features.append(
                    run_features(
                        statistic[keep] >= thresholds[subject],
                        args.decision_step,
                    )
                )
            values = np.asarray(event_features)
            rows.append(
                {
                    "subject": subject,
                    "window_auc": auc_lookup[subject],
                    "excess_tpr": excess_lookup[subject],
                    "hit_density_mean": float(np.mean(values[:, 0])),
                    "longest_run_seconds_mean": float(np.mean(values[:, 1])),
                    "burst_count_mean": float(np.mean(values[:, 2])),
                }
            )
    auc_values = np.asarray([row["window_auc"] for row in rows])
    excess = np.asarray([row["excess_tpr"] for row in rows])
    association = {}
    hit_density = np.asarray([row["hit_density_mean"] for row in rows])
    for key in ("hit_density_mean", "longest_run_seconds_mean", "burst_count_mean"):
        values = np.asarray([row[key] for row in rows])
        rho, p_value = spearmanr(values, excess)
        association[key] = {
            "spearman_rho_with_excess": float(rho),
            "spearman_p": float(p_value),
            "partial_spearman_controlling_auc": partial_spearman(
                values, excess, auc_values
            ),
        }
        if key != "hit_density_mean":
            association[key][
                "partial_spearman_controlling_auc_and_hit_density"
            ] = partial_spearman(
                values,
                excess,
                np.c_[auc_values, hit_density],
            )
    payload = {"association": association, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(association, indent=2))


if __name__ == "__main__":
    main()
