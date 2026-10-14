from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.features import decision_subsample, sample_mask


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def normalizer(caches: list[dict[str, np.ndarray]]) -> LocationScale:
    return LocationScale.fit(
        np.concatenate([cache["pre_features"] for cache in caches], axis=0),
        axis=0,
    )


def scores(
    cache: dict[str, np.ndarray], model: LocationScale, step_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    channels = tuple(str(item) for item in cache["channels"])
    values = tier0_from_features(cache["continuous_features"], channels, model)
    return decision_subsample(cache["continuous_times"], values, step_seconds)


def percentile(values: np.ndarray, threshold: float) -> float:
    """Empirical CDF at threshold; high values imply a conservative gate."""
    return float(np.mean(np.asarray(values) <= threshold))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/processed/tier0_revised_session1.json"),
    )
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/operating_point_collapse_session1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/processed/operating_point_collapse_session1.png"),
    )
    args = parser.parse_args()

    paths = sorted(
        (args.cache_root / f"session{args.session}").glob(
            "sub-*_tier0_features.npz"
        )
    )
    caches = {int(load_cache(path)["subject"]): load_cache(path) for path in paths}
    result_rows = json.loads(args.results.read_text(encoding="utf-8"))
    lookup = {
        (int(row["subject"]), float(row["target_fp_setting"]), row["calibration"]): row
        for row in result_rows
        if row["decision"] == "threshold"
    }

    rows: list[dict] = []
    for subject, target in sorted(caches.items()):
        source = [cache for key, cache in caches.items() if key != subject]
        models = {
            "no_calibration": normalizer(source),
            "idle_feature_alignment": normalizer([target]),
        }
        for calibration, model in models.items():
            times, target_scores = scores(target, model, args.decision_step)
            idle_mask = sample_mask(times, target["iti_intervals"].tolist())
            command_mask = sample_mask(times, target["command_intervals"].tolist())
            idle = target_scores[idle_mask]
            command = target_scores[command_mask]
            for target_fp in (1.0, 2.0, 4.0):
                result = lookup[(subject, target_fp, calibration)]
                threshold = float(result["operating_point"]["threshold"])
                idle_std = float(np.std(idle, ddof=1))
                rows.append(
                    {
                        "subject": subject,
                        "session": args.session,
                        "calibration": calibration,
                        "target_fp_setting": target_fp,
                        "source_threshold": threshold,
                        "target_idle_threshold_percentile": percentile(idle, threshold),
                        "target_idle_threshold_z": (
                            float((threshold - np.mean(idle)) / idle_std)
                            if idle_std > 0
                            else None
                        ),
                        "target_command_threshold_percentile": percentile(
                            command, threshold
                        ),
                        "target_idle_exceedance_fraction": float(np.mean(idle > threshold)),
                        "target_command_exceedance_fraction": float(
                            np.mean(command > threshold)
                        ),
                        "target_fp_per_minute": result["iti"]["fp_per_minute"],
                        "target_true_positive_rate": result["iti"][
                            "true_positive_rate"
                        ],
                        "dead_gate": bool(
                            result["iti"]["fp_per_minute"] == 0
                            and result["iti"]["true_positive_rate"] == 0
                        ),
                    }
                )

    summary: dict[str, dict] = {}
    for calibration in ("no_calibration", "idle_feature_alignment"):
        for target_fp in (1.0, 2.0, 4.0):
            selected = [
                row
                for row in rows
                if row["calibration"] == calibration
                and row["target_fp_setting"] == target_fp
            ]
            idle_percentiles = np.asarray(
                [row["target_idle_threshold_percentile"] for row in selected]
            )
            summary[f"{calibration}:fp{target_fp:g}"] = {
                "n": len(selected),
                "threshold_percentile_median": float(np.median(idle_percentiles)),
                "threshold_percentile_iqr": [
                    float(value) for value in np.quantile(idle_percentiles, [0.25, 0.75])
                ],
                "threshold_z_median": float(
                    np.median(
                        [row["target_idle_threshold_z"] for row in selected]
                    )
                ),
                "threshold_z_iqr": [
                    float(value)
                    for value in np.quantile(
                        [row["target_idle_threshold_z"] for row in selected],
                        [0.25, 0.75],
                    )
                ],
                "dead_gate_fraction": float(
                    np.mean([row["dead_gate"] for row in selected])
                ),
                "median_fp_per_minute": float(
                    np.median([row["target_fp_per_minute"] for row in selected])
                ),
                "median_tpr": float(
                    np.median([row["target_true_positive_rate"] for row in selected])
                ),
            }

    payload = {"summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(json.dumps(summary, indent=2))
        print("matplotlib unavailable; JSON written without figure")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
    colors = {"no_calibration": "#C44E52", "idle_feature_alignment": "#4C72B0"}
    labels = {
        "no_calibration": "No calibration",
        "idle_feature_alignment": "Idle alignment",
    }
    for axis, target_fp in zip(axes, (1.0, 2.0, 4.0)):
        for calibration in colors:
            selected = [
                row
                for row in rows
                if row["calibration"] == calibration
                and row["target_fp_setting"] == target_fp
            ]
            axis.scatter(
                [row["target_idle_threshold_z"] for row in selected],
                [row["target_true_positive_rate"] for row in selected],
                s=20,
                alpha=0.7,
                color=colors[calibration],
                label=labels[calibration],
            )
        axis.set_title(f"Source target: {target_fp:g} FP/min")
        axis.set_xlabel("(source threshold - target ITI mean)\n/ target ITI SD")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Target TPR")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Operating-point collapse, Lee2019 session {args.session}", fontsize=11
    )
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
