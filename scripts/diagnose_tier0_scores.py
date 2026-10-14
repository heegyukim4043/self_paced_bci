from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.features import sample_mask


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def auc(command: np.ndarray, idle: np.ndarray) -> float:
    command = np.asarray(command, dtype=float)
    idle = np.asarray(idle, dtype=float)
    values = np.r_[command, idle]
    ranks = rankdata(values, method="average")
    n_pos = len(command)
    n_neg = len(idle)
    if not n_pos or not n_neg:
        return float("nan")
    positive_rank_sum = np.sum(ranks[:n_pos])
    return float(
        (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tier0_score_diagnostics.json"),
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.cache_root.glob("session*/sub-*_tier0_features.npz")):
        cache = load_cache(path)
        channels = tuple(str(item) for item in cache["channels"])
        bands = tuple(str(item) for item in cache["bands"])
        normalizer = LocationScale.fit(cache["pre_features"], axis=0)
        z = normalizer.transform(cache["continuous_features"])
        scores = tier0_from_features(
            cache["continuous_features"], channels, normalizer
        )
        times = cache["continuous_times"]
        command_mask = sample_mask(times, cache["command_intervals"].tolist())
        iti_mask = sample_mask(times, cache["iti_intervals"].tolist())
        fixation_mask = sample_mask(times, cache["fixation_intervals"].tolist())
        command = scores[command_mask]
        iti = scores[iti_mask]
        fixation = scores[fixation_mask]

        effects = {}
        for channel_index, channel in enumerate(channels):
            for band_index, band in enumerate(bands):
                # Positive value means lower power (stronger ERD) during command.
                effects[f"{channel}:{band}"] = float(
                    np.mean(z[iti_mask, channel_index, band_index])
                    - np.mean(z[command_mask, channel_index, band_index])
                )

        rows.append(
            {
                "subject": int(cache["subject"]),
                "session": int(cache["session"]),
                "command_vs_iti_auc": auc(command, iti),
                "command_vs_fixation_auc": auc(command, fixation),
                "command_score_mean": float(np.mean(command)),
                "iti_score_mean": float(np.mean(iti)),
                "fixation_score_mean": float(np.mean(fixation)),
                "command_minus_iti": float(np.mean(command) - np.mean(iti)),
                "channel_band_erd_effect": effects,
            }
        )

    aucs = np.asarray([row["command_vs_iti_auc"] for row in rows])
    effects_by_feature = {
        name: np.asarray(
            [row["channel_band_erd_effect"][name] for row in rows], dtype=float
        )
        for name in rows[0]["channel_band_erd_effect"]
    } if rows else {}
    summary = {
        "recordings": len(rows),
        "subjects": len({row["subject"] for row in rows}),
        "sessions": sorted({row["session"] for row in rows}),
        "command_vs_iti_auc": {
            "median": float(np.median(aucs)) if len(aucs) else None,
            "iqr": (
                [float(value) for value in np.quantile(aucs, [0.25, 0.75])]
                if len(aucs)
                else None
            ),
            "fraction_above_0_5": float(np.mean(aucs > 0.5)) if len(aucs) else None,
        },
        "channel_band_effect_median": {
            name: float(np.median(values))
            for name, values in effects_by_feature.items()
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

