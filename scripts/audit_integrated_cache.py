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


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.r_[positive, negative]
    ranks = rankdata(values, method="average")
    n = len(positive)
    return float((np.sum(ranks[:n]) - n * (n + 1) / 2) / (n * len(negative)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    channels = tuple(str(x) for x in data["bandpower_channels"])
    rows = []
    for phase in ("train", "test"):
        for window in ("0p5", "1p0", "2p0", "3p0"):
            times = data[f"{phase}_w{window}_times"]
            features = data[f"{phase}_w{window}_bandpower"]
            model = LocationScale.fit(data[f"{phase}_w{window}_pre_bandpower"], axis=0)
            scores = tier0_from_features(features, channels, model)
            command = sample_mask(times, data[f"{phase}_command_intervals"].tolist())
            idle = sample_mask(times, data[f"{phase}_idle_iti_intervals"].tolist())
            cov = data[f"{phase}_w{window}_covariance"]
            sample_index = np.linspace(0, len(cov) - 1, min(100, len(cov)), dtype=int)
            min_eigenvalue = float(
                np.min([np.linalg.eigvalsh(cov[index])[0] for index in sample_index])
            )
            rows.append({
                "phase": phase,
                "window_seconds": float(window.replace("p", ".")),
                "command_windows": int(np.count_nonzero(command)),
                "idle_windows": int(np.count_nonzero(idle)),
                "tier0_window_auc": auc(scores[command], scores[idle]),
                "covariance_shape": list(cov.shape),
                "sampled_min_eigenvalue": min_eigenvalue,
                "pre_rest_covariance_windows": len(data[f"{phase}_w{window}_pre_covariance"]),
            })
    result = {
        "cache": str(args.cache),
        "bandpower_channels": channels,
        "covariance_channels": [str(x) for x in data["covariance_channels"]],
        "rows": rows,
    }
    output = args.output or args.cache.with_suffix(".audit.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
