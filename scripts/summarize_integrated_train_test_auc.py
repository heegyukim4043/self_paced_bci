from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

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
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_integrated/session1"))
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/integrated_train_test_auc_subset20.json"))
    args = parser.parse_args()
    rows = []
    for subject in args.subjects:
        path = args.cache_root / f"sub-{subject:02d}_integrated.npz"
        with np.load(path, allow_pickle=False) as data:
            channels = tuple(str(x) for x in data["bandpower_channels"])
            for window in ("0p5", "1p0", "2p0", "3p0"):
                row = {"subject": subject, "window_seconds": float(window.replace("p", "."))}
                for phase in ("train", "test"):
                    times = data[f"{phase}_w{window}_times"]
                    model = LocationScale.fit(data[f"{phase}_w{window}_pre_bandpower"], axis=0)
                    scores = tier0_from_features(data[f"{phase}_w{window}_bandpower"], channels, model)
                    command = sample_mask(times, data[f"{phase}_command_intervals"].tolist())
                    idle = sample_mask(times, data[f"{phase}_idle_iti_intervals"].tolist())
                    row[f"{phase}_auc"] = auc(scores[command], scores[idle])
                row["test_minus_train_auc"] = row["test_auc"] - row["train_auc"]
                rows.append(row)
    summary = []
    for window in (0.5, 1.0, 2.0, 3.0):
        selected = [x for x in rows if x["window_seconds"] == window]
        delta = np.asarray([x["test_minus_train_auc"] for x in selected])
        summary.append({
            "window_seconds": window,
            "subjects": len(selected),
            "train_auc_median": float(np.median([x["train_auc"] for x in selected])),
            "test_auc_median": float(np.median([x["test_auc"] for x in selected])),
            "paired_delta_median": float(np.median(delta)),
            "wilcoxon_p": float(wilcoxon(delta).pvalue),
        })
    result = {"summary": summary, "rows": rows}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
