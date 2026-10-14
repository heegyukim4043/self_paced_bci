"""Evaluate the break-fixed aligned Tier-0 gate on Lee2019 artifact recordings."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.config import ExperimentConfig
from uncued_mi.features import decision_subsample, extract_log_power
from uncued_mi.operating_point import ScoredSequence, apply_decision


def text(value: object) -> str:
    array = np.asarray(value)
    while array.size == 1 and array.dtype.kind == "O":
        array = np.asarray(array.item())
    return str(array.squeeze())


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_operating_points(path: Path) -> dict[int, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    selected = {}
    for row in rows:
        if (
            row["decision"] == "ma_dwell"
            and row["calibration"] == "idle_feature_alignment"
            and float(row["target_fp_setting"]) == 2.0
        ):
            selected[int(row["subject"])] = row
    if len(selected) != 54:
        raise ValueError(f"Expected 54 aligned FP=2 rows in {path}, got {len(selected)}")
    return selected


def interval_count(alarms: np.ndarray, start: float, stop: float) -> int:
    return int(np.sum((alarms >= start) & (alarms < stop)))


def analyze_one(
    artifact_path: Path,
    cache_path: Path,
    operating_row: dict,
    session: int,
    subject: int,
    config: ExperimentConfig,
) -> list[dict]:
    artifact = loadmat(artifact_path)["EEG_Artifact"][0, 0]
    eeg = np.asarray(artifact["x"], dtype=np.float64)
    sfreq = float(np.asarray(artifact["fs"]).squeeze())
    channels = tuple(text(item) for item in np.asarray(artifact["chan"]).ravel())
    starts = np.asarray(artifact["t"]).squeeze().astype(float)
    labels = [text(item) for item in np.asarray(artifact["y_class"]).ravel()]

    cache = load_cache(cache_path)
    output_channels = tuple(str(item) for item in cache["channels"])
    features = extract_log_power(
        eeg,
        sfreq,
        channels,
        config,
        output_channels=output_channels,
    )
    normalizer = LocationScale.fit(cache["pre_features"], axis=0)
    scores = tier0_from_features(features.log_power, output_channels, normalizer)
    times, scores = decision_subsample(features.times, scores, 0.1)
    sequence = ScoredSequence(
        times=times,
        scores=scores,
        idle_intervals=np.empty((0, 2)),
        command_intervals=np.empty((0, 2)),
    )
    alarm_mask = apply_decision(
        sequence,
        "ma_dwell",
        float(operating_row["operating_point"]["threshold"]),
        0.1,
        ma_seconds=0.5,
        dwell_seconds=0.3,
        refractory_seconds=1.0,
    ).alarms
    alarm_times = times[alarm_mask]

    rows = []
    for start_sample, label in zip(starts, labels):
        start = (start_sample - 1.0) / sfreq
        stop = start + 10.0
        count = interval_count(alarm_times, start, stop)
        rows.append(
            {
                "subject": subject,
                "session": session,
                "artifact": label,
                "alarms": count,
                "seconds": 10.0,
                "fp_per_minute": count * 6.0,
                "clean_post_alarms": int(
                    operating_row["post_rest"]["false_alarms"]
                ),
                "clean_post_seconds": float(
                    operating_row["post_rest"]["idle_seconds"]
                ),
            }
        )
    return rows


def pooled_rate(rows: list[dict], count_key: str, seconds_key: str) -> float:
    seconds = sum(float(row[seconds_key]) for row in rows)
    return 60.0 * sum(int(row[count_key]) for row in rows) / seconds


def cluster_bootstrap_ratio(
    rows: list[dict], artifact: str, repeats: int = 10_000
) -> tuple[float, list[float]]:
    artifact_rows = [row for row in rows if row["artifact"] == artifact]
    by_subject: dict[int, list[dict]] = defaultdict(list)
    for row in artifact_rows:
        by_subject[int(row["subject"])].append(row)
    subjects = np.asarray(sorted(by_subject))

    def ratio(sampled: np.ndarray) -> float:
        sampled_rows = [
            row for subject in sampled for row in by_subject[int(subject)]
        ]
        artifact_rate = pooled_rate(sampled_rows, "alarms", "seconds")
        clean_rate = pooled_rate(
            sampled_rows, "clean_post_alarms", "clean_post_seconds"
        )
        return artifact_rate / clean_rate if clean_rate > 0 else np.nan

    point = ratio(subjects)
    rng = np.random.default_rng(20260725)
    values = np.asarray(
        [ratio(rng.choice(subjects, size=len(subjects), replace=True)) for _ in range(repeats)]
    )
    values = values[np.isfinite(values)]
    return point, [float(value) for value in np.quantile(values, [0.025, 0.975])]


def summarize(rows: list[dict]) -> list[dict]:
    clean_rate = pooled_rate(rows, "clean_post_alarms", "clean_post_seconds")
    summaries = []
    for artifact in sorted({row["artifact"] for row in rows}):
        subset = [row for row in rows if row["artifact"] == artifact]
        artifact_rate = pooled_rate(subset, "alarms", "seconds")
        ratio, ratio_ci = cluster_bootstrap_ratio(rows, artifact)
        summaries.append(
            {
                "artifact": artifact,
                "subject_sessions": len(subset),
                "pooled_fp_per_minute": artifact_rate,
                "median_subject_session_fp_per_minute": float(
                    np.median([row["fp_per_minute"] for row in subset])
                ),
                "clean_post_pooled_fp_per_minute": clean_rate,
                "rate_ratio_vs_clean_post": ratio,
                "rate_ratio_cluster_bootstrap_95_ci": ratio_ci,
            }
        )
    return summaries


def plot_summary(summary: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["artifact"] for row in summary]
    rates = [row["pooled_fp_per_minute"] for row in summary]
    clean = summary[0]["clean_post_pooled_fp_per_minute"]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(np.arange(len(labels)), rates, color="#3B82F6")
    axis.axhline(clean, color="#111827", linestyle="--", label=f"clean post-rest: {clean:.2f}")
    axis.set_ylabel("False positives / minute")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    axis.set_title("Lee2019 artifact stress: aligned Tier-0 gate at source FP=2")
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("data/raw/lee2019_artifact")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed")
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    all_rows: list[dict] = []
    for session in (1, 2):
        operating_points = load_operating_points(
            args.processed_root / f"tier0_breakfixed_session{session}.json"
        )
        for subject in range(1, 55):
            artifact_path = (
                args.artifact_root
                / f"session{session}"
                / f"s{subject}"
                / f"sess{session:02d}_subj{subject:02d}_EEG_Artifact.mat"
            )
            cache_path = (
                args.cache_root
                / f"session{session}"
                / f"sub-{subject:02d}_tier0_features.npz"
            )
            all_rows.extend(
                analyze_one(
                    artifact_path,
                    cache_path,
                    operating_points[subject],
                    session,
                    subject,
                    config,
                )
            )
            print(f"processed session={session} subject={subject}", flush=True)

    result = {
        "protocol": {
            "gate": "Tier-0 idle-z bandpower",
            "calibration": "current-session pre-rest mean/std alignment",
            "decision": "MA 0.5 s + dwell 0.3 s + refractory 1.0 s",
            "operating_point": "break-fixed LOSO source FP=2",
            "artifact_exposure_seconds_per_type": 10.0,
            "clean_reference": "held-out post-train rest",
        },
        "summary": summarize(all_rows),
        "rows": all_rows,
    }
    output = args.processed_root / "lee2019_artifact_stress.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    plot_summary(result["summary"], args.processed_root / "lee2019_artifact_stress.png")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
