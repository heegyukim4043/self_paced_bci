"""Compare CNN-GRU, GCN-GRU and Fourier-GKAN-GRU on identical pilot streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_cnn_gru_vs_riemannian_matched_fp import (
    MA_SECONDS, SCREEN, SFREQ, STEP, circular_shift_null, selected_metrics,
    threshold_curve,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z, ma_trace


MODELS = ("cnn_gru", "gcn_gru", "gkan_gru")
TARGETS = (2.0, 4.0)


def load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def bootstrap_median(values: np.ndarray, subjects: np.ndarray, seed: int,
                     repeats: int) -> list[float]:
    unique = np.unique(subjects)
    lookup = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    for index in range(repeats):
        chosen = rng.choice(unique, len(unique), replace=True)
        draw = np.concatenate([values[lookup[subject]] for subject in chosen])
        draws[index] = np.median(draw)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def matched_rows(roots: dict[str, Path], subjects: list[int], sessions: list[int],
                 thresholds: int, shifts: int) -> list[dict]:
    rows = []
    for session in sessions:
        for subject in subjects:
            caches = {
                model: load_cache(root / f"session{session}" / f"sub-{subject:02d}.npz")
                for model, root in roots.items()
            }
            reference = caches["cnn_gru"]
            for model, cache in caches.items():
                for key in ("times", "starts", "allowed", "events"):
                    if not np.array_equal(cache[key], reference[key]):
                        raise AssertionError(f"{model} {key} mismatch S{session} sub-{subject:02d}")
            times = reference["times"].astype(float)
            allowed = reference["allowed"].astype(bool)
            events = reference["events"].astype(float)
            evalmask = times >= events[SCREEN, 2] / SFREQ
            cues = events[:, 5] / SFREQ
            onsets = cues[SCREEN:] + 0.5
            gt = np.c_[onsets, cues[SCREEN:] + 4.0]
            state = np.zeros(len(times), bool)
            for start, stop in gt:
                state |= (times >= start) & (times <= stop)
            idle = evalmask & allowed & ~state
            exposure = float(idle.sum() * STEP)
            traces = {}
            curves = {}
            for model, cache in caches.items():
                raw = cache["raw_scores"].astype(float)
                anchor = cache["anchor_scores"].astype(float)
                z = ewma_z(raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed)
                ma = ma_trace(z, n=int(round(MA_SECONDS / STEP)))
                traces[model] = ma
                curves[model] = threshold_curve(
                    ma, times, allowed, evalmask, idle, exposure, onsets, thresholds,
                )
            shared = sorted(set.intersection(*(set(curve) for curve in curves.values())))
            if not shared:
                raise RuntimeError(f"No common FP count S{session} sub-{subject:02d}")
            for target in TARGETS:
                desired = target * exposure / 60.0
                count = min(shared, key=lambda value: (abs(value - desired), value))
                offsets = np.random.default_rng(
                    20260915 + 1000 * session + 10 * subject + int(target)
                ).integers(1, len(times), size=shifts)
                metrics = {}
                for model in MODELS:
                    point = curves[model][count]
                    metrics[model] = selected_metrics(
                        traces[model], point["threshold"], times, allowed, evalmask,
                        idle, exposure, onsets, gt, offsets,
                    )
                rows.append({
                    "subject": subject, "session": session, "target_fp": target,
                    "idle_exposure_seconds": exposure,
                    "matched_false_alarm_count": int(count),
                    "matched_actual_fp_per_min": float(count / (exposure / 60.0)),
                    "models": metrics,
                })
            print(f"matched three-model S{session} sub-{subject:02d}", flush=True)
    return rows


def value(row: dict, model: str, metric: str) -> float:
    item = row["models"][model]
    if metric == "sensitivity1":
        return float(item["onset_sensitivity"]["1.0"])
    if metric == "null1":
        return float(item["circular_shift_null_median"]["1.0"])
    if metric == "excess1":
        return float(item["circular_shift_excess"]["1.0"])
    return float(item[metric])


def summarize_matched(rows: list[dict], repeats: int) -> dict:
    output = {}
    comparisons = (
        ("gcn_minus_cnn", "gcn_gru", "cnn_gru"),
        ("gkan_minus_cnn", "gkan_gru", "cnn_gru"),
        ("gkan_minus_gcn", "gkan_gru", "gcn_gru"),
    )
    for target in TARGETS:
        selected = [row for row in rows if row["target_fp"] == target]
        subjects = np.asarray([row["subject"] for row in selected])
        block = {"models": {}, "paired": {}}
        for model in MODELS:
            block["models"][model] = {
                metric: float(np.median([value(row, model, metric) for row in selected]))
                for metric in ("sensitivity1", "null1", "excess1", "coverage", "temporal_iou", "strict_f1")
            }
        for index, (name, first, second) in enumerate(comparisons):
            block["paired"][name] = {}
            for metric_index, metric in enumerate(("excess1", "coverage", "temporal_iou")):
                delta = np.asarray([
                    value(row, first, metric) - value(row, second, metric)
                    for row in selected
                ])
                block["paired"][name][metric] = {
                    "median_delta": float(np.median(delta)),
                    "ci95": bootstrap_median(
                        delta, subjects,
                        202609150 + int(target) * 1000 + index * 20 + metric_index,
                        repeats,
                    ),
                    "session_median_delta": {
                        str(session): float(np.median(delta[np.asarray([
                            row["session"] == session for row in selected
                        ])])) for session in (1, 2)
                    },
                }
        output[str(target)] = block
    return output


def indexed_rows(path: Path) -> dict[tuple[int, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(int(row["subject"]), int(row["session"])): row for row in payload["rows"]}


def summarize_prospective(paths: dict[str, Path], subjects: list[int],
                          sessions: list[int]) -> dict:
    tables = {model: indexed_rows(path) for model, path in paths.items()}
    output = {}
    for target in TARGETS:
        output[str(target)] = {}
        for operating in ("source_transferred", "common_fp_target_pre_rest"):
            output[str(target)][operating] = {}
            for model in MODELS:
                points = [
                    tables[model][(subject, session)]["metrics"][str(target)][operating]["metrics"]
                    for session in sessions for subject in subjects
                ]
                fp = np.asarray([point["actual_fp_per_min"] for point in points])
                if target == 2.0:
                    admissible = (fp >= 1.5) & (fp <= 2.5)
                else:
                    admissible = (fp >= 3.0) & (fp <= 5.0)
                output[str(target)][operating][model] = {
                    "actual_fp_per_min_median": float(np.median(fp)),
                    "actual_fp_per_min_range": [float(np.min(fp)), float(np.max(fp))],
                    "admissible_n": int(np.sum(admissible)),
                    "admissible_rate": float(np.mean(admissible)),
                    "sensitivity_1s_median": float(np.median([
                        point["onset_sensitivity"]["1.0"] for point in points
                    ])),
                    "excess_1s_median": float(np.median([
                        point["circular_shift_excess"]["1.0"] for point in points
                    ])),
                    "coverage_median": float(np.median([
                        point["interval_0.1"]["coverage"] for point in points
                    ])),
                }
    return output


def plot(payload: dict, output: Path) -> None:
    colors = ("#4c72b0", "#55a868", "#c44e52")
    labels = ("CNN-GRU", "GCN-GRU", "GKAN-GRU")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, target in enumerate(TARGETS):
        matched = payload["matched_summary"][str(target)]["models"]
        ax = axes[0, column]
        excess = [matched[model]["excess1"] for model in MODELS]
        coverage = [matched[model]["coverage"] for model in MODELS]
        x = np.arange(3)
        ax.bar(x - 0.18, excess, 0.36, label="Excess@1s", color=colors)
        ax.bar(x + 0.18, coverage, 0.36, label="Coverage", color=colors, alpha=0.45)
        ax.set_xticks(x, labels)
        ax.set_title(f"Matched achieved FP={target:g}/min")
        ax.grid(axis="y", alpha=0.2)
        if column == 0:
            ax.legend()

        prospective = payload["prospective_summary"][str(target)]["source_transferred"]
        ax = axes[1, column]
        fp = [prospective[model]["actual_fp_per_min_median"] for model in MODELS]
        exc = [prospective[model]["excess_1s_median"] for model in MODELS]
        ax.bar(x - 0.18, fp, 0.36, label="Actual FP/min", color=colors)
        ax2 = ax.twinx()
        ax2.plot(x + 0.18, exc, "ko-", label="Excess@1s")
        ax.axhline(target, color="black", linestyle="--", lw=1)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Actual FP/min")
        ax2.set_ylabel("Excess@1s")
        ax.set_title("Source-transferred operating point")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Lee2019 spatial graph pilot: CNN vs GCN vs Fourier-GKAN", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[3, 10, 20, 25, 54])
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--thresholds", type=int, default=301)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/lee2019_cnn_graph_gru_pilot5_comparison.json"))
    parser.add_argument("--figure", type=Path, default=Path(
        "data/processed/lee2019_cnn_graph_gru_pilot5_comparison.png"))
    args = parser.parse_args()
    roots = {
        "cnn_gru": Path("data/interim/lee2019_cnn_gru_loso_arm_a_stream_v2_score_cache"),
        "gcn_gru": Path("data/interim/lee2019_cnn_gcn_gru_scores_pilot5"),
        "gkan_gru": Path("data/interim/lee2019_cnn_gkan_gru_scores_pilot5"),
    }
    prospective_paths = {
        "cnn_gru": Path("data/processed/lee2019_cnn_gru_loso_arm_a_stream_v2_full54.json"),
        "gcn_gru": Path("data/processed/lee2019_cnn_gcn_gru_loso_pilot5.json"),
        "gkan_gru": Path("data/processed/lee2019_cnn_gkan_gru_loso_pilot5.json"),
    }
    rows = matched_rows(
        roots, args.subjects, args.sessions, args.thresholds, args.shifts,
    )
    payload = {
        "protocol": {
            "cohort": "pre-specified architecture pilot subjects 3,10,20,25,54 x sessions 1,2",
            "models": "same 3-band 20-channel input and GRU; spatial block CNN versus physical-prior GCN versus Fourier-GKAN",
            "parameters": {"cnn_gru": 6453, "gcn_gru": 6619, "gkan_gru": 6775},
            "matched_fp": "three models at identical evaluation-idle false-alarm count; diagnostic oracle only",
            "prospective": "source-transferred and target-pre-rest thresholds; no evaluation labels",
            "decision": "EWMA H30 + MA .5 s + dwell .3 s + refractory 1 s",
            "claim_limit": "pilot only; no promotion without full 54-subject replication",
        },
        "matched_summary": summarize_matched(rows, args.bootstrap),
        "prospective_summary": summarize_prospective(
            prospective_paths, args.subjects, args.sessions,
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(payload, args.figure)
    print(json.dumps({
        "matched_summary": payload["matched_summary"],
        "prospective_summary": payload["prospective_summary"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
