"""Temporal and subject-concentration audit of the CNN-GRU idle-score tail."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS, HALF_LIFE_SECONDS, MA_SECONDS, REFRACTORY_SECONDS, STEP,
    normalize_anchor, tune_ma_dwell,
)
from scripts.run_lee2019_cnn_gru_oracle_alarm_controller import interval_mask
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration


def episodes(mask):
    mask = np.asarray(mask, bool)
    return int(np.sum(mask & ~np.r_[False, mask[:-1]]))


def analyze(path, subject, session):
    sequence, blocks = load_cache(path)
    mean, variance, normalized = normalize_anchor(blocks)
    anchor = np.concatenate(normalized)
    tail_threshold = float(np.quantile(anchor, .995))
    raw_anchor = np.concatenate(blocks).astype(float)
    raw_tail_threshold = float(np.quantile(raw_anchor, .995))
    operating = tune_ma_dwell([calibration(x) for x in normalized], 160)
    adapted = ewma_normalize_and_decide(
        sequence.scores, initial_mean=mean, initial_variance=variance,
        threshold=operating.threshold, step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS, update_mode="all",
        ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, sigma_floor_fraction=.25,
    )
    times, z = sequence.times, adapted.statistic
    idle = interval_mask(times, sequence.idle_intervals)
    command = interval_mask(times, sequence.command_intervals)
    tail = z >= tail_threshold
    raw_tail = sequence.scores >= raw_tail_threshold
    idle_tail = idle & tail
    idle_raw_tail = idle & raw_tail
    command_tail = command & tail
    elapsed = np.full(len(times), np.nan)
    remaining = np.full(len(times), np.nan)
    segment_counts = []
    for start, stop in sequence.idle_intervals:
        member = (times >= start) & (times < stop)
        elapsed[member] = times[member] - start
        remaining[member] = stop - times[member]
        segment_counts.append(int(np.sum(idle_tail & member)))
    duration = max(float(times[-1] - times[0]), STEP)
    tercile = np.minimum((3 * (times - times[0]) / duration).astype(int), 2)
    idle_windows_by_tercile = [int(np.sum(idle & (tercile == k))) for k in range(3)]
    tail_by_tercile = [int(np.sum(idle_tail & (tercile == k))) for k in range(3)]
    raw_tail_by_tercile = [int(np.sum(idle_raw_tail & (tercile == k))) for k in range(3)]
    bins = ((0., .5), (.5, 1.), (1., 2.), (2., np.inf))
    elapsed_counts = [int(np.sum(idle_tail & (elapsed >= lo) & (elapsed < hi)))
                      for lo, hi in bins]
    elapsed_exposure = [int(np.sum(idle & (elapsed >= lo) & (elapsed < hi)))
                        for lo, hi in bins]
    end_near = idle & (remaining <= .5 + 1e-9)
    return {
        "subject": subject, "session": session,
        "tail_threshold_anchor_q995": tail_threshold,
        "raw_tail_threshold_anchor_q995": raw_tail_threshold,
        "idle_windows": int(idle.sum()),
        "idle_tail_windows": int(idle_tail.sum()),
        "idle_tail_fraction": float(np.mean(tail[idle])),
        "idle_tail_episodes": episodes(idle_tail),
        "idle_raw_tail_windows": int(idle_raw_tail.sum()),
        "idle_raw_tail_fraction": float(np.mean(raw_tail[idle])),
        "idle_raw_tail_episodes": episodes(idle_raw_tail),
        "command_windows": int(command.sum()),
        "command_tail_fraction": float(np.mean(tail[command])),
        "idle_segments_with_tail": int(np.sum(np.asarray(segment_counts) > 0)),
        "idle_segments": len(segment_counts),
        "idle_tail_by_stream_tercile": tail_by_tercile,
        "idle_raw_tail_by_stream_tercile": raw_tail_by_tercile,
        "idle_windows_by_stream_tercile": idle_windows_by_tercile,
        "idle_tail_by_elapsed_bin": elapsed_counts,
        "idle_windows_by_elapsed_bin": elapsed_exposure,
        "tail_within_first_0p5s_of_idle": int(np.sum(idle_tail & (elapsed < .5))),
        "tail_within_last_0p5s_of_idle": int(np.sum(idle_tail & end_near)),
    }


def boot_median(values, seed):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (20000, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def summarize(rows, session):
    chosen = [row for row in rows if row["session"] == session]
    total_tail = sum(row["idle_tail_windows"] for row in chosen)
    total_idle = sum(row["idle_windows"] for row in chosen)
    sorted_counts = sorted((row["idle_tail_windows"] for row in chosen), reverse=True)
    tercile_tail = np.sum([row["idle_tail_by_stream_tercile"] for row in chosen], axis=0)
    tercile_raw_tail = np.sum([row["idle_raw_tail_by_stream_tercile"] for row in chosen], axis=0)
    tercile_idle = np.sum([row["idle_windows_by_stream_tercile"] for row in chosen], axis=0)
    elapsed_tail = np.sum([row["idle_tail_by_elapsed_bin"] for row in chosen], axis=0)
    elapsed_idle = np.sum([row["idle_windows_by_elapsed_bin"] for row in chosen], axis=0)
    fractions = [row["idle_tail_fraction"] for row in chosen]
    return {
        "session": session, "n": len(chosen),
        "pooled_idle_tail_fraction": float(total_tail / total_idle),
        "subject_idle_tail_fraction_median": float(np.median(fractions)),
        "subject_idle_tail_fraction_ci95": boot_median(fractions, 20261600 + session),
        "subjects_with_zero_idle_tail": int(sum(row["idle_tail_windows"] == 0 for row in chosen)),
        "share_of_tail_in_top_20pct_subjects": float(sum(sorted_counts[:4]) / max(total_tail, 1)),
        "stream_tercile_tail_rates": (tercile_tail / tercile_idle).tolist(),
        "stream_tercile_raw_tail_rates": (tercile_raw_tail / tercile_idle).tolist(),
        "pooled_idle_raw_tail_fraction": float(
            sum(row["idle_raw_tail_windows"] for row in chosen) / total_idle
        ),
        "pooled_idle_raw_tail_episodes": int(sum(row["idle_raw_tail_episodes"] for row in chosen)),
        "idle_elapsed_bins_seconds": [[0., .5], [.5, 1.], [1., 2.], [2., None]],
        "idle_elapsed_tail_rates": (elapsed_tail / np.maximum(elapsed_idle, 1)).tolist(),
        "pooled_idle_tail_episodes": int(sum(row["idle_tail_episodes"] for row in chosen)),
        "pooled_idle_tail_windows": int(total_tail),
        "pooled_idle_segments_with_tail": int(sum(row["idle_segments_with_tail"] for row in chosen)),
        "pooled_idle_segments": int(sum(row["idle_segments"] for row in chosen)),
        "pooled_command_tail_fraction": float(
            sum(row["command_tail_fraction"] * row["command_windows"] for row in chosen)
            / sum(row["command_windows"] for row in chosen)
        ),
    }


def plot(payload, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for session, color in ((1, "#1f77b4"), (2, "#d62728")):
        rows = [row for row in payload["rows"] if row["session"] == session]
        axes[0].hist([row["idle_tail_fraction"] for row in rows], bins=np.linspace(0, .25, 16),
                     alpha=.5, label=f"S{session}", color=color)
        summary = payload["summary"][session - 1]
        axes[1].plot((1, 2, 3), summary["stream_tercile_tail_rates"], marker="o",
                     label=f"S{session}", color=color)
        axes[1].plot((1, 2, 3), summary["stream_tercile_raw_tail_rates"], marker="x",
                     linestyle="--", label=f"S{session} raw", color=color, alpha=.7)
        axes[2].plot((.25, .75, 1.5, 2.5), summary["idle_elapsed_tail_rates"], marker="o",
                     label=f"S{session}", color=color)
    axes[0].set_title("Subject idle-tail fraction"); axes[0].set_xlabel("Tail-window fraction")
    axes[0].set_ylabel("Subjects")
    axes[1].set_title("Tail rate across stream"); axes[1].set_xlabel("Stream tercile")
    axes[1].set_ylabel("Idle tail-window rate"); axes[1].set_xticks((1, 2, 3))
    axes[2].set_title("Tail rate within idle segment"); axes[2].set_xlabel("Elapsed idle bin (s)")
    axes[2].set_ylabel("Idle tail-window rate"); axes[2].set_xticks((.25, .75, 1.5, 2.5), ("0-.5", ".5-1", "1-2", ">2"))
    for ax in axes:
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle("CNN-GRU EWMA rest-tail temporal audit")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path, default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_rest_tail_audit.json"))
    parser.add_argument("--figure", type=Path, default=Path("data/processed/lee2019_cnn_gru_rest_tail_audit.png"))
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz", subject, session))
    payload = {
        "protocol": {
            "tail": "EWMA z >= subject-session anchor-idle Q99.5",
            "score": "strict all-score EWMA H30",
            "warning": "random-injected idle; temporal audit is conditional on this stream construction",
        },
        "summary": [summarize(rows, session) for session in args.sessions],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(payload, args.figure)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
