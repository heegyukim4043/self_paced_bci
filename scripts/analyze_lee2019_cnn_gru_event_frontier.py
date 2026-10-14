"""P1: FP-vs-observed/null/excess frontier for frozen CNN-GRU scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.decision import instantaneous
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import (
    SUBSET20,
    load_cache,
    tune_empirical,
)
from scripts.run_lee2019_stateful_gru import onset_rates

STEP = 0.1
FP_GRID = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0)


def metric_at(sequence: ScoredSequence, threshold: float, offsets: np.ndarray):
    alarms = instantaneous(sequence.scores, threshold, STEP, 1.0).alarms
    conventional = evaluate_alarms(
        sequence.times, alarms, sequence.idle_intervals, sequence.command_intervals
    )
    observed, delay = onset_rates(sequence.times, alarms, sequence.command_intervals)
    # Vectorized exact equivalent of np.roll(alarms, offset) followed by the
    # 1-s onset hit test. Shape is shifts x score samples (about 500 x 2,000).
    indices = (
        np.arange(len(alarms), dtype=np.int64)[None, :]
        - np.asarray(offsets, dtype=np.int64)[:, None]
    ) % len(alarms)
    shifted = alarms[indices]
    hit_counts = np.zeros(len(offsets), dtype=float)
    for start, _ in sequence.command_intervals:
        left = int(np.searchsorted(sequence.times, start, side="left"))
        right = int(np.searchsorted(sequence.times, start + 1.0, side="right"))
        hit_counts += np.any(shifted[:, left:right], axis=1)
    null_value = float(np.median(hit_counts / len(sequence.command_intervals)))
    return {
        "threshold": float(threshold),
        "fp": float(conventional.fp_per_minute),
        "observed": float(observed["1.0"]),
        "null": null_value,
        "excess": float(observed["1.0"] - null_value),
        "delay": None if not np.isfinite(delay) else float(delay),
    }


def curve_for(sequence, offsets, grid_size):
    finite = sequence.scores[np.isfinite(sequence.scores)]
    probabilities = np.linspace(0.40, 0.9998, grid_size)
    thresholds = np.unique(np.quantile(finite, probabilities))
    rows = [metric_at(sequence, value, offsets) for value in thresholds]
    rows.sort(key=lambda item: item["fp"])
    return rows


def nearest(curve, target):
    return min(curve, key=lambda item: (abs(item["fp"] - target), -item["observed"]))


def bootstrap(values, seed, repeats):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()


def summarize(rows, session, repeats):
    selected = [row for row in rows if row["session"] == session]
    output = {"session": session, "subjects": len(selected), "frontier": []}
    for target in FP_GRID:
        points = [row["common_fp"][str(target)] for row in selected]
        item = {"target_fp": target}
        for key in ("fp", "observed", "null", "excess"):
            values = [point[key] for point in points]
            item[f"{key}_median"] = float(np.median(values))
            item[f"{key}_ci"] = bootstrap(
                values, 20260908 + session * 100 + int(target * 10), repeats
            )
        delays = [point["delay"] for point in points if point["delay"] is not None]
        item["delay_median"] = float(np.median(delays)) if delays else None
        output["frontier"].append(item)
    transferred = [row["transferred"] for row in selected]
    output["transferred"] = {
        "fp_median": float(np.median([x["fp"] for x in transferred])),
        "observed_median": float(np.median([x["observed"] for x in transferred])),
        "null_median": float(np.median([x["null"] for x in transferred])),
        "excess_median": float(np.median([x["excess"] for x in transferred])),
    }
    output["paired_increment_excess"] = []
    for lower, upper in zip(FP_GRID[:-1], FP_GRID[1:]):
        differences = np.asarray([
            row["common_fp"][str(upper)]["excess"]
            - row["common_fp"][str(lower)]["excess"]
            for row in selected
        ])
        output["paired_increment_excess"].append({
            "from_fp": lower,
            "to_fp": upper,
            "median_delta": float(np.median(differences)),
            "ci": bootstrap(
                differences,
                20261008 + session * 100 + int(upper * 10),
                repeats,
            ),
            "subjects_positive": int(np.sum(differences > 0)),
        })
    # Threshold-free diagnostic area over the fixed actual-FP domain.
    x = np.asarray(FP_GRID, dtype=float)
    y = np.asarray([item["excess_median"] for item in output["frontier"]])
    output["excess_pauc_0p5_10"] = float(np.trapezoid(y, x) / (x[-1] - x[0]))
    peak_index = int(np.argmax(y))
    output["median_frontier_peak"] = {
        "target_fp": float(x[peak_index]),
        "excess": float(y[peak_index]),
    }
    return output


def plot(summary, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, item in zip(axes, summary):
        x = np.asarray([p["fp_median"] for p in item["frontier"]])
        for key, label, color in (
            ("observed", "Observed SEN@1s", "#1f77b4"),
            ("null", "Circular-shift null", "#999999"),
            ("excess", "Excess SEN", "#d62728"),
        ):
            y = np.asarray([p[f"{key}_median"] for p in item["frontier"]])
            lo = np.asarray([p[f"{key}_ci"][0] for p in item["frontier"]])
            hi = np.asarray([p[f"{key}_ci"][1] for p in item["frontier"]])
            ax.plot(x, y, marker="o", label=label, color=color)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12)
        t = item["transferred"]
        ax.scatter([t["fp_median"]], [t["excess_median"]], marker="X", s=90,
                   color="black", label="Transferred threshold")
        ax.axvline(2.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Session {item['session']} (n={item['subjects']})")
        ax.set_xlabel("Actual false positives / min")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Sensitivity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Frozen CNN-GRU event-information frontier", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path, default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--thresholds", type=int, default=120)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_event_frontier.json"))
    parser.add_argument("--figure", type=Path, default=Path("data/processed/lee2019_cnn_gru_event_frontier.png"))
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Reuse rows already present in --output and only rebuild summary/figure.",
    )
    args = parser.parse_args()
    if args.reuse_output:
        rows = json.loads(args.output.read_text(encoding="utf-8"))["rows"]
    else:
        rows = []
    for session in (() if args.reuse_output else args.sessions):
        for subject in args.subjects:
            sequence, calibrations = load_cache(
                args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz"
            )
            rng = np.random.default_rng(88000 + 100 * session + subject)
            offsets = rng.integers(1, len(sequence.scores), size=args.shifts)
            curve = curve_for(sequence, offsets, args.thresholds)
            threshold = tune_empirical(calibrations)
            rows.append({
                "subject": subject,
                "session": session,
                "transferred": metric_at(sequence, threshold, offsets),
                "common_fp": {str(target): nearest(curve, target) for target in FP_GRID},
                "curve": curve,
            })
            print(f"P1 session={session} subject={subject}", flush=True)
    summary = [summarize(rows, session, args.bootstrap) for session in args.sessions]
    payload = {
        "protocol": {
            "score": "frozen CNN-GRU 3-seed mean",
            "onset_horizon_seconds": 1.0,
            "fp_grid": FP_GRID,
            "circular_shifts": args.shifts,
            "thresholds_per_curve": args.thresholds,
            "common_fp_is_diagnostic": True,
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(summary, args.figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
