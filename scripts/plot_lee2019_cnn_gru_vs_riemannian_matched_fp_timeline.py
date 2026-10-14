"""Time-domain matched-FP comparison of CNN-GRU and Riemannian Arm A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import evaluate_lee2019_loso_deployment as base
from scripts.evaluate_lee2019_cnn_gru_vs_riemannian_matched_fp import (
    DWELL_SECONDS,
    MA_SECONDS,
    REFRACTORY_SECONDS,
    SCREEN,
    SFREQ,
    STEP,
    fit_riemannian_scores,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma,
    ewma_z,
    ma_trace,
)
from uncued_mi.riemannian import tangent_features


def spans_from_mask(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [(float(times[g[0]]), float(times[g[-1]] + STEP)) for g in groups]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--stop", type=float, default=60.)
    parser.add_argument("--target-fps", type=float, nargs="+", default=[2., 4.])
    parser.add_argument(
        "--results", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_vs_riemannian_matched_fp.json"),
    )
    parser.add_argument(
        "--cnn-root", type=Path,
        default=Path("data/interim/lee2019_cnn_gru_loso_arm_a_stream_v2_score_cache"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    selected_rows = {
        float(row["target_fp_per_min"]): row
        for row in payload["rows"]
        if int(row["subject"]) == args.subject and int(row["session"]) == args.session
    }
    missing = set(args.target_fps) - set(selected_rows)
    if missing:
        raise ValueError(f"Missing target-FP rows: {sorted(missing)}")

    cache_path = args.cnn_root / f"session{args.session}" / f"sub-{args.subject:02d}.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        times = np.asarray(cache["times"], float)
        allowed = np.asarray(cache["allowed"], bool)
        events = np.asarray(cache["events"], float)
        cnn_raw = np.asarray(cache["raw_scores"], float)
        cnn_anchor = np.asarray(cache["anchor_scores"], float)

    domains = base.load_covariances(args.session)
    own_cache = {}
    for subject in base.SUBJECTS:
        own_ref = base.reference(domains[subject]["pre"])
        own_cache[subject] = {
            "command": tangent_features(domains[subject]["command"], own_ref),
            "idle": tangent_features(domains[subject]["idle"], own_ref),
        }
    anchor_cov, stream_cov, _, locked_times, _, locked_allowed = base.prepare_target(
        args.subject, args.session
    )
    if not (np.array_equal(times, locked_times) and np.array_equal(allowed, locked_allowed)):
        raise AssertionError("CNN cache does not match the locked Riemannian stream")
    ria_raw, ria_anchor = fit_riemannian_scores(
        args.subject, domains, own_cache, anchor_cov, stream_cov
    )

    def decision_trace(raw: np.ndarray, anchor: np.ndarray, threshold: float):
        z = ewma_z(raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed)
        ma = ma_trace(z, n=int(round(MA_SECONDS / STEP)))
        alarms = alarms_from_ma(
            ma, threshold, allowed,
            dwell=int(round(DWELL_SECONDS / STEP)),
            refractory=int(round(REFRACTORY_SECONDS / STEP)),
        )
        return ma - threshold, alarms

    eval_start = float(events[SCREEN, 2] / SFREQ)
    relative_times = times - eval_start
    evalmask = relative_times >= 0
    cues = events[:, 5] / SFREQ
    gt = np.c_[cues[SCREEN:] + .5 - eval_start, cues[SCREEN:] + 4. - eval_start]
    state = np.zeros(len(times), bool)
    for lo, hi in gt:
        state |= (relative_times >= lo) & (relative_times <= hi)
    idle = evalmask & allowed & ~state
    view = evalmask & (relative_times >= args.start) & (relative_times <= args.stop)

    models = (
        ("cnn_gru", "CNN-GRU", cnn_raw, cnn_anchor, "#2878b5"),
        ("riemannian", "Riemannian Arm A", ria_raw, ria_anchor, "#e36c21"),
    )
    fig, axes = plt.subplots(
        len(args.target_fps) * len(models), 1,
        figsize=(16, 3.05 * len(args.target_fps) * len(models)), sharex=True,
    )
    axes = np.atleast_1d(axes)
    panel = 0
    for target_fp in args.target_fps:
        row = selected_rows[target_fp]
        for model_key, model_label, raw, anchor, color in models:
            ax = axes[panel]
            metric = row[model_key]
            threshold = float(metric["threshold"])
            margin, alarms = decision_trace(raw, anchor, threshold)
            alarms &= evalmask

            first_span = True
            for lo, hi in gt:
                if hi >= args.start and lo <= args.stop:
                    ax.axvspan(
                        lo, hi, color="#e98b9a", alpha=.22,
                        label="true MI" if first_span else None,
                    )
                    first_span = False
            for lo, hi in spans_from_mask(relative_times, view & ~allowed):
                ax.axvspan(lo, hi, color="0.75", alpha=.16)

            ax.plot(
                relative_times[view], margin[view], color=color, lw=1.25,
                label="MA(z) - matched-FP threshold",
            )
            ax.axhline(0, color="#c82433", ls="--", lw=1.1, label="decision boundary")
            alarm_view = alarms & view
            ax.scatter(
                relative_times[alarm_view], margin[alarm_view], marker="v", s=38,
                color="black", zorder=5, label="alarm",
            )
            idle_exceedance = (margin >= 0) & idle & view
            ax.scatter(
                relative_times[idle_exceedance], margin[idle_exceedance], s=10,
                color="#e15759", alpha=.65, label="idle threshold exceedance",
            )
            ax.set_ylabel("margin")
            ax.set_title(
                f"target FP={target_fp:g}/min — {model_label}   "
                f"threshold={threshold:.2f}, matched actual FP={metric['actual_fp_per_min']:.2f}/min, "
                f"SEN@1s={metric['onset_sensitivity']['1.0']:.2f}, "
                f"excess@1s={metric['circular_shift_excess']['1.0']:+.2f}",
                loc="left", fontsize=11,
            )
            ax.grid(axis="y", alpha=.18)
            ax.legend(loc="upper right", ncol=5, fontsize=8)
            panel += 1

    axes[-1].set_xlabel("Time from evaluation-stream start (s)")
    axes[-1].set_xlim(args.start, args.stop)
    fig.suptitle(
        f"Matched-achieved-FP time-domain comparison — Lee2019 subject {args.subject:02d}, "
        f"session {args.session}\n"
        "identical locked stream; upper-tail diagnostic operating points",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, .955))
    output = args.output or ROOT / (
        "data/processed/"
        f"lee2019_cnn_gru_vs_riemannian_matched_fp_timeline_sub{args.subject:02d}_"
        f"s{args.session}_fp{'-'.join(f'{x:g}' for x in args.target_fps)}_"
        f"{args.start:g}-{args.stop:g}s.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
