"""Four-panel descriptive I6 CNN-GRU timeline on original continuous Lee EEG.

This is a cue-paced, exploratory display, not an uncued MI performance figure.
The score, calibration threshold, and decision rule are the frozen I6 stateful
T-all FP2 condition. The 200-s view is a crop; panel metrics cover all future
trials 70--99. No EEG or thresholds are fitted here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from scripts.run_e4_original_continuous_fp_20261009 import (
    DWELL,
    REFRACTORY,
    _trial_strata,
    _union,
)
from scripts.run_i6_cnn_gru_original_continuous_20261012 import prepare_cache
from uncued_mi.decision import guard_rearmed_alarms

OUT = ROOT / "data/processed/i6_cnn_gru_original_continuous_fp2_four_panel_20261014.png"
RESULT = ROOT / "data/processed/i6_cnn_gru_original_continuous_20261012.json"
CACHE = ROOT / "data/interim/lee2019_i6_cnn_gru_original_cuda"
TIER0 = ROOT / "data/interim/lee2019_tier0"
CASES = ((3, 1), (3, 2), (54, 1), (54, 2))


def one_case(subject: int, session: int, row: dict,
             view_seconds: float | None = 200.0) -> dict:
    path = CACHE / f"session{session}" / f"sub-{subject:02d}.npz"
    tier_path = TIER0 / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz"
    times, cues, fixation, iti, ma = prepare_cache(
        path, tier_path, subject, session, "stateful"
    )
    with np.load(path, allow_pickle=False) as d:
        raw = np.asarray(d["raw_scores"], float)
        pre_raw = np.asarray(d["pre_raw_scores"], float)
        pre_times = np.asarray(d["pre_times"], float)
    anchor = (pre_times >= 4.0 - .001) & (pre_times < 60.0 - .001)
    z = ewma_z(
        raw,
        float(np.mean(pre_raw[anchor])),
        float(np.var(pre_raw[anchor], ddof=1)),
        np.ones(len(raw), dtype=bool),
    )
    threshold = float(row["calibration"]["threshold"])
    alarm = guard_rearmed_alarms(
        ma, threshold, np.ones(len(ma), dtype=bool),
        dwell_steps=DWELL, refractory_steps=REFRACTORY,
    )
    strata = _trial_strata(times, cues, fixation, iti, list(range(70, 100)))
    idle = _union(strata, False)
    if int(np.sum(alarm & idle)) != int(row["future_idle"]["count"]):
        raise AssertionError("plotted alarms do not reproduce I6 future-idle count")
    start = float(times[np.flatnonzero(idle)[0]])
    future_end = float(cues[99] + 4.0)
    stop = future_end if view_seconds is None else min(future_end, start + view_seconds)
    view = (times >= start) & (times <= stop)
    if not np.any(view):
        raise AssertionError("empty timeline crop")
    command = np.zeros(len(times), dtype=bool)
    for cue in cues[70:100]:
        command |= (times >= cue + .5) & (times < cue + 3.5)
    return {
        "times": times[view] - start,
        "z_margin": z[view] - threshold,
        "ma_margin": ma[view] - threshold,
        "alarm": alarm[view],
        "idle": idle[view],
        "fixation": strata["fixation"][view],
        "iti_early": strata["iti_early"][view],
        "iti_late": strata["iti_late"][view],
        "command": command[view],
        "cues": cues[70:100] - start,
        "future_duration_s": future_end - start,
        "threshold": threshold,
        "row": row,
    }


def _shade_mask(ax, times: np.ndarray, mask: np.ndarray, color: str, alpha: float) -> None:
    """Draw contiguous original-grid strata without interpolating gaps."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for i, j in zip(starts, stops):
        right = float(times[j]) if j < len(times) else float(times[-1] + .1)
        ax.axvspan(float(times[i]), right, color=color, alpha=alpha, lw=0, zorder=0)


def main(cases: tuple[tuple[int, int], ...] = CASES, out: Path = OUT,
         *, view_seconds: float | None = 200.0,
         show_strata: bool = False, one_row: bool = False,
         four_rows: bool = False) -> None:
    if len(cases) != 4:
        raise ValueError("four subject-session cases are required")
    if one_row and four_rows:
        raise ValueError("choose only one panel layout")
    if out.exists():
        raise FileExistsError(out)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = {
        (int(r["subject"]), int(r["session"])): r
        for r in result["arms"]["stateful"]["rows"]
        if r["arm"] == "T_all" and r["target_fp"] == 2.0
    }
    traces = [one_case(s, sess, rows[(s, sess)], view_seconds) for s, sess in cases]
    lo = min(float(np.min(np.r_[t["z_margin"], t["ma_margin"]])) for t in traces)
    hi = max(float(np.max(np.r_[t["z_margin"], t["ma_margin"]])) for t in traces)
    pad = .07 * (hi - lo)
    if four_rows:
        fig, axes = plt.subplots(4, 1, figsize=(21, 14.5), sharex=True, sharey=True)
        fig.subplots_adjust(left=.055, right=.99, bottom=.105, top=.91, hspace=.33)
    elif one_row:
        fig, axes = plt.subplots(1, 4, figsize=(42, 6.7), sharex=True, sharey=True)
        fig.subplots_adjust(left=.04, right=.992, bottom=.19, top=.78, wspace=.075)
    elif show_strata:
        fig, axes = plt.subplots(2, 2, figsize=(15, 8.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=.07, right=.985, bottom=.16, top=.93, hspace=.24, wspace=.08)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 7.6), layout="constrained", sharex=True, sharey=True)
    for ax, (subject, session), t in zip(axes.flat, cases, traces):
        if show_strata:
            _shade_mask(ax, t["times"], t["fixation"], "#7aa6d8", .16)
            _shade_mask(ax, t["times"], t["iti_early"], "#e7bd6b", .19)
            _shade_mask(ax, t["times"], t["iti_late"], "#79bc9c", .16)
        for cue in t["cues"]:
            left, right = cue + .5, cue + 3.5
            if right > 0 and left < t["times"][-1]:
                ax.axvspan(max(0, left), min(float(t["times"][-1]), right), color="#d88c9a", alpha=.30,
                           lw=0, zorder=0)
        ax.plot(t["times"], t["z_margin"], color="#3977be", lw=.85,
                alpha=.85, label="EWMA z − threshold", zorder=2)
        ax.plot(t["times"], t["ma_margin"], color="#df5a46", lw=1.05,
                label="MA(0.5 s) − threshold", zorder=3)
        ax.axhline(0, color="#252525", lw=1.0, zorder=1)
        mi_alarm = t["alarm"] & t["command"]
        idle_alarm = t["alarm"] & t["idle"]
        other_alarm = t["alarm"] & ~(t["command"] | t["idle"])
        ax.scatter(t["times"][mi_alarm], np.zeros(np.sum(mi_alarm)), marker="v",
                   color="black", s=28, zorder=5, label="alarm in cue-defined MI")
        ax.scatter(t["times"][idle_alarm], np.zeros(np.sum(idle_alarm)), marker="x",
                   color="#9b2f29", s=28, zorder=6, label="idle false alarm")
        ax.scatter(t["times"][other_alarm], np.zeros(np.sum(other_alarm)), marker="o",
                   facecolors="none", edgecolors="#555555", s=25, zorder=6,
                   label="other alarm")
        row = t["row"]
        if one_row:
            panel_title = (
                f"sub-{subject:02d} S{session} | FP2 threshold={t['threshold']:.2f}\n"
                f"future FP={row['future_idle']['fp_per_min']:.2f}/min | "
                f"SEN@2s={row['mi']['sen_2s']:.2f}"
            )
        else:
            panel_title = (
                f"sub-{subject:02d} S{session} | T-all FP2 threshold={t['threshold']:.2f} "
                f"| full future FP={row['future_idle']['fp_per_min']:.2f}/min, "
                f"SEN@2s={row['mi']['sen_2s']:.2f}"
            )
        ax.set_title(panel_title, loc="left", fontsize=15 if one_row else 12 if four_rows else 10)
        if one_row:
            ax.tick_params(axis="both", labelsize=12)
        elif four_rows:
            ax.tick_params(axis="both", labelsize=10)
        ax.set_xlim(0, max(float(x["times"][-1]) for x in traces))
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(axis="y", color="#888888", alpha=.17, lw=.6)
    for ax in (axes if four_rows else [axes[0]] if one_row else axes[:, 0]):
        ax.set_ylabel("Score margin (z units)", fontsize=14 if one_row else 11 if four_rows else None)
    for ax in ([axes[-1]] if four_rows else axes if one_row else axes[1, :]):
        ax.set_xlabel("Time from future-evaluation start (s)", fontsize=14 if one_row else 11 if four_rows else None)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if show_strata:
        handles += [
            Patch(facecolor="#d88c9a", alpha=.45),
            Patch(facecolor="#7aa6d8", alpha=.45),
            Patch(facecolor="#e7bd6b", alpha=.50),
            Patch(facecolor="#79bc9c", alpha=.45),
        ]
        labels += ["cue-defined MI", "fixation", "ITI early", "ITI late"]
    fig.legend(handles, labels, loc="lower center", ncol=9 if one_row else 5,
               fontsize=13 if one_row else 10 if four_rows else 8.5,
               bbox_to_anchor=(.5, .045 if one_row else .01 if four_rows else .005),
               frameon=False)
    scope = "full future trials 70–99" if view_seconds is None else f"first {view_seconds:g} s of future"
    if four_rows:
        fig.suptitle(
            f"Frozen CNN-GRU | Lee2019 {scope} | 0 = T-all FP2 threshold",
            fontsize=15, y=.975,
        )
    elif one_row:
        fig.suptitle(
            f"Frozen CNN-GRU | Lee2019 {scope} | 0 = T-all FP2 threshold",
            fontsize=18, y=.98,
        )
    elif not show_strata:
        fig.suptitle(
            f"Frozen CNN-GRU | Lee2019 {scope} | 0 = calibrated FP2 threshold",
            fontsize=12.5,
        )
    # Reserve room for the shared legend and title; constrained layout alone
    # does not account reliably for a figure-level legend across MPL versions.
    if not show_strata and not one_row and not four_rows:
        fig.get_layout_engine().set(rect=(0, .075, 1, .91))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
