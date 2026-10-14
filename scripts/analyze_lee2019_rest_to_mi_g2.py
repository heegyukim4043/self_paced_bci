"""Lee2019 G2: causal high-beta medial-to-sensorimotor timing.

This analysis estimates *cue-locked scalp high-beta state-change timing*.
It must not be interpreted as self-initiated intention or SMA activation.

The analysis specification is fixed before inspecting G2 outcomes:

* primary band: 20--30 Hz;
* common-average reference;
* fourth-order causal Butterworth bandpass applied to the continuous record;
* squared amplitude followed by a 150-ms causal exponential smoother;
* broad medial frontocentral ROI: Fz/FC1/FC2/Cz;
* strict medial sensitivity ROI: Fz/Cz;
* label 1/right -> C3 contralateral; label 2/left -> C4 contralateral;
* continuous segmented-regression breakpoint searched over 0.3--2.0 s.

S1 is reported as discovery and S2 as frozen replication.  The script does
not adapt any parameter after seeing S1 results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, sosfilt
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.data import lee2019_path, load_lee2019_run


PRIMARY_BAND = (20.0, 30.0)
SELECTED_CHANNELS = ("Fz", "FC1", "FC2", "Cz", "C3", "C4")
MEDIAL_BROAD = ("Fz", "FC1", "FC2", "Cz")
MEDIAL_STRICT = ("Fz", "Cz")
CURVE_START = -1.0
CURVE_STOP = 3.0
CURVE_STEP = 0.02
FIT_RANGE = (-0.5, 2.5)
CHANGE_SEARCH = (0.3, 2.0)
SMOOTHER_TAU_SECONDS = 0.150
MIN_PERSISTENCE_SECONDS = 0.200
QUALITY_DELTA_BIC = 6.0
MAPPING_VERSION = "mat_class_field_v2_label1_right_label2_left"


def car_selected(eeg: np.ndarray, selected: list[int]) -> np.ndarray:
    """Apply CAR over all electrodes, returning only requested channels."""
    common = np.mean(eeg, axis=1, dtype=np.float64)
    return np.asarray(eeg[:, selected], np.float64) - common[:, None]


def causal_power(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Causal high-beta squared amplitude with causal exponential smoothing."""
    sos = butter(4, PRIMARY_BAND, btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfilt(sos, signal, axis=0)
    squared = filtered * filtered
    alpha = 1.0 - np.exp(-1.0 / (sfreq * SMOOTHER_TAU_SECONDS))
    # The preceding 60-s pre-rest warms both filters before the task record.
    return lfilter([alpha], [1.0, -(1.0 - alpha)], squared, axis=0)


def piecewise_fit(times: np.ndarray, values: np.ndarray) -> dict:
    """Estimate a continuous, downward ERD changepoint on a fixed grid.

    Both an unconstrained breakpoint and an ERD-directed breakpoint are kept.
    The directed estimate requires the slope change and post-change slope to be
    negative.  Delta-BIC is reported as quality evidence rather than silently
    selecting subjects on fit quality.
    """
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    fit = (
        (times >= FIT_RANGE[0]) & (times <= FIT_RANGE[1])
        & np.isfinite(values)
    )
    t = times[fit]
    y = values[fit]
    if len(y) < 10:
        return {"tau": None, "tau_unconstrained": None, "delta_bic": None}

    base_x = np.column_stack([np.ones_like(t), t])
    base_beta, *_ = np.linalg.lstsq(base_x, y, rcond=None)
    base_residual = y - base_x @ base_beta
    base_sse = max(float(base_residual @ base_residual), 1e-20)
    base_bic = len(y) * np.log(base_sse / len(y)) + 2 * np.log(len(y))

    candidates = times[
        (times >= CHANGE_SEARCH[0]) & (times <= CHANGE_SEARCH[1])
    ]
    records = []
    for tau in candidates:
        x = np.column_stack([np.ones_like(t), t, np.maximum(0.0, t - tau)])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        sse = max(float(residual @ residual), 1e-20)
        pre_slope = float(beta[1])
        slope_change = float(beta[2])
        post_slope = pre_slope + slope_change
        records.append((sse, float(tau), pre_slope, post_slope, slope_change, beta))

    best_all = min(records, key=lambda item: item[0])
    directed = [item for item in records if item[3] < 0.0 and item[4] < 0.0]
    if not directed:
        return {
            "tau": None,
            "tau_unconstrained": best_all[1],
            "delta_bic": None,
            "pre_slope_db_per_s": None,
            "post_slope_db_per_s": None,
            "slope_change_db_per_s": None,
            "quality_pass": False,
        }

    best = min(directed, key=lambda item: item[0])
    piece_bic = len(y) * np.log(best[0] / len(y)) + 3 * np.log(len(y))
    delta_bic = float(base_bic - piece_bic)
    return {
        "tau": best[1],
        "tau_unconstrained": best_all[1],
        "delta_bic": delta_bic,
        "pre_slope_db_per_s": best[2],
        "post_slope_db_per_s": best[3],
        "slope_change_db_per_s": best[4],
        "quality_pass": bool(delta_bic >= QUALITY_DELTA_BIC),
    }


def first_half_amplitude_time(times: np.ndarray, values: np.ndarray) -> float | None:
    """First time the curve reaches half of its early 0.5--2.5-s ERD depth."""
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    baseline_mask = (times >= -0.5) & (times < 0.0)
    response_mask = (times >= 0.5) & (times <= 2.5)
    if not np.any(baseline_mask) or not np.any(response_mask):
        return None
    baseline = float(np.median(values[baseline_mask]))
    minimum = float(np.min(values[response_mask]))
    depth = minimum - baseline
    if not np.isfinite(depth) or depth >= -0.05:
        return None
    target = baseline + 0.5 * depth
    indices = np.flatnonzero((times >= CHANGE_SEARCH[0]) & (values <= target))
    return float(times[indices[0]]) if len(indices) else None


def longest_erd_interval(
    times: np.ndarray, values: np.ndarray, onset: float | None
) -> float | None:
    """Longest contiguous below-baseline interval after the estimated onset."""
    if onset is None:
        return None
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    baseline = float(np.median(values[(times >= -0.5) & (times < 0.0)]))
    active = (times >= onset) & (times <= CURVE_STOP) & (values < baseline)
    longest = current = 0
    for flag in active:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    duration = longest * CURVE_STEP
    return float(duration if duration >= MIN_PERSISTENCE_SECONDS else 0.0)


def endpoint(times: np.ndarray, curve: np.ndarray) -> dict:
    change = piecewise_fit(times, curve)
    change["t50"] = first_half_amplitude_time(times, curve)
    change["persistence_s"] = longest_erd_interval(times, curve, change.get("tau"))
    return change


def analyze_one(
    data_root: Path, subject: int, session: int,
    power_estimator=causal_power,
) -> dict:
    run = load_lee2019_run(lee2019_path(data_root, subject, session), "train")
    lookup = {name.casefold(): index for index, name in enumerate(run.channels)}
    selected = [lookup[name.casefold()] for name in SELECTED_CHANNELS]
    local = {name: index for index, name in enumerate(SELECTED_CHANNELS)}
    sfreq = float(run.sfreq)

    # Pre-rest is used only as causal filter history; normalization remains the
    # trial-local -1..0-s baseline, as fixed in G1.
    rest_signal = car_selected(run.pre_rest, selected)
    task_signal = car_selected(run.eeg, selected)
    warm_samples = len(rest_signal)
    joined = np.vstack([rest_signal, task_signal])
    power = power_estimator(joined, sfreq)[warm_samples:]
    del joined, rest_signal, task_signal

    offsets = np.arange(
        int(round(CURVE_START * sfreq)),
        int(round(CURVE_STOP * sfreq)),
        max(1, int(round(CURVE_STEP * sfreq))),
    )
    times = offsets / sfreq
    broad_indices = [local[name] for name in MEDIAL_BROAD]
    strict_indices = [local[name] for name in MEDIAL_STRICT]
    c3_index, c4_index = local["C3"], local["C4"]
    trial_curves = {key: [] for key in (
        "medial_broad", "medial_strict", "contra", "ipsi", "laterality"
    )}

    for cue, label in zip(np.asarray(run.cue_samples, int), np.asarray(run.labels, int)):
        baseline = np.mean(power[cue - int(sfreq):cue], axis=0)
        sampled = power[cue + offsets]
        channel_db = 10.0 * np.log10(
            np.maximum(sampled, 1e-20) / np.maximum(baseline, 1e-20)
        )
        # Power is combined across channels, avoiding voltage cancellation.
        for name, indices in (
            ("medial_broad", broad_indices), ("medial_strict", strict_indices)
        ):
            roi_sample = np.mean(sampled[:, indices], axis=1)
            roi_base = float(np.mean(baseline[indices]))
            trial_curves[name].append(
                10.0 * np.log10(np.maximum(roi_sample, 1e-20) / max(roi_base, 1e-20))
            )
        contra_index = c3_index if label == 1 else c4_index
        ipsi_index = c4_index if label == 1 else c3_index
        contra = channel_db[:, contra_index]
        ipsi = channel_db[:, ipsi_index]
        trial_curves["contra"].append(contra)
        trial_curves["ipsi"].append(ipsi)
        trial_curves["laterality"].append(contra - ipsi)

    curves = {key: np.median(np.asarray(value), axis=0) for key, value in trial_curves.items()}
    endpoints = {key: endpoint(times, curve) for key, curve in curves.items()}
    return {
        "subject": subject,
        "session": session,
        "n_trials": len(run.labels),
        "times": times.tolist(),
        "curves_db": {key: value.tolist() for key, value in curves.items()},
        "endpoints": endpoints,
    }


def finite_values(values: list[float | None]) -> np.ndarray:
    return np.asarray([value for value in values if value is not None and np.isfinite(value)], float)


def bootstrap_median(values: np.ndarray, seed: int, repeats: int) -> list[float] | None:
    values = finite_values(values.tolist())
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(values), size=(repeats, len(values)))
    draws = np.median(values[chosen], axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def positive_summary(values: list[float | None], seed: int, repeats: int) -> dict:
    array = finite_values(values)
    if len(array) == 0:
        return {"n": 0, "median_s": None, "bootstrap_95ci": None, "wilcoxon_greater_p": None}
    try:
        p_value = float(wilcoxon(array, zero_method="pratt", alternative="greater").pvalue)
    except ValueError:
        p_value = 1.0
    return {
        "n": int(len(array)),
        "median_s": float(np.median(array)),
        "bootstrap_95ci": bootstrap_median(array, seed, repeats),
        "wilcoxon_greater_p": p_value,
        "negative_zero_positive": [
            int(np.sum(array < 0)), int(np.sum(array == 0)), int(np.sum(array > 0))
        ],
    }


def scalar_summary(values: list[float | None], seed: int, repeats: int) -> dict:
    array = finite_values(values)
    return {
        "n": int(len(array)),
        "median": float(np.median(array)) if len(array) else None,
        "bootstrap_95ci": bootstrap_median(array, seed, repeats) if len(array) else None,
        "iqr": np.quantile(array, [0.25, 0.75]).tolist() if len(array) else None,
    }


def paired_contrast(rows: list[dict], first: str, second: str, field: str = "tau") -> list[float]:
    values = []
    for row in rows:
        a = row["endpoints"][first].get(field)
        b = row["endpoints"][second].get(field)
        if a is not None and b is not None:
            values.append(float(a - b))
    return values


def icc3_1(first: np.ndarray, second: np.ndarray) -> float | None:
    """Two-way mixed, consistency, single-measure ICC(3,1)."""
    x = np.column_stack([first, second]).astype(float)
    n, k = x.shape
    if n < 3:
        return None
    row_means = np.mean(x, axis=1)
    grand = float(np.mean(x))
    ms_rows = k * float(np.sum((row_means - grand) ** 2)) / (n - 1)
    residual = x - row_means[:, None] - np.mean(x, axis=0)[None, :] + grand
    ms_error = float(np.sum(residual ** 2)) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error
    return float((ms_rows - ms_error) / denominator) if denominator > 0 else None


def reproducibility(
    rows: list[dict], contrast: tuple[str, str], field: str = "tau"
) -> dict:
    by_key = {(row["subject"], row["session"]): row for row in rows}
    first, second = [], []
    for subject in sorted({row["subject"] for row in rows}):
        pair = []
        for session in (1, 2):
            row = by_key.get((subject, session))
            if row is None:
                pair = []
                break
            a = row["endpoints"][contrast[0]].get(field)
            b = row["endpoints"][contrast[1]].get(field)
            if a is None or b is None:
                pair = []
                break
            pair.append(float(a - b))
        if len(pair) == 2:
            first.append(pair[0])
            second.append(pair[1])
    first_array, second_array = np.asarray(first), np.asarray(second)
    if len(first_array) >= 3:
        rho = spearmanr(first_array, second_array)
        rho_value, p_value = float(rho.statistic), float(rho.pvalue)
    else:
        rho_value = p_value = None
    return {
        "n": len(first),
        "spearman_rho": rho_value,
        "spearman_p": p_value,
        "icc_3_1": icc3_1(first_array, second_array),
    }


def make_summary(rows: list[dict], repeats: int) -> dict:
    output: dict[str, dict] = {}
    for session in (1, 2):
        selected = [row for row in rows if row["session"] == session]
        endpoints = {}
        for index, roi in enumerate(("medial_broad", "medial_strict", "contra", "ipsi", "laterality")):
            endpoints[roi] = {
                "tau_s": scalar_summary(
                    [row["endpoints"][roi].get("tau") for row in selected],
                    10000 * session + 100 * index, repeats,
                ),
                "t50_s": scalar_summary(
                    [row["endpoints"][roi].get("t50") for row in selected],
                    10000 * session + 100 * index + 1, repeats,
                ),
                "persistence_s": scalar_summary(
                    [row["endpoints"][roi].get("persistence_s") for row in selected],
                    10000 * session + 100 * index + 2, repeats,
                ),
                "quality_pass_n": int(sum(
                    bool(row["endpoints"][roi].get("quality_pass")) for row in selected
                )),
            }
        contrasts = {
            "contra_minus_medial_broad_tau_s": positive_summary(
                paired_contrast(selected, "contra", "medial_broad"),
                20000 * session + 1, repeats,
            ),
            "contra_minus_medial_strict_tau_s": positive_summary(
                paired_contrast(selected, "contra", "medial_strict"),
                20000 * session + 2, repeats,
            ),
            "laterality_minus_medial_broad_tau_s": positive_summary(
                paired_contrast(selected, "laterality", "medial_broad"),
                20000 * session + 3, repeats,
            ),
            "contra_minus_medial_broad_t50_s": positive_summary(
                paired_contrast(selected, "contra", "medial_broad", "t50"),
                20000 * session + 4, repeats,
            ),
            # Exact plan-specified segmented model without ERD-direction
            # constraints. These forced estimates are retained as an audit;
            # they must be read with boundary concentration and direction QC.
            "unconstrained_contra_minus_medial_broad_tau_s": positive_summary(
                paired_contrast(
                    selected, "contra", "medial_broad", "tau_unconstrained"
                ),
                20000 * session + 6, repeats,
            ),
            "unconstrained_contra_minus_medial_strict_tau_s": positive_summary(
                paired_contrast(
                    selected, "contra", "medial_strict", "tau_unconstrained"
                ),
                20000 * session + 7, repeats,
            ),
            "unconstrained_laterality_minus_medial_broad_tau_s": positive_summary(
                paired_contrast(
                    selected, "laterality", "medial_broad", "tau_unconstrained"
                ),
                20000 * session + 8, repeats,
            ),
        }
        # Quality-qualified sensitivity analysis, never substituted for primary.
        quality_rows = [row for row in selected if (
            row["endpoints"]["contra"].get("quality_pass")
            and row["endpoints"]["medial_broad"].get("quality_pass")
        )]
        contrasts["quality_subset_contra_minus_medial_broad_tau_s"] = positive_summary(
            paired_contrast(quality_rows, "contra", "medial_broad"),
            20000 * session + 5, repeats,
        )
        output[str(session)] = {"n_sessions": len(selected), "endpoints": endpoints, "contrasts": contrasts}

    output["replication"] = {
        "contra_minus_medial_broad": reproducibility(rows, ("contra", "medial_broad")),
        "laterality_minus_medial_broad": reproducibility(rows, ("laterality", "medial_broad")),
        "unconstrained_contra_minus_medial_broad": reproducibility(
            rows, ("contra", "medial_broad"), "tau_unconstrained"
        ),
        "unconstrained_laterality_minus_medial_broad": reproducibility(
            rows, ("laterality", "medial_broad"), "tau_unconstrained"
        ),
    }
    return output


def plot_results(
    rows: list[dict], output: Path,
    estimator_label: str = "causal Butterworth + power EMA",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        "medial_broad": "#6a3d9a", "contra": "#1f78b4",
        "ipsi": "#e66101", "laterality": "#2ca25f",
    }
    labels = {
        "medial_broad": "medial Fz/FC1/FC2/Cz", "contra": "contralateral C3/C4",
        "ipsi": "ipsilateral C4/C3", "laterality": "contra - ipsi",
    }
    for panel, session in enumerate((1, 2)):
        ax = axes[0, panel]
        selected = [row for row in rows if row["session"] == session]
        if not selected:
            ax.set_axis_off()
            ax.set_title(f"S{session}: not present in this run")
            continue
        times = np.asarray(selected[0]["times"])
        for roi in ("medial_broad", "contra", "ipsi", "laterality"):
            curves = np.stack([row["curves_db"][roi] for row in selected])
            median = np.median(curves, axis=0)
            lo, hi = np.quantile(curves, [0.25, 0.75], axis=0)
            ax.plot(times, median, color=colors[roi], lw=1.8, label=labels[roi])
            ax.fill_between(times, lo, hi, color=colors[roi], alpha=0.10)
        ax.axvspan(0, 0.3, color="0.75", alpha=0.25, label="visual exclusion")
        ax.axhline(0, color="0.25", lw=0.8)
        ax.axvline(0, color="0.25", ls=":", lw=0.8)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(f"S{session}: {'discovery' if session == 1 else 'frozen replication'}")
        ax.set_xlabel("Time from visual cue (s)")
        ax.set_ylabel("Causal high-beta change (dB)")
        ax.grid(alpha=0.2)
        quality = {
            roi: sum(bool(row["endpoints"][roi].get("quality_pass")) for row in selected)
            for roi in ("medial_broad", "contra", "laterality")
        }
        ax.text(
            0.98, 0.03,
            "directed changepoint ΔBIC≥6: "
            f"medial {quality['medial_broad']}/54, "
            f"contra {quality['contra']}/54, lat {quality['laterality']}/54",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        if panel == 0:
            ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 0]
    positions, data, tick_labels = [], [], []
    position = 1
    for session in (1, 2):
        selected = [row for row in rows if row["session"] == session]
        for first, second, short in (
            ("contra", "medial_broad", "contra-medial"),
            ("laterality", "medial_broad", "laterality-medial"),
        ):
            values = finite_values(
                paired_contrast(selected, first, second, "tau_unconstrained")
            )
            if len(values):
                positions.append(position)
                data.append(values)
                tick_labels.append(f"S{session}\n{short}")
            position += 1
        position += 0.4
    if data:
        parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.8)
        for body in parts["bodies"]:
            body.set_facecolor("#80b1d3")
            body.set_alpha(0.6)
    ax.axhline(0, color="0.2", lw=1)
    ax.set_xticks(positions, tick_labels, fontsize=8)
    ax.set_ylabel("Onset difference (s); positive = medial earlier")
    ax.set_title("Forced one-breakpoint audit (all sessions)")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    by_key = {(row["subject"], row["session"]): row for row in rows}
    x, y = [], []
    for subject in sorted({row["subject"] for row in rows}):
        if (subject, 1) not in by_key or (subject, 2) not in by_key:
            continue
        values = []
        for session in (1, 2):
            row = by_key[(subject, session)]
            contra = row["endpoints"]["contra"].get("tau_unconstrained")
            medial = row["endpoints"]["medial_broad"].get("tau_unconstrained")
            values.append(None if contra is None or medial is None else contra - medial)
        if all(value is not None for value in values):
            x.append(values[0])
            y.append(values[1])
    ax.scatter(x, y, s=30, alpha=0.7, color="#4d4d4d")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(0, color="0.4", lw=0.8)
    lo = min([-1.7, *x, *y]) if x else -1.7
    hi = max([1.7, *x, *y]) if x else 1.7
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.6", lw=1)
    ax.set_xlabel("S1 contra - medial forced breakpoint (s)")
    ax.set_ylabel("S2 contra - medial forced breakpoint (s)")
    ax.set_title("Within-subject forced-breakpoint replication")
    ax.grid(alpha=0.2)

    fig.suptitle(
        f"Lee2019 G2: cue-locked high-beta timing ({estimator_label})\n"
        "54 subjects; S1 discovery and parameter-frozen S2 replication",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("data/processed/lee2019_rest_to_mi_g2_checkpoint.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_rest_to_mi_g2.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("data/processed/lee2019_rest_to_mi_g2.png"),
    )
    args = parser.parse_args()

    specification = {
        "mapping_version": MAPPING_VERSION,
        "primary_band_hz": list(PRIMARY_BAND),
        "reference": "CAR across all 62 channels before ROI selection",
        "causal_filter": "4th-order Butterworth bandpass, continuous with 60-s pre-rest warm-up",
        "causal_power_smoother_tau_s": SMOOTHER_TAU_SECONDS,
        "trial_baseline_s": [-1.0, 0.0],
        "curve_s": [CURVE_START, CURVE_STOP, CURVE_STEP],
        "fit_range_s": list(FIT_RANGE),
        "changepoint_search_s": list(CHANGE_SEARCH),
        "medial_broad": list(MEDIAL_BROAD),
        "medial_strict_sensitivity": list(MEDIAL_STRICT),
        "minimum_persistence_s": MIN_PERSISTENCE_SECONDS,
        "quality_delta_bic": QUALITY_DELTA_BIC,
        "primary_contrast": "tau_contralateral - tau_medial_broad; positive means medial earlier",
        "control_contrast": "tau_(contra-minus-ipsi) - tau_medial_broad",
        "discovery_replication": "parameters fixed a priori; S1 discovery, unchanged S2 replication",
        "claim_limit": "cue-locked scalp high-beta state change; not SMA, intention, or propagation",
    }

    rows: list[dict] = []
    if args.checkpoint.exists():
        cached = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if cached.get("specification") != specification:
            raise RuntimeError("Checkpoint specification differs; use a new checkpoint path")
        rows = cached.get("rows", [])
    complete = {(int(row["subject"]), int(row["session"])) for row in rows}
    for session in args.sessions:
        for index, subject in enumerate(args.subjects, start=1):
            if (subject, session) in complete:
                print(f"G2 S{session} sub-{subject:02d} cached", flush=True)
                continue
            row = analyze_one(args.data_root, subject, session)
            rows.append(row)
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            args.checkpoint.write_text(json.dumps({
                "specification": specification, "rows": rows,
            }), encoding="utf-8")
            print(f"G2 S{session} sub-{subject:02d} [{index}/{len(args.subjects)}]", flush=True)

    expected = {(subject, session) for subject in args.subjects for session in args.sessions}
    rows = [row for row in rows if (row["subject"], row["session"]) in expected]
    payload = {
        "protocol": specification,
        "summary": make_summary(rows, args.bootstrap),
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(rows, args.figure)
    print(args.output)
    print(args.figure)


if __name__ == "__main__":
    main()
