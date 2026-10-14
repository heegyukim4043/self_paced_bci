"""Per-channel cue/rest ERPs with paired max-T permutation significance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def t_stat(values):
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return np.divide(mean, sem, out=np.zeros_like(mean), where=sem > 0)


def contiguous_ranges(mask, times):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, splits)
    return [
        [float(times[group[0]] * 1000), float(times[group[-1]] * 1000)]
        for group in groups
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cue-cache", type=Path,
        default=Path("data/processed/lee2019_cue_erp_subjects.npz"),
    )
    parser.add_argument(
        "--rest-cache", type=Path,
        default=Path("data/processed/lee2019_rest_pseudoevent_erp_subjects.npz"),
    )
    parser.add_argument("--permutations", type=int, default=4096)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cue_vs_rest_by_channel.png"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("data/processed/lee2019_cue_vs_rest_by_channel.json"),
    )
    args = parser.parse_args()
    cue_cache = np.load(args.cue_cache, allow_pickle=False)
    rest_cache = np.load(args.rest_cache, allow_pickle=False)
    cue = cue_cache["subject_erps"]
    rest = rest_cache["subject_erps"]
    times = cue_cache["times"]
    channels = cue_cache["channels"].tolist()
    if cue.shape != rest.shape:
        raise ValueError("Cue/rest subject ERP shapes differ")

    display = (times >= -0.2) & (times <= 0.8)
    test = (times >= 0) & (times <= 0.6)
    test_times = times[test]
    difference = (cue - rest)[:, :, test]
    observed_t = t_stat(difference)

    # Paired sign-flip max-T: family-wise error control across every tested
    # channel and time point, with no spatial-cluster assumption.
    flattened = difference.reshape(len(difference), -1).astype(np.float64)
    sumsq = np.sum(flattened ** 2, axis=0)
    rng = np.random.default_rng(20260730)
    maxima = []
    remaining = args.permutations
    while remaining:
        count = min(128, remaining)
        signs = rng.choice((-1.0, 1.0), size=(count, len(difference)))
        sums = signs @ flattened
        means = sums / len(difference)
        variances = (
            sumsq[None] - len(difference) * means ** 2
        ) / (len(difference) - 1)
        sem = np.sqrt(np.maximum(variances, 1e-20) / len(difference))
        maxima.extend(np.max(np.abs(means / sem), axis=1))
        remaining -= count
    threshold = float(np.quantile(maxima, 0.95))
    significant = np.abs(observed_t) > threshold

    cue_mean, rest_mean = cue.mean(axis=0), rest.mean(axis=0)
    cue_sem = cue.std(axis=0, ddof=1) / np.sqrt(len(cue))
    rest_sem = rest.std(axis=0, ddof=1) / np.sqrt(len(rest))
    limit = float(np.quantile(np.abs(np.r_[cue_mean[:, display], rest_mean[:, display]]), 0.995))
    limit = max(limit, 1.0)

    fig, axes = plt.subplots(8, 8, figsize=(20, 20), sharex=True, sharey=True)
    axes = axes.ravel()
    results = {}
    for channel_index, (axis, channel) in enumerate(zip(axes, channels)):
        axis.plot(times[display], cue_mean[channel_index, display],
                  color="#2166ac", lw=1.2)
        axis.plot(times[display], rest_mean[channel_index, display],
                  color="#666666", lw=1.0)
        axis.fill_between(
            times[display],
            (cue_mean[channel_index] - cue_sem[channel_index])[display],
            (cue_mean[channel_index] + cue_sem[channel_index])[display],
            color="#2166ac", alpha=0.12, linewidth=0,
        )
        axis.fill_between(
            times[display],
            (rest_mean[channel_index] - rest_sem[channel_index])[display],
            (rest_mean[channel_index] + rest_sem[channel_index])[display],
            color="#777777", alpha=0.10, linewidth=0,
        )
        axis.axvline(0, color="black", ls="--", lw=0.6)
        axis.axhline(0, color="0.7", lw=0.5)
        ranges = contiguous_ranges(significant[channel_index], test_times)
        for start_ms, stop_ms in ranges:
            axis.plot(
                [start_ms / 1000, stop_ms / 1000],
                [-0.91 * limit, -0.91 * limit],
                color="#b2182b", lw=3, solid_capstyle="butt",
            )
        axis.set_title(channel, fontsize=9, color=(
            "#b2182b" if ranges else "black"
        ))
        axis.set_xlim(-0.2, 0.8)
        axis.set_ylim(-limit, limit)
        axis.tick_params(labelsize=7, length=2)
        results[channel] = {
            "fwer_significant": bool(ranges),
            "significant_ranges_ms": ranges,
            "peak_abs_t": float(np.max(np.abs(observed_t[channel_index]))),
        }
    for axis in axes[len(channels):]:
        axis.axis("off")
    fig.supxlabel("Time from visual cue onset (s)")
    fig.supylabel("Amplitude (µV)")
    fig.suptitle(
        "Lee2019 cue vs resting ERP by channel\n"
        "blue=cue, gray=rest, red bottom bar=max-T FWER p<0.05",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "subjects": int(len(cue)),
        "permutations": args.permutations,
        "test_interval_ms": [0, 600],
        "method": "paired sign-flip max-T across all 62 channels and time points",
        "fwer_alpha": 0.05,
        "max_t_threshold": threshold,
        "significant_channel_count": int(sum(
            item["fwer_significant"] for item in results.values()
        )),
        "channels": results,
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        key: value for key, value in payload.items() if key != "channels"
    }, indent=2))


if __name__ == "__main__":
    main()
