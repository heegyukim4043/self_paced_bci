"""Channel-wise paired comparison of post-cue and immediately pre-cue EEG."""

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


def ranges(mask, times):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
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
    parser.add_argument("--permutations", type=int, default=4096)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_postcue_vs_precue_by_channel.png"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("data/processed/lee2019_postcue_vs_precue_by_channel.json"),
    )
    args = parser.parse_args()
    cache = np.load(args.cue_cache, allow_pickle=False)
    erp = cache["subject_erps"]
    times = cache["times"]
    channels = cache["channels"].tolist()

    pre_mask = (times >= -0.5) & (times < 0)
    post_mask = (times >= 0) & (times < 0.5)
    pre = erp[:, :, pre_mask]
    post = erp[:, :, post_mask]
    count = min(pre.shape[-1], post.shape[-1])
    pre, post = pre[..., -count:], post[..., :count]
    relative_times = np.arange(count) / 250.0
    difference = post - pre
    observed_t = t_stat(difference)

    flattened = difference.reshape(len(difference), -1).astype(np.float64)
    sumsq = np.sum(flattened ** 2, axis=0)
    rng = np.random.default_rng(20260731)
    maxima, remaining = [], args.permutations
    while remaining:
        size = min(128, remaining)
        signs = rng.choice((-1.0, 1.0), size=(size, len(difference)))
        sums = signs @ flattened
        means = sums / len(difference)
        variances = (
            sumsq[None] - len(difference) * means ** 2
        ) / (len(difference) - 1)
        sem = np.sqrt(np.maximum(variances, 1e-20) / len(difference))
        maxima.extend(np.max(np.abs(means / sem), axis=1))
        remaining -= size
    threshold = float(np.quantile(maxima, 0.95))
    significant = np.abs(observed_t) > threshold

    pre_mean, post_mean = pre.mean(axis=0), post.mean(axis=0)
    pre_sem = pre.std(axis=0, ddof=1) / np.sqrt(len(pre))
    post_sem = post.std(axis=0, ddof=1) / np.sqrt(len(post))
    limit = float(np.quantile(np.abs(np.r_[pre_mean, post_mean]), 0.995))
    limit = max(limit, 1.0)

    fig, axes = plt.subplots(8, 8, figsize=(20, 20), sharex=True, sharey=True)
    axes = axes.ravel()
    result = {}
    for index, (axis, channel) in enumerate(zip(axes, channels)):
        axis.plot(relative_times, post_mean[index], color="#2166ac", lw=1.2)
        axis.plot(relative_times, pre_mean[index], color="#666666", lw=1.0)
        axis.fill_between(
            relative_times, post_mean[index]-post_sem[index],
            post_mean[index]+post_sem[index], color="#2166ac",
            alpha=0.12, linewidth=0,
        )
        axis.fill_between(
            relative_times, pre_mean[index]-pre_sem[index],
            pre_mean[index]+pre_sem[index], color="#777777",
            alpha=0.10, linewidth=0,
        )
        axis.axhline(0, color="0.7", lw=0.5)
        channel_ranges = ranges(significant[index], relative_times)
        for start, stop in channel_ranges:
            axis.plot(
                [start / 1000, stop / 1000], [-0.91*limit, -0.91*limit],
                color="#b2182b", lw=3, solid_capstyle="butt",
            )
        axis.set_title(
            channel, fontsize=9,
            color="#b2182b" if channel_ranges else "black",
        )
        axis.set(xlim=(0, 0.5), ylim=(-limit, limit))
        axis.tick_params(labelsize=7, length=2)
        result[channel] = {
            "fwer_significant": bool(channel_ranges),
            "significant_ranges_ms_after_alignment": channel_ranges,
            "peak_abs_t": float(np.max(np.abs(observed_t[index]))),
        }
    for axis in axes[len(channels):]:
        axis.axis("off")
    fig.supxlabel(
        "Matched time within 500-ms segment "
        "(gray: cue−500..0 ms, blue: cue+0..500 ms)"
    )
    fig.supylabel("Amplitude (µV)")
    fig.suptitle(
        "Lee2019 post-cue vs immediately pre-cue EEG by channel\n"
        "red bottom bar=max-T FWER p<0.05",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "subjects": int(len(erp)),
        "comparison": {
            "pre_cue_ms": [-500, 0],
            "post_cue_ms": [0, 500],
            "paired_mapping": "pre[-500+t] vs post[0+t]",
        },
        "permutations": args.permutations,
        "method": "paired sign-flip max-T across 62 channels and 125 time points",
        "max_t_threshold": threshold,
        "significant_channel_count": int(sum(
            item["fwer_significant"] for item in result.values()
        )),
        "channels": result,
        "interpretation_limit": (
            "Pre-cue segment is late fixation and may contain anticipation; "
            "it is not equivalent to resting state."
        ),
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        key: value for key, value in payload.items() if key != "channels"
    }, indent=2))


if __name__ == "__main__":
    main()
