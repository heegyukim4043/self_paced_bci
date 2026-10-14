"""Monte-Carlo feasibility audit for the direct-collection timing protocol.

This simulation evaluates acquisition burden only. It never uses detector
performance and therefore cannot tune the EEG model or operating point.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROFILES = {
    "early_beta": (2.0, 5.0),
    "uniform": None,
    "late_beta": (5.0, 2.0),
}
BLOCK_SECONDS = (600.0, 660.0, 720.0)
NO_INIT = (0.0, 0.1, 0.2, 0.3)
QUOTAS = (20, 25, 50)


def draw_wait(rng: np.random.Generator, profile: str) -> float:
    shape = PROFILES[profile]
    if shape is None:
        unit = rng.random()
    else:
        unit = rng.beta(*shape)
    return 3.0 + 22.0 * unit


def target_stream(rng: np.random.Generator):
    while True:
        block = np.array([0, 0, 1, 1], dtype=np.int8)
        rng.shuffle(block)
        for side in block:
            yield int(side)


def simulate_block(
    rng: np.random.Generator,
    duration_s: float,
    profile: str,
    p_no_init: float,
) -> tuple[int, int, int]:
    time_s = 0.0
    left = 0
    right = 0
    no_init = 0
    targets = target_stream(rng)
    while True:
        recovery = rng.uniform(4.0, 10.0)
        target_time = time_s + recovery
        if target_time >= duration_s:
            break
        side = next(targets)
        if rng.random() < p_no_init:
            finish = target_time + 25.0
            if finish > duration_s:
                break
            no_init += 1
            time_s = finish
            continue
        wait = draw_wait(rng, profile)
        mi_duration = rng.uniform(2.0, 6.0)
        finish = target_time + wait + mi_duration
        if finish > duration_s:
            break
        if side == 0:
            left += 1
        else:
            right += 1
        time_s = finish
    return left, right, no_init


def simulate_quota_blocks(
    rng: np.random.Generator,
    quota_per_side: int,
    profile: str,
    p_no_init: float,
) -> tuple[int, float, int, int]:
    left = 0
    right = 0
    blocks = 0
    active_s = 0.0
    while left < quota_per_side or right < quota_per_side:
        duration_s = BLOCK_SECONDS[blocks % len(BLOCK_SECONDS)]
        l_count, r_count, no_init = simulate_block(
            rng, duration_s, profile, p_no_init
        )
        left += l_count
        right += r_count
        blocks += 1
        active_s += duration_s
        if blocks > 50:
            raise RuntimeError("Quota simulation exceeded 50 blocks")
    return blocks, active_s, left, right


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
    }


def run(seed: int, draws: int) -> dict:
    rng = np.random.default_rng(seed)
    block_results = []
    for profile in PROFILES:
        for p_no_init in NO_INIT:
            for duration_s in BLOCK_SECONDS:
                samples = np.array(
                    [
                        simulate_block(rng, duration_s, profile, p_no_init)
                        for _ in range(draws)
                    ],
                    dtype=float,
                )
                total = samples[:, 0] + samples[:, 1]
                imbalance = np.abs(samples[:, 0] - samples[:, 1])
                block_results.append(
                    {
                        "profile": profile,
                        "p_no_init": p_no_init,
                        "block_seconds": duration_s,
                        "successful_events": quantiles(total),
                        "left_events": quantiles(samples[:, 0]),
                        "right_events": quantiles(samples[:, 1]),
                        "absolute_side_imbalance": quantiles(imbalance),
                        "no_initiation_count": quantiles(samples[:, 2]),
                    }
                )

    quota_results = []
    quota_draws = max(3000, draws // 2)
    for profile in PROFILES:
        for p_no_init in NO_INIT:
            for quota in QUOTAS:
                samples = np.array(
                    [
                        simulate_quota_blocks(
                            rng, quota, profile, p_no_init
                        )
                        for _ in range(quota_draws)
                    ],
                    dtype=float,
                )
                blocks = samples[:, 0]
                active_minutes = samples[:, 1] / 60.0
                lower_bound_minutes = active_minutes + 28.0
                with_breaks_minutes = lower_bound_minutes + np.maximum(
                    blocks - 1.0, 0.0
                ) * 5.0
                quota_results.append(
                    {
                        "profile": profile,
                        "p_no_init": p_no_init,
                        "quota_per_side": quota,
                        "draws": quota_draws,
                        "blocks": quantiles(blocks),
                        "active_self_paced_minutes": quantiles(active_minutes),
                        "lower_bound_session_minutes": quantiles(
                            lower_bound_minutes
                        ),
                        "session_minutes_with_5min_interblock_breaks": quantiles(
                            with_breaks_minutes
                        ),
                        "probability_within_2_blocks": float(np.mean(blocks <= 2)),
                        "probability_within_3_blocks": float(np.mean(blocks <= 3)),
                        "probability_within_4_blocks": float(np.mean(blocks <= 4)),
                        "probability_within_5_blocks": float(np.mean(blocks <= 5)),
                    }
                )

    return {
        "seed": seed,
        "draws_per_block_scenario": draws,
        "quota_draws_per_scenario": quota_draws,
        "assumptions": {
            "recovery_s": "Uniform(4,10)",
            "onset_wait_s": {
                "early_beta": "3 + 22*Beta(2,5)",
                "uniform": "Uniform(3,25)",
                "late_beta": "3 + 22*Beta(5,2)",
            },
            "mi_duration_s": "Uniform(2,6)",
            "no_initiation_consumes_s": 25,
            "target_balance": "permuted blocks of 2 left + 2 right",
            "lower_bound_fixed_minutes": {
                "baseline": 10,
                "scripted_adversarial": 18,
                "excluded": "setup, marker-only controls, optional free behavior, non-self-paced transitions",
            },
        },
        "block_results": block_results,
        "quota_results": quota_results,
    }


def write_csv(result: dict, path: Path) -> None:
    fields = [
        "profile",
        "p_no_init",
        "quota_per_side",
        "blocks_median",
        "blocks_p95",
        "active_minutes_median",
        "active_minutes_p95",
        "lower_bound_minutes_median",
        "with_breaks_minutes_median",
        "prob_within_2",
        "prob_within_3",
        "prob_within_4",
        "prob_within_5",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["quota_results"]:
            writer.writerow(
                {
                    "profile": row["profile"],
                    "p_no_init": row["p_no_init"],
                    "quota_per_side": row["quota_per_side"],
                    "blocks_median": row["blocks"]["median"],
                    "blocks_p95": row["blocks"]["p95"],
                    "active_minutes_median": row[
                        "active_self_paced_minutes"
                    ]["median"],
                    "active_minutes_p95": row["active_self_paced_minutes"]["p95"],
                    "lower_bound_minutes_median": row[
                        "lower_bound_session_minutes"
                    ]["median"],
                    "with_breaks_minutes_median": row[
                        "session_minutes_with_5min_interblock_breaks"
                    ]["median"],
                    "prob_within_2": row["probability_within_2_blocks"],
                    "prob_within_3": row["probability_within_3_blocks"],
                    "prob_within_4": row["probability_within_4_blocks"],
                    "prob_within_5": row["probability_within_5_blocks"],
                }
            )


def select_rows(result: dict, profile: str, p_no_init: float):
    return [
        row
        for row in result["quota_results"]
        if row["profile"] == profile
        and np.isclose(row["p_no_init"], p_no_init)
    ]


def plot(result: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    for p_no_init in NO_INIT:
        rows = [
            row
            for row in result["block_results"]
            if row["profile"] == "uniform"
            and np.isclose(row["p_no_init"], p_no_init)
        ]
        rows.sort(key=lambda row: row["block_seconds"])
        ax.plot(
            [row["block_seconds"] / 60 for row in rows],
            [row["successful_events"]["median"] for row in rows],
            marker="o",
            label=f"no-init {int(100*p_no_init)}%",
        )
    ax.set_title("A. Successful events per block (uniform onset wait)")
    ax.set_xlabel("Block length (min)")
    ax.set_ylabel("Median completed events")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    x = np.arange(len(NO_INIT))
    for profile in PROFILES:
        rows = [
            row
            for row in result["quota_results"]
            if row["profile"] == profile and row["quota_per_side"] == 50
        ]
        rows.sort(key=lambda row: row["p_no_init"])
        ax.plot(
            x,
            [row["active_self_paced_minutes"]["median"] for row in rows],
            marker="o",
            label=profile,
        )
    ax.set_xticks(x, [f"{int(100*p)}%" for p in NO_INIT])
    ax.set_title("B. Active MI time to 50 events/side")
    ax.set_xlabel("No-initiation probability")
    ax.set_ylabel("Median active acquisition (min)")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    rows = select_rows(result, "uniform", 0.1)
    rows.sort(key=lambda row: row["quota_per_side"])
    quotas = [row["quota_per_side"] for row in rows]
    lower = [row["lower_bound_session_minutes"]["median"] for row in rows]
    breaks = [
        row["session_minutes_with_5min_interblock_breaks"]["median"]
        for row in rows
    ]
    width = 0.36
    xx = np.arange(len(quotas))
    ax.bar(xx - width / 2, lower, width, label="no inter-block breaks")
    ax.bar(xx + width / 2, breaks, width, label="5-min breaks")
    ax.set_xticks(xx, [str(q) for q in quotas])
    ax.set_title("C. Lower-bound session burden (uniform, no-init 10%)")
    ax.set_xlabel("Required completed events per side")
    ax.set_ylabel("Median minutes\n(+10-min baseline, +18-min adversarial)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    row = [r for r in rows if r["quota_per_side"] == 50][0]
    probs = [
        row["probability_within_2_blocks"],
        row["probability_within_3_blocks"],
        row["probability_within_4_blocks"],
        row["probability_within_5_blocks"],
    ]
    ax.bar([2, 3, 4, 5], probs, color="#4C78A8")
    ax.axhline(0.9, color="#D62728", linestyle="--", label="90%")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_title("D. Probability of reaching 50 events/side")
    ax.set_xlabel("Completed 10/11/12-min blocks")
    ax.set_ylabel("Probability")
    ax.legend()

    fig.suptitle(
        "Direct-collection event-yield and burden simulation",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "direct_collection_event_yield_simulation"
    result = run(args.seed, args.draws)
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_csv(result, args.output_dir / f"{stem}.csv")
    plot(result, args.output_dir / f"{stem}.png")

    primary = [
        row
        for row in result["quota_results"]
        if row["profile"] == "uniform"
        and np.isclose(row["p_no_init"], 0.1)
        and row["quota_per_side"] == 50
    ][0]
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
