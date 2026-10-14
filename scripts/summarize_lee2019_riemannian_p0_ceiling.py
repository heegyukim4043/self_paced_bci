"""Combine frozen subset/holdout P0 results and make an audit figure."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_lee2019_riemannian_p0_ceiling import (
    ALL_SUBJECTS,
    HOLDOUT34,
    SUBSET20,
    summarize,
)


PROTOCOL_SCRIPT = ROOT / "scripts" / "evaluate_lee2019_riemannian_p0_ceiling.py"
EXPECTED_SHA256 = "B8583068C36C891E2C6F05D019291F6D285EBEE03A9505AA2AD3F7C765D612D6"
SUBSET_PATH = ROOT / "data" / "processed" / "lee2019_riemannian_p0_ceiling_subset20.json"
HOLDOUT_PATH = ROOT / "data" / "processed" / "lee2019_riemannian_p0_ceiling_holdout34.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "lee2019_riemannian_p0_ceiling_full54.json"
FIGURE_PATH = ROOT / "data" / "processed" / "lee2019_riemannian_p0_ceiling_paired.png"


def load(path: Path, expected_cohort: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["cohort"] != expected_cohort:
        raise AssertionError(f"unexpected cohort in {path}")
    return payload


def value(row: dict, arm: str, metric: str) -> float:
    target = row["arms"][arm]["targets"]["2.0"]
    if metric == "excess":
        return float(target["circular_shift_excess"]["1.0"])
    if metric == "coverage":
        return float(target["coverage"])
    if metric == "auc":
        return float(row["arms"][arm]["window_auc"])
    raise ValueError(metric)


def make_figure(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    rng = np.random.default_rng(20260924)
    subset = set(SUBSET20)
    for column, session in enumerate((1, 2)):
        selected = sorted(
            [row for row in rows if int(row["session"]) == session],
            key=lambda row: int(row["subject"]),
        )
        for row_index, (metric, label) in enumerate(
            (("excess", "Excess onset sensitivity @1 s, FP=2/min"),
             ("coverage", "Interval coverage, FP=2/min"))
        ):
            ax = axes[row_index, column]
            for row in selected:
                before = value(row, "first50_supervised", metric)
                after = value(row, "eval_oof_oracle", metric)
                color = "#1f77b4" if int(row["subject"]) in subset else "#9aa0a6"
                jitter = rng.normal(0, 0.015)
                ax.plot([0 + jitter, 1 + jitter], [before, after], color=color, alpha=.32, lw=.8)
                ax.scatter([0 + jitter, 1 + jitter], [before, after], color=color, alpha=.55, s=10)
            medians = [
                np.median([value(row, arm, metric) for row in selected])
                for arm in ("first50_supervised", "eval_oof_oracle")
            ]
            ax.plot([0, 1], medians, color="black", marker="o", lw=2.5, ms=6, zorder=10)
            ax.set_xticks([0, 1], ["first-50\nmodel", "evaluation-OOF\noracle"])
            ax.set_xlim(-.18, 1.18)
            ax.grid(axis="y", alpha=.22)
            ax.set_title(f"Session {session}")
            if column == 0:
                ax.set_ylabel(label)
    fig.suptitle(
        "Lee2019 P0 diagnostic ceiling: target-specific within-evaluation cross-fitting\n"
        "Blue=subset20, gray=locked holdout34; black=full-cohort median",
        fontsize=13,
    )
    fig.text(
        .5, .015,
        "Diagnostic oracle only: evaluation-trial labels and evaluation-idle FP calibration are used; cue-relative pseudo-onsets.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, .045, 1, .93))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    digest = hashlib.sha256(PROTOCOL_SCRIPT.read_bytes()).hexdigest().upper()
    if digest != EXPECTED_SHA256:
        raise AssertionError(f"protocol script changed after holdout lock: {digest}")
    subset_payload = load(SUBSET_PATH, "subset20")
    holdout_payload = load(HOLDOUT_PATH, "holdout34")
    rows = subset_payload["rows"] + holdout_payload["rows"]
    pairs = {(int(row["subject"]), int(row["session"])) for row in rows}
    expected = {(subject, session) for subject in ALL_SUBJECTS for session in (1, 2)}
    if pairs != expected or len(rows) != 108:
        raise AssertionError("subset and holdout do not form the expected 54 x 2 grid")
    if set(SUBSET20) & set(HOLDOUT34):
        raise AssertionError("development and holdout subject sets overlap")
    payload = {
        "protocol_sha256": digest,
        "subset20_summary": subset_payload["summary"],
        "holdout34_summary": holdout_payload["summary"],
        "full54_summary": summarize(rows, bootstrap=10000, seed=20260924),
        "rows": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_figure(rows)
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {FIGURE_PATH}")
    print(json.dumps(payload["full54_summary"], indent=2))


if __name__ == "__main__":
    main()
