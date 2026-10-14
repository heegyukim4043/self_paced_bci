"""Validate and summarize frozen Stieger CNN-GRU score caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def quantiles(values: list[float]) -> dict[str, float]:
    points = np.quantile(np.asarray(values, dtype=float), [0, 0.25, 0.5, 0.75, 1])
    return dict(zip(("min", "q25", "median", "q75", "max"), map(float, points)))


def validate_one(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    with np.load(path, allow_pickle=False) as cache:
        scores = np.asarray(cache["scores"], dtype=float)
        seed_scores = np.asarray(cache["seed_scores"], dtype=float)
        offsets = np.asarray(cache["trial_offsets"], dtype=int)
        trial_index = np.asarray(cache["trial_index"], dtype=int)
        times = np.asarray(cache["time_from_feedback_s"], dtype=float)
        feedback = np.asarray(cache["feedback_mask"], dtype=bool)
        preparation = np.asarray(cache["preparation_mask"], dtype=bool)
        targets = np.asarray(cache["target_number"], dtype=int)
        labels = np.asarray(cache["window_label_active_feedback"], dtype=int)
        trial_artifact = np.asarray(cache["trial_artifact"], dtype=bool)
        trial_lengths = np.asarray(cache["trial_length_s"], dtype=float)
        seeds = np.asarray(cache["seeds"], dtype=int)
        audit = json.loads(str(cache["audit_json"]))

        n = len(scores)
        if seed_scores.shape != (3, n):
            errors.append(f"seed_scores shape {seed_scores.shape} != (3,{n})")
        if not np.all(np.isfinite(seed_scores)) or not np.all(np.isfinite(scores)):
            errors.append("non-finite scores")
        if seed_scores.shape[0] and not np.allclose(scores, seed_scores.mean(axis=0), atol=1e-6):
            errors.append("ensemble score is not seed mean")
        if len(offsets) != 451 or offsets[0] != 0 or offsets[-1] != n:
            errors.append("trial offsets are inconsistent")
        if np.any(np.diff(offsets) < 0):
            errors.append("trial offsets are not monotone")
        if not all(len(array) == n for array in (trial_index, times, feedback, preparation, targets, labels)):
            errors.append("flat arrays have inconsistent lengths")
        if np.any(feedback == preparation) or not np.all(feedback | preparation):
            errors.append("feedback/preparation masks are not complementary")
        expected_labels = (feedback & (targets != 4)).astype(int)
        if not np.array_equal(labels, expected_labels):
            errors.append("window labels violate active-feedback definition")
        if n and np.any(trial_artifact[trial_index]):
            errors.append("artifact trial contributed score windows")
        if n and (np.min(times) < -1.00001 or np.any(times > trial_lengths[trial_index] + 1e-5)):
            errors.append("window times exceed causal cell support")
        for trial in range(450):
            values = times[offsets[trial] : offsets[trial + 1]]
            if len(values) > 1 and np.any(np.diff(values) <= 0):
                errors.append(f"trial {trial} times are not increasing")
                break
        if tuple(seeds.tolist()) != (20260729, 20260730, 20260731):
            errors.append(f"unexpected seeds {seeds.tolist()}")
        if int(audit["training"]["epochs"]) != 10:
            errors.append("epochs != 10")
        if int(audit["parameter_count"]) != 6453:
            errors.append("parameter count != 6453")
        if not all(item["passed"] for item in audit["causality_reset_audits"]):
            errors.append("saved causality/reset audit failed")
        if not audit["seam_contract"]["hidden_reset_each_cell"]:
            errors.append("hidden reset contract missing")
        if not audit["seam_contract"]["normalizer_fit_on_first_half_only"]:
            errors.append("normalization leakage contract missing")

        correlations = []
        for first in range(seed_scores.shape[0]):
            for second in range(first + 1, seed_scores.shape[0]):
                if np.std(seed_scores[first]) > 0 and np.std(seed_scores[second]) > 0:
                    correlations.append(float(np.corrcoef(seed_scores[first], seed_scores[second])[0, 1]))
        losses = [item["epoch_mean_loss"] for item in audit["training"]["seed_audits"]]
        prefix_errors = [item["prefix_causality_max_abs_error"] for item in audit["causality_reset_audits"]]
        reset_errors = [item["independent_vs_batched_reset_max_abs_error"] for item in audit["causality_reset_audits"]]

    return {
        "path": str(path.resolve()),
        "subject": int(row["subject"]),
        "session": int(row["session"]),
        "passed": not errors,
        "errors": errors,
        "score_windows": n,
        "score_sd": float(np.std(scores, ddof=1)),
        "pairwise_seed_correlation_median": float(np.median(correlations)),
        "training_loss_first_median": float(np.median([value[0] for value in losses])),
        "training_loss_last_median": float(np.median([value[-1] for value in losses])),
        "all_seed_losses_decreased": bool(all(value[-1] < value[0] for value in losses)),
        "prefix_causality_max_abs_error": float(max(prefix_errors)),
        "reset_equivalence_max_abs_error": float(max(reset_errors)),
        "file_size_mb": float(path.stat().st_size / (1024**2)),
    }


def render(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# Stieger2021 frozen CNN-GRU score-cache validation",
        "",
        "## Decision",
        "",
        f"- Valid caches: **{s['passed_subject_sessions']}/{s['subject_sessions']}**",
        f"- All validation gates passed: **{s['all_passed']}**",
        "- Scope: trial-wise preparation-to-feedback partial external validation",
        "",
        "## Cache summary",
        "",
        f"- Total score windows: {s['total_score_windows']:,}",
        f"- Total compressed cache size: {s['total_cache_size_mb']:.1f} MB",
        f"- Windows/session: {s['score_windows']['min']:.0f}--{s['score_windows']['max']:.0f} (median {s['score_windows']['median']:.0f})",
        f"- Pairwise seed correlation: median {s['seed_correlation']['median']:.3f}, range {s['seed_correlation']['min']:.3f}--{s['seed_correlation']['max']:.3f}",
        f"- Score SD/session: median {s['score_sd']['median']:.3f}, range {s['score_sd']['min']:.3f}--{s['score_sd']['max']:.3f}",
        f"- Maximum prefix-causality error: {s['max_prefix_causality_error']:.3g}",
        f"- Maximum reset-equivalence error: {s['max_reset_equivalence_error']:.3g}",
        "",
        "## Training QC",
        "",
        f"- Sessions where every seed's final loss was below its first-epoch loss: {s['all_seed_loss_decrease_sessions']}/{s['subject_sessions']}",
        f"- First-epoch median loss across sessions: {s['first_loss']['median']:.4f}",
        f"- Final-epoch median loss across sessions: {s['last_loss']['median']:.4f}",
        "",
        "The cache stores raw logits only. Threshold, EWMA, MA, dwell, and refractory are intentionally deferred to the locked evaluation stage.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/stieger2021_cnn_gru_score_cache_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stieger2021_cnn_gru_score_cache_validation.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("STIEGER_CNN_GRU_SCORE_CACHE_20260921.md"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [validate_one(Path(row["target"]), row) for row in manifest["rows"]]
    summary = {
        "subject_sessions": len(rows),
        "passed_subject_sessions": sum(row["passed"] for row in rows),
        "all_passed": bool(rows and all(row["passed"] for row in rows)),
        "total_score_windows": int(sum(row["score_windows"] for row in rows)),
        "total_cache_size_mb": float(sum(row["file_size_mb"] for row in rows)),
        "score_windows": quantiles([row["score_windows"] for row in rows]),
        "score_sd": quantiles([row["score_sd"] for row in rows]),
        "seed_correlation": quantiles([row["pairwise_seed_correlation_median"] for row in rows]),
        "first_loss": quantiles([row["training_loss_first_median"] for row in rows]),
        "last_loss": quantiles([row["training_loss_last_median"] for row in rows]),
        "all_seed_loss_decrease_sessions": sum(row["all_seed_losses_decreased"] for row in rows),
        "max_prefix_causality_error": float(max(row["prefix_causality_max_abs_error"] for row in rows)),
        "max_reset_equivalence_error": float(max(row["reset_equivalence_max_abs_error"] for row in rows)),
        "failed": [row for row in rows if not row["passed"]],
    }
    payload = {"summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report.write_text(render(payload), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit("one or more score caches failed validation")


if __name__ == "__main__":
    main()
