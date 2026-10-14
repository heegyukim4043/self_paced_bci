"""Prospective target-pre-rest calibration on frozen chronological score caches.

The target pre-rest contains no MI labels. This is an operating-point audit,
not the matched-FP score-quality comparison or a validated EEG null test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from scripts.evaluate_lee2019_rest_sham_matched_fp import compute_guard_local_ma
from uncued_mi.decision import guard_rearmed_alarms
from uncued_mi.guard_rearm_calibration import tune_guard_rearmed_anchor


def eval_cache(path: Path, model: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_scores"], dtype=float)
        anchor_key = "anchor_raw_scores" if model == "riemannian" else "anchor_scores"
        anchor = np.asarray(data[anchor_key], dtype=float)
        times = np.asarray(data["times"], dtype=float)
        allowed = np.asarray(data["allowed"], dtype=bool)
        events = np.asarray(data["events"], dtype=np.int64)
        if model == "cnn_gru" and not np.array_equal(
            data["seeds"], np.asarray([20260729, 20260730, 20260731]),
        ):
            raise AssertionError(f"CNN-GRU model seeds changed: {path}")
    if len(raw) != len(times) or len(anchor) != 571 or len(events) != 50:
        raise AssertionError(f"unexpected score geometry: {path}")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(anchor)):
        raise AssertionError(f"nonfinite score: {path}")
    mean, var = float(anchor.mean()), float(anchor.var(ddof=1))
    anchor_z = ewma_z(anchor, mean, var, np.ones(len(anchor), dtype=bool))
    stream_ma = compute_guard_local_ma(raw, mean, var, allowed)
    future = times >= events[20, 2] / 250.
    task = np.zeros(len(times), dtype=bool)
    onsets = []
    for event in events[20:]:
        cue = event[5] / 250.
        onsets.append(cue + .5)
        task |= (times >= cue + .5) & (times <= cue + 4.)
    idle = future & allowed & ~task
    exposure_minutes = float(idle.sum() * .1 / 60.)
    results = {}
    for target in (2., 4.):
        point = tune_guard_rearmed_anchor(
            anchor_z, target, .1, ma_steps=5, dwell_steps=3,
            refractory_steps=10,
            admissible_half_width=.5 if target == 2. else 1.,
        )
        alarms = guard_rearmed_alarms(
            stream_ma, point.threshold, allowed, 3, 10,
        ) & future
        hits = [bool(np.any(alarms & (times >= onset) &
                            (times <= onset + 2.))) for onset in onsets]
        count = int(np.sum(alarms & idle))
        results[str(target)] = {
            "threshold": float(point.threshold),
            "anchor_fp_per_min": float(point.anchor_fp_per_min),
            "anchor_alarm_count": int(point.anchor_alarm_count),
            "anchor_within_target_band": bool(point.within_target_band),
            "future_fp_count": count,
            "future_fp_per_min": count / exposure_minutes,
            "future_onset_hits2": int(sum(hits)),
            "future_sen2": float(np.mean(hits)),
            "future_idle_exposure_min": exposure_minutes,
        }
    return {"times": times, "allowed": allowed, "events": events,
            "results": results}


def run(args: argparse.Namespace) -> dict:
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            r_path = args.riemannian_root / f"session{session}" / f"sub-{subject:02d}.npz"
            c_path = args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if not c_path.exists() or not r_path.exists():
                if args.allow_missing:
                    continue
                raise FileNotFoundError(f"paired cache missing: {r_path} or {c_path}")
            riem = eval_cache(r_path, "riemannian")
            cnn = eval_cache(c_path, "cnn_gru")
            for key in ("times", "allowed", "events"):
                if not np.array_equal(riem[key], cnn[key]):
                    raise AssertionError(f"paired {key} mismatch S{session} sub-{subject:02d}")
            for target in (2., 4.):
                rows.append({
                    "session": session, "subject": subject, "target_fp": target,
                    "riemannian": riem["results"][str(target)],
                    "cnn_gru": cnn["results"][str(target)],
                })
        print(f"anchor operating-point audit S{session} complete", flush=True)
    summary = {
        "scope": "prospective target-pre-rest anchor on chronological cue-relative simulation",
        "warning": "Unequal future achieved FP confounds raw SEN comparison; no EEG-level null or Type-I validation here.",
        "rows": rows,
        "by_session": {},
    }
    for session in args.sessions:
        for target in (2., 4.):
            matched = [r for r in rows if r["session"] == session and r["target_fp"] == target]
            entry = {"paired_subjects": len(matched)}
            for model in ("riemannian", "cnn_gru"):
                entry[model] = {
                    "anchor_within_target_band_fraction": float(np.mean([
                        r[model]["anchor_within_target_band"] for r in matched
                    ])) if matched else None,
                    "median_future_fp_per_min": float(np.median([
                        r[model]["future_fp_per_min"] for r in matched
                    ])) if matched else None,
                    "future_fp_q25_q75_per_min": np.quantile([
                        r[model]["future_fp_per_min"] for r in matched
                    ], [.25, .75]).tolist() if matched else None,
                    "future_fp_min_max_per_min": [
                        float(min(r[model]["future_fp_per_min"] for r in matched)),
                        float(max(r[model]["future_fp_per_min"] for r in matched)),
                    ] if matched else None,
                    "median_future_sen2": float(np.median([
                        r[model]["future_sen2"] for r in matched
                    ])) if matched else None,
                    "admissible_future_fp_fraction": float(np.mean([
                        abs(r[model]["future_fp_per_min"] - target) <=
                        (.5 if target == 2. else 1.) for r in matched
                    ])) if matched else None,
                }
            summary["by_session"][f"FP{target:g}_S{session}"] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--riemannian-root", type=Path,
                        default=ROOT / "data/interim/lee2019_revalidation_riemannian_scores")
    parser.add_argument("--cnn-root", type=Path,
                        default=ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/performance_revalidation_20261001/"
                        "a2_chronological_anchor_operating_points.json")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({"scope": summary["scope"],
                      "by_session": summary["by_session"]}, indent=2))


if __name__ == "__main__":
    main()
