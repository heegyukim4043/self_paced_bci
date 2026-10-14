"""Frozen E6 follow-up: nominal strict-support alarms and MA occupancy.

Lee original-continuous cue-paced diagnostic, not covert MI onset validation.
No threshold tuning on future trials. MA-shift null is algebraically evaluated
by shifting its threshold indicator on a circular timeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_e4_original_continuous_fp_20261009 import STEP, DWELL, REFRACTORY, _trial_strata, _union
from scripts.run_e4b_task_idle_20261010 import _prepare
from scripts.run_e4c_original_riemannian_fp_20261010 import prepare as prepare_riemannian
from scripts.run_e6_tall_fixed_onset_20261010 import ci_median
from uncued_mi.decision import guard_rearmed_alarms

SEED = 20261010
WINDOWS = {"tier0": 2.0, "riemannian": 1.0}


def collar_indices(times: np.ndarray, starts_s: np.ndarray, ends_s: np.ndarray):
    starts = np.searchsorted(times, starts_s, side="left")
    ends = np.searchsorted(times, ends_s, side="right")
    if np.any(starts < 0) or np.any(ends > len(times)) or np.any(ends <= starts):
        raise AssertionError("invalid collar geometry")
    return starts.astype(np.int32), ends.astype(np.int32)


def circular_counts(binary: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                    offsets: np.ndarray) -> np.ndarray:
    """Counts in fixed collars after circularly shifting original by offset."""
    n = len(binary)
    widths = ends - starts
    if np.any(widths <= 0) or np.any(widths > n):
        raise AssertionError("invalid circular collar")
    prefix = np.r_[0, np.cumsum(np.r_[binary, binary], dtype=np.int64)]
    shifted_starts = (starts[None, :] - offsets[:, None]) % n
    return prefix[shifted_starts + widths[None, :]] - prefix[shifted_starts]


def metric(binary: np.ndarray, starts: np.ndarray, ends: np.ndarray,
           offsets: np.ndarray, keep: np.ndarray, occupancy: bool = False):
    observed_counts = circular_counts(binary, starts, ends, np.array([0], dtype=np.int32))[0]
    null_counts = circular_counts(binary, starts, ends, offsets)
    observed_hit = float(np.mean(observed_counts > 0))
    null_hit = np.mean(null_counts > 0, axis=1)
    result = {
        "n_events": int(len(starts)),
        "observed_any": observed_hit,
        "null_any_mean": float(np.mean(null_hit)),
        "excess_any": observed_hit - float(np.mean(null_hit)),
        "null_any_period_excluded_mean": float(np.mean(null_hit[keep])),
        "excess_any_period_excluded": observed_hit - float(np.mean(null_hit[keep])),
        "null_any_q05_q50_q95": [float(x) for x in np.quantile(null_hit, (.05, .5, .95))],
    }
    if occupancy:
        widths = ends - starts
        observed_frac = float(np.mean(observed_counts / widths))
        null_frac = np.mean(null_counts / widths[None, :], axis=1)
        result.update({
            "observed_fraction": observed_frac,
            "null_fraction_mean": float(np.mean(null_frac)),
            "excess_fraction": observed_frac - float(np.mean(null_frac)),
            "null_fraction_period_excluded_mean": float(np.mean(null_frac[keep])),
            "excess_fraction_period_excluded": observed_frac - float(np.mean(null_frac[keep])),
            "null_fraction_q05_q50_q95": [float(x) for x in np.quantile(null_frac, (.05, .5, .95))],
        })
    return result


def allowed_offsets(n: int, period: float):
    min_steps = int(round(10.0 / STEP))
    offsets = np.arange(min_steps + 1, n - min_steps, dtype=np.int32)
    if len(offsets) < 1000:
        raise AssertionError("too few null offsets")
    seconds = offsets * STEP
    multiple = np.rint(seconds / period)
    keep = ~((multiple >= 1) & (np.abs(seconds - multiple * period) <= 2.0 + 1e-12))
    if not np.any(keep):
        raise AssertionError("no period-excluded offsets")
    return offsets, keep


def one(model: str, frozen_row: dict, tier0_root: Path, riemannian_root: Path,
        old_e6: dict | None):
    subject, session = frozen_row["subject"], frozen_row["session"]
    tier0 = tier0_root / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz"
    if model == "tier0":
        times, cues, fixation, iti, ma = _prepare(tier0, subject, session)
    else:
        score = riemannian_root / f"session{session}" / f"sub-{subject:02d}.npz"
        times, cues, fixation, iti, ma = prepare_riemannian(score, tier0, subject, session)
    if len(cues) != 100 or len(fixation) != 100 or WINDOWS[model] not in (1., 2.):
        raise AssertionError("unexpected score/event geometry")
    idle = _union(_trial_strata(times, cues, fixation, iti, list(range(70, 100))), False)
    threshold = float(frozen_row["calibration"]["threshold"])
    alarms = guard_rearmed_alarms(ma, threshold, np.ones(len(ma), dtype=bool),
                                   dwell_steps=DWELL, refractory_steps=REFRACTORY)
    observed_idle = {"count": int(np.sum(alarms & idle)),
                     "seconds": float(np.sum(idle) * STEP)}
    frozen_idle = frozen_row["evaluation"]["E_full"]
    if observed_idle["count"] != frozen_idle["count"] or not np.isclose(
            observed_idle["seconds"], frozen_idle["seconds"], atol=1e-9):
        raise AssertionError(f"{model} E4b/E4c frozen idle count mismatch")
    i0 = int(np.flatnonzero(idle)[0])
    i1 = int(np.searchsorted(times, cues[99] + 4., side="right"))
    if np.any(idle[:i0]) or np.any(idle[i1:]):
        raise AssertionError("future idle outside evaluation segment")
    segment_times = times[i0:i1]
    segment_alarm = alarms[i0:i1].astype(np.int8)
    segment_ma = ma[i0:i1]
    segment_occ = (segment_ma >= threshold).astype(np.int8)
    future_cues = cues[70:100]
    collar_sec = {
        "primary": (future_cues + .5, future_cues + 2.5),
        "fixation": (future_cues - 2.5, future_cues - .5),
        "strict": (future_cues + .5 + WINDOWS[model], future_cues + 3.5),
        "common_late": (future_cues + 2.5, future_cues + 3.5),
    }
    collars = {name: collar_indices(segment_times, *pair)
               for name, pair in collar_sec.items()}
    for name, (starts, ends) in collars.items():
        if len(starts) != 30 or np.any(ends[:-1] > starts[1:]):
            raise AssertionError(f"overlapping {name} collars")
    offsets, keep = allowed_offsets(len(segment_alarm), float(np.mean(np.diff(future_cues))))
    alarm_metrics = {name: metric(segment_alarm, *collars[name], offsets, keep)
                     for name in collars}
    occupancy_metrics = {name: metric(segment_occ, *collars[name], offsets, keep, True)
                         for name in ("strict", "common_late")}
    if old_e6 is not None:
        if (not np.isclose(alarm_metrics["primary"]["observed_any"], old_e6["mi"]["sen_2s"], atol=1e-12)
            or not np.isclose(alarm_metrics["primary"]["null_any_mean"], old_e6["mi"]["null_sen_2s"], atol=2e-6)
            or len(offsets) != old_e6["null"]["n_primary_offsets"]):
            raise AssertionError("tier0 primary E6 event/null mismatch")
    return {
        "model": model, "subject": subject, "session": session,
        "threshold": threshold, "window_seconds": WINDOWS[model],
        "future_idle": {**observed_idle, "fp_per_min": observed_idle["count"] * 60. / observed_idle["seconds"]},
        "null_offsets": {"primary": int(len(offsets)), "period_excluded": int(np.sum(keep))},
        "alarm": alarm_metrics, "occupancy": occupancy_metrics,
    }


def summarize(rows: list[dict]):
    out = {}
    for model in WINDOWS:
        out[model] = {}
        for session in (1, 2):
            sub = [r for r in rows if r["model"] == model and r["session"] == session]
            if len(sub) != 54 or len({r["subject"] for r in sub}) != 54:
                raise AssertionError("incomplete subject cohort")
            fields = {
                "fp_per_min": lambda r: r["future_idle"]["fp_per_min"],
                "primary_sen2": lambda r: r["alarm"]["primary"]["observed_any"],
                "primary_excess": lambda r: r["alarm"]["primary"]["excess_any"],
                "primary_excess_period_excluded": lambda r: r["alarm"]["primary"]["excess_any_period_excluded"],
                "fixation_excess": lambda r: r["alarm"]["fixation"]["excess_any"],
                "strict_alarm_hit": lambda r: r["alarm"]["strict"]["observed_any"],
                "strict_alarm_excess": lambda r: r["alarm"]["strict"]["excess_any"],
                "strict_alarm_excess_period_excluded": lambda r: r["alarm"]["strict"]["excess_any_period_excluded"],
                "strict_occ_any": lambda r: r["occupancy"]["strict"]["observed_any"],
                "strict_occ_any_excess": lambda r: r["occupancy"]["strict"]["excess_any"],
                "strict_occ_fraction": lambda r: r["occupancy"]["strict"]["observed_fraction"],
                "strict_occ_fraction_excess": lambda r: r["occupancy"]["strict"]["excess_fraction"],
                "common_alarm_excess": lambda r: r["alarm"]["common_late"]["excess_any"],
                "common_occ_any_excess": lambda r: r["occupancy"]["common_late"]["excess_any"],
                "common_occ_fraction_excess": lambda r: r["occupancy"]["common_late"]["excess_fraction"],
                "strict_occ_minus_alarm_hit": lambda r: (r["occupancy"]["strict"]["observed_any"] -
                                                           r["alarm"]["strict"]["observed_any"]),
            }
            block = {}
            for j, (name, fn) in enumerate(fields.items()):
                values = np.asarray([fn(r) for r in sub], dtype=float)
                block[name] = {"median": float(np.median(values)),
                               "mean": float(np.mean(values)),
                               "subject_bootstrap_median_ci95": ci_median(values, SEED + 1000 * session + 100 * (model == "riemannian") + j)}
            block["fp_admissible_3_to_5"] = int(sum(3 <= r["future_idle"]["fp_per_min"] <= 5 for r in sub))
            block["fp_q90"] = float(np.quantile([r["future_idle"]["fp_per_min"] for r in sub], .9))
            out[model][str(session)] = block
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--riemannian-root", type=Path, default=Path("data/interim/lee2019_e4c_riemannian_original"))
    ap.add_argument("--e4b", type=Path, default=Path("data/processed/e4b_task_idle_fp_20261010.json"))
    ap.add_argument("--e4c", type=Path, default=Path("data/processed/e4c_original_riemannian_fp_20261010.json"))
    ap.add_argument("--e6", type=Path, default=Path("data/processed/e6_tall_fixed_onset_20261010_v2.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/e6_strict_e6r_20261010.json"))
    ap.add_argument("--limit-cases", type=int, default=0, help="preflight only; no output")
    args = ap.parse_args()
    if args.output.exists() and not args.limit_cases:
        raise FileExistsError(args.output)
    frozen = {
        "tier0": json.loads(args.e4b.read_text(encoding="utf-8")),
        "riemannian": json.loads(args.e4c.read_text(encoding="utf-8")),
    }
    old = json.loads(args.e6.read_text(encoding="utf-8"))
    old_rows = {(r["subject"], r["session"]): r for r in old["rows"]}
    rows = []
    for model in WINDOWS:
        selected = [r for r in frozen[model]["rows"] if r["target_fp"] == 4.0 and r["arm"] == "T_all"]
        if len(selected) != 108 or len({(r["subject"], r["session"]) for r in selected}) != 108:
            raise AssertionError("frozen FP4 cohort incomplete")
        for j, r in enumerate(sorted(selected, key=lambda x: (x["session"], x["subject"]))):
            rows.append(one(model, r, args.tier0_root, args.riemannian_root,
                            old_rows[(r["subject"], r["session"])] if model == "tier0" else None))
            if (j + 1) % 10 == 0:
                print(f"E6 strict {model} {j+1}/108", flush=True)
            if args.limit_cases and j + 1 >= args.limit_cases:
                print(f"preflight only: {model} {j+1} case(s); no output")
                break
        if args.limit_cases:
            continue
    if args.limit_cases:
        return
    result = {
        "status": "Lee2019 original-continuous post-E6 exploratory diagnostic; not covert MI specificity",
        "design": "E6_STRICT_SUPPORT_E6R_PRERUN_DESIGN_20261010.md, pre-run amendment 1",
        "unit": "subject; S1 and S2 separate; 2000 subject-bootstrap draws",
        "frozen_operating_point": "E4b/E4c T-all FP4; no future tuning",
        "ma_null": "circular shift MA series; threshold after shift; binary indicator circular identity used in implementation",
        "summary": summarize(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
