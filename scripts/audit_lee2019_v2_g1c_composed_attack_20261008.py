"""E3 G1c' EEG-detector-output to gap-attack diagnostic, Lee simulation.

Exploratory because 1/2-s alternative waits were specified after observing
G1b detector diagnostics, though before any composed-attack result. The fixed
3.5-s E1 attack is retained. This cannot establish cue specificity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from simulate_composed_splice_attack_20261008 import composed_attack  # noqa: E402

WAITS = (1.0, 2.0, 3.5)
N_SHIFTS = 1000
N_BOOT = 2000
SEED = 20261008


def audit_one(row: dict, stream_root: Path) -> dict:
    subject, session = int(row["subject"]), int(row["session"])
    path = stream_root / f"session{session}/sub-{subject:02d}.npz"
    with np.load(path, allow_pickle=False) as stream:
        events = stream["events"]
        horizon = len(stream["signal"]) / 250.0
    observed = np.sort(np.asarray(row["detection_samples_available_at"], dtype=float) / 250.)
    # Identical edge restriction for actual and all time-shifted controls.
    idle = events[:, 2:4] / 250.0
    idle[:, 0] = np.maximum(idle[:, 0], 10.0)
    idle[:, 1] = np.minimum(idle[:, 1], horizon - 10.0)
    idle = idle[idle[:, 1] > idle[:, 0]]
    onsets = events[:, 6] / 250.0
    onsets = onsets[(onsets >= 10.) & (onsets + 2. <= horizon - 10.)]
    if len(onsets) < 25:
        raise AssertionError("edge restriction removed too many onsets")
    actual = [composed_attack(observed, onsets, idle, horizon, wait_s=wait)[0]
              for wait in WAITS]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, subject, session, 6]))
    offsets = rng.uniform(10.0, horizon - 10.0, size=N_SHIFTS)
    null_max = np.empty(N_SHIFTS)
    null_fixed = np.empty(N_SHIFTS)
    for i, offset in enumerate(offsets):
        shifted = np.sort((observed + offset) % horizon)
        values = [composed_attack(shifted, onsets, idle, horizon, wait_s=wait)[0]
                  for wait in WAITS]
        null_max[i] = max(values)
        null_fixed[i] = values[-1]
    actual_max = float(max(actual))
    return {"subject": subject, "session": session,
            "n_observed_detections": len(observed), "n_onsets_after_edge": len(onsets),
            "actual_hit_waits_1_2_3p5": actual,
            "actual_max_hit": actual_max,
            "null_shift_max_median_hit": float(np.median(null_max)),
            "paired_max_hit_excess": actual_max - float(np.median(null_max)),
            "fixed_3p5_hit_excess": float(actual[-1] - np.median(null_fixed))}


def summarize(rows: list[dict]) -> dict:
    result = {}
    for session in (1, 2):
        group = sorted((row for row in rows if row["session"] == session),
                       key=lambda row: row["subject"])
        delta = np.asarray([row["paired_max_hit_excess"] for row in group])
        fixed = np.asarray([row["fixed_3p5_hit_excess"] for row in group])
        rng = np.random.default_rng(np.random.SeedSequence([SEED, session, 7]))
        indices = rng.integers(0, len(delta), size=(N_BOOT, len(delta)))
        upper = float(np.quantile(np.median(delta[indices], axis=1), .95))
        result[str(session)] = {
            "median_attack_family_excess": float(np.median(delta)),
            "one_sided_95_upper_median_attack_family_excess": upper,
            "median_fixed_3p5_excess": float(np.median(fixed)),
            "nominal_g1c_threshold_0p01_pass": bool(upper < .01),
            "note": "exploratory family due post-G1b specification; not confirmatory G1 pass",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1b", type=Path, default=Path(
        "data/processed/simulation_redesign_20261008/e3_g1b_splice_detector_matched_window.json"))
    parser.add_argument("--stream-root", type=Path, default=Path(
        "data/interim/lee2019_v2_random_precue_balanced_20261008"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/simulation_redesign_20261008/e3_g1c_composed_attack.json"))
    args = parser.parse_args()
    source = json.loads(args.g1b.read_text(encoding="utf-8"))
    if len(source["rows"]) != 108:
        raise AssertionError("expected 108 source-fixed EEG detector rows")
    rows = [audit_one(row, args.stream_root) for row in source["rows"]]
    result = {"scope": "exploratory E3 G1c' on Lee cue-paced simulation",
              "g1b_source": str(args.g1b),
              "attack_waits_seconds": WAITS, "fixed_e1_wait_seconds": 3.5,
              "alternative_waits_specified_after_g1b": True,
              "null": "1000 circular shifts of detector availability times, 10s edge excluded",
              "bootstrap": "2000 subject resamples, session-specific median, one-sided 95% upper",
              "diagnostic_fp2": "target-idle oracle expected thinning; not deployable",
              "g2_cue_sham_complete": False, "g3_positive_control_complete": False,
              "session_summary": summarize(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
