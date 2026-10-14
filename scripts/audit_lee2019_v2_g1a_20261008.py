"""E3 G1a: seam-blind decision boundary controls on constructed timelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uncued_mi.decision import moving_average_dwell  # noqa: E402


def audit(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        n = len(saved["signal"])
        seams = saved["seams"][:, 0] / 250.0
    times = np.arange(250, n + 1, 25) / 250.0
    constant_high = moving_average_dwell(np.ones(len(times)), 0.0, .1,
                                         ma_seconds=.5, dwell_seconds=.3,
                                         refractory_seconds=1.)
    constant_low = moving_average_dwell(np.zeros(len(times)), 1.0, .1,
                                        ma_seconds=.5, dwell_seconds=.3,
                                        refractory_seconds=1.)
    high_times = times[constant_high.alarms]
    if len(high_times) != 1 or high_times[0] > times[0] + 1 or np.any(constant_low.alarms):
        raise AssertionError("seam-free constant-score boundary control failed")
    # A single pre-seam rise is not allowed to retrigger at any later seam.
    before_seam = float(np.min(seams))
    plateau = np.where(times < max(1.0, before_seam - 1.0), -1.0, 1.0)
    plateau_result = moving_average_dwell(plateau, 0.0, .1,
                                          ma_seconds=.5, dwell_seconds=.3,
                                          refractory_seconds=1.)
    plateau_times = times[plateau_result.alarms]
    if len(plateau_times) != 1 or plateau_times[0] >= before_seam:
        raise AssertionError("plateau re-triggered at a seam")
    return {"file": str(path), "n_seams": len(seams),
            "constant_high_alarm_count": len(high_times),
            "constant_low_alarm_count": int(np.sum(constant_low.alarms)),
            "plateau_alarm_count": len(plateau_times),
            "constant_alarm_within_0p5s_of_seam": int(np.sum(
                np.min(np.abs(seams[:, None] - high_times[None, :]), axis=0) <= .5)),
            "plateau_retrigger_after_first_seam": int(np.sum(plateau_times >= before_seam))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("data/interim/lee2019_v2_random_precue_balanced_20261008"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/simulation_redesign_20261008/e3_g1a_boundary.json"))
    parser.add_argument("--expect-files", type=int, default=108)
    args = parser.parse_args()
    paths = sorted(args.root.glob("session*/sub-*.npz"))
    if len(paths) != args.expect_files:
        raise AssertionError(f"expected {args.expect_files}, found {len(paths)}")
    rows = [audit(path) for path in paths]
    result = {"scope": "E3 G1a constant/plateau boundary control only",
              "n_subject_sessions": len(rows),
              "seams_total": sum(row["n_seams"] for row in rows),
              "constant_alarm_near_seam_total": sum(
                  row["constant_alarm_within_0p5s_of_seam"] for row in rows),
              "plateau_retrigger_total": sum(
                  row["plateau_retrigger_after_first_seam"] for row in rows),
              "g1b_splice_detectability_tested": False,
              "g1c_composed_attack_tested": False,
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
