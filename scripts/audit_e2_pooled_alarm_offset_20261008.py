"""Independent row-level audit of E2 source calibration and target FP distribution."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "data/processed/rate_curve_transfer_20261008"


def load_cases():
    cases = []
    for session in (1, 2):
        for subject in range(1, 55):
            path = DIRECTORY / f"session{session}" / f"sub-{subject:02d}.json"
            cases.append(json.loads(path.read_text(encoding="utf-8")))
    return cases


def summarize_session(cases):
    by_arm = {}
    for arm in ("K_A", "pooled_offset", "beta_slope", "K_0"):
        available = [case for case in cases if arm in case["evaluation"]]
        fp = np.array([case["evaluation"][arm]["future_fp_per_min"] for case in available])
        by_arm[arm] = {
            "available": len(available), "median_fp": float(np.median(fp)),
            "q90_fp": float(np.quantile(fp, .9)),
            "under_1p5": int(np.sum(fp < 1.5)),
            "in_band_1p5_to_2p5": int(np.sum((fp >= 1.5) & (fp <= 2.5))),
            "over_2p5": int(np.sum(fp > 2.5)),
            "over_4": int(np.sum(fp > 4)),
            "threshold_above_anchor_max": int(sum(
                case["evaluation"][arm]["threshold_above_anchor_ma_max"] for case in available)),
        }
    paired = {"both_admissible": 0, "K_A_only": 0,
              "pooled_only": 0, "neither": 0}
    for case in cases:
        a = bool(case["evaluation"]["K_A"]["admissible_fp2"])
        b = bool(case["evaluation"]["pooled_offset"]["admissible_fp2"])
        paired[("both_admissible" if a else "pooled_only") if b else
               ("K_A_only" if a else "neither")] += 1
    a_fp = np.array([case["evaluation"]["K_A"]["future_fp_per_min"] for case in cases])
    b_fp = np.array([case["evaluation"]["pooled_offset"]["future_fp_per_min"]
                     for case in cases])
    rng = np.random.default_rng(20261008 + cases[0]["session"])
    bootstrap_indices = rng.integers(0, len(cases), size=(2000, len(cases)))
    q90_delta = np.quantile(b_fp[bootstrap_indices], .9, axis=1) - np.quantile(
        a_fp[bootstrap_indices], .9, axis=1)
    sd_delta = np.std(b_fp[bootstrap_indices], axis=1) - np.std(
        a_fp[bootstrap_indices], axis=1)
    tail_delta = np.mean(b_fp[bootstrap_indices] > 4.0, axis=1) - np.mean(
        a_fp[bootstrap_indices] > 4.0, axis=1)
    offsets = np.array([case["source_pooled_offset"]["delta"] for case in cases])
    source_rates = np.array([case["source_pooled_offset"]["pooled_rate"] for case in cases])
    source_counts = np.array([case["source_pooled_offset"]["pooled_alarm_count"] for case in cases])
    target_counts = np.array([case["target_fp16"]["alarm_count"] for case in cases])
    target_episodes = np.array([case["target_fp16_episodes"]["merged_1s_episodes"]
                                for case in cases])
    target_multi_fraction = np.array([
        case["target_fp16_episodes"]["multi_alarm_episode_fraction"] for case in cases])
    donor_multi_fraction = np.array([
        episode["multi_alarm_episode_fraction"]
        for case in cases for episode in case["source_donor_fp16_episodes"]])
    donor_jackknife_spans = np.array([max(case["source_offset_leave_one_donor_out"])
                                      - min(case["source_offset_leave_one_donor_out"])
                                      for case in cases])
    nonmonotone = np.array([case["source_pooled_offset"]["nonmonotone_steps"]
                            for case in cases])
    return {
        "arms": by_arm, "paired_admissibility": paired,
        "paired_q90_fp_delta_ci95": [float(x) for x in np.quantile(q90_delta, (.025, .975))],
        "paired_between_subject_fp_sd_delta_ci95": [float(x) for x in np.quantile(sd_delta, (.025, .975))],
        "paired_above4_fraction_delta_ci95": [float(x) for x in np.quantile(tail_delta, (.025, .975))],
        "source_delta_median_range": [float(np.median(offsets)), float(offsets.min()),
                                      float(offsets.max())],
        "source_pooled_fp2_rate_median_range": [float(np.median(source_rates)),
                                                 float(source_rates.min()),
                                                 float(source_rates.max())],
        "source_pooled_alarm_count_median_range": [float(np.median(source_counts)),
                                                    int(source_counts.min()),
                                                    int(source_counts.max())],
        "target_fp16_alarm_count_unique": sorted(set(target_counts.tolist())),
        "target_fp16_alarms_exceed_merged_episodes": int(np.sum(target_counts > target_episodes)),
        "target_multi_alarm_episode_fraction_median": float(np.median(target_multi_fraction)),
        "source_donor_multi_alarm_episode_fraction_median": float(np.median(donor_multi_fraction)),
        "jackknife_delta_span_median_max": [float(np.median(donor_jackknife_spans)),
                                            float(donor_jackknife_spans.max())],
        "pooled_rate_nonmonotone_step_cases": int(np.sum(nonmonotone > 0)),
    }


def main():
    cases = load_cases()
    if len(cases) != 108 or any(case["source_donor_count"] != 53 for case in cases):
        raise AssertionError("cohort/source donor geometry changed")
    if any(not case["source_pooled_offset"]["valid"] for case in cases):
        raise AssertionError("primary pooled offset failed in at least one case")
    result = {"status": "E2 exploratory row-level audit; G1 failed",
              "sessions": {f"S{session}": summarize_session(
                  [case for case in cases if case["session"] == session])
                  for session in (1, 2)}}
    path = DIRECTORY / "row_level_audit.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
