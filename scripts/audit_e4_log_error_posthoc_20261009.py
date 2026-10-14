"""Post-hoc, descriptive E4 log-error and two-sided-tail audit.

Does not revise the prospectively fixed E4 success rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import poisson


INPUT = Path("data/processed/e4_original_continuous_fp_20261009_v2.json")
OUTPUT = Path("data/processed/e4_log_error_posthoc_20261009.json")
SEED = 20261009
BOOT = 2000


def describe(rows: list[dict], target: float) -> dict:
    rows = sorted(rows, key=lambda r: r["subject"])
    if len(rows) != 54 or len({r["subject"] for r in rows}) != 54:
        raise AssertionError("expected 54 unique subjects")
    count = np.asarray([r["future"]["count"] for r in rows], dtype=float)
    minutes = np.asarray([r["future"]["seconds"] / 60. for r in rows])
    fp = count / minutes
    log_error = np.log((count + .5) / (minutes * target))
    # The direct log error matches the user-proposed definition; unlike the
    # original E4 summary, its denominator has no +0.5 pseudo-count.
    lower, upper = target / 2, target * 2
    band = (3., 5.) if target == 4. else (1.5, 2.5)
    expected = minutes * target
    min_count = np.ceil(band[0] * minutes - 1e-12).astype(int)
    max_count = np.floor(band[1] * minutes + 1e-12).astype(int)
    reference = poisson.cdf(max_count, expected) - poisson.cdf(min_count - 1, expected)
    return {
        "median_log_error": float(np.median(log_error)),
        "sd_log_error_ddof1": float(np.std(log_error, ddof=1)),
        "iqr_log_error": float(np.quantile(log_error, .75) - np.quantile(log_error, .25)),
        "under_half_target_n": int(np.sum(fp < lower)),
        "over_double_target_n": int(np.sum(fp > upper)),
        "admissible_n": int(np.sum((fp >= band[0]) & (fp <= band[1]))),
        "mean_poisson_admissible_reference": float(np.mean(reference)),
        "median_future_alarm_count": float(np.median(count)),
        "median_future_seconds": float(np.median(minutes * 60.)),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = source["rows"]
    result: dict = {
        "status": "post-hoc descriptive audit; does not replace fixed E4 decision criteria",
        "log_error": "log((future_alarm_count+0.5)/(future_idle_minutes*target_FP_per_min))",
        "spread_sd": "sample SD across 54 subjects (ddof=1)",
        "poisson_reference": "P(admissible integer count | independent Poisson rate=target and each subject's actual exposure); illustrative, not ceiling",
        "summary": {},
    }
    for session in (1, 2):
        result["summary"][str(session)] = {}
        for mode in ("all_idle", "late_only"):
            result["summary"][str(session)][mode] = {}
            for target in (4., 2.):
                subset = [r for r in rows if r["session"] == session and
                          r["mode"] == mode and r["target_fp"] == target]
                arms = {}
                for arm in ("A", "T_far", "T_near"):
                    arms[arm] = describe([r for r in subset if r["arm"] == arm], target)
                result["summary"][str(session)][mode][str(target)] = arms
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
