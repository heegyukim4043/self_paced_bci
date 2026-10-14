"""Post-hoc, assumption-dependent count-noise reference for E4b FP4.

The gamma rate multiplier is a sensitivity model, NOT an identified lower
bound for the actual dwell/refractory alarm process or threshold estimator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEED = 20261010
N_COHORTS = 4000
TARGET_FP = 4.0
ARMS = {"T56": 4.0, "T_all": 24.0}


def summarize(sd: np.ndarray, observed: float) -> dict:
    return {
        "median_sd": float(np.median(sd)),
        "central_95_percent_range": [float(v) for v in np.quantile(sd, (.025, .975))],
        "fraction_simulated_sd_at_least_observed": float(np.mean(sd >= observed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("data/processed/e4b_task_idle_fp_20261010.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/e4b_counting_reference_20261010.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = {
        "status": "post-hoc simulation, not an empirical lower bound or preregistered endpoint",
        "seed": SEED, "n_cohorts": N_COHORTS, "n_subjects_per_session": 54,
        "target_fp_per_min": TARGET_FP,
        "metric": "sample SD across subjects of log((N+0.5)/(T_min*4))",
        "models": {
            "evaluation_only": "N_i ~ Poisson(4*T_i)",
            "gamma_multiplier": "U_i ~ Gamma(shape=k, scale=1/k), N_i ~ Poisson(4*T_i*U_i); k=4/24 is illustrative, not fitted",
        },
        "sessions": {},
    }
    for session in (1, 2):
        session_result = {}
        for arm, k in ARMS.items():
            rows = sorted((r for r in data["rows"] if r["session"] == session and
                           r["target_fp"] == TARGET_FP and r["arm"] == arm),
                          key=lambda r: r["subject"])
            if len(rows) != 54 or len({r["subject"] for r in rows}) != 54:
                raise AssertionError("not 54 distinct subject rows")
            seconds = np.asarray([r["evaluation"]["E_full"]["seconds"] for r in rows])
            counts = np.asarray([r["evaluation"]["E_full"]["count"] for r in rows])
            mu = TARGET_FP * seconds / 60.
            observed = float(np.std(np.log((counts + .5) / mu), ddof=1))
            rng_eval = np.random.default_rng(np.random.SeedSequence([SEED, session, int(k), 0]))
            n_eval = rng_eval.poisson(mu, size=(N_COHORTS, len(mu)))
            sd_eval = np.std(np.log((n_eval + .5) / mu), axis=1, ddof=1)
            rng_both = np.random.default_rng(np.random.SeedSequence([SEED, session, int(k), 1]))
            multiplier = rng_both.gamma(shape=k, scale=1. / k,
                                         size=(N_COHORTS, len(mu)))
            n_both = rng_both.poisson(mu * multiplier)
            sd_both = np.std(np.log((n_both + .5) / mu), axis=1, ddof=1)
            session_result[arm] = {
                "observed_sd": observed,
                "median_evaluation_seconds": float(np.median(seconds)),
                "k_assumed": k,
                "evaluation_only": summarize(sd_eval, observed),
                "gamma_multiplier_plus_evaluation": summarize(sd_both, observed),
            }
        result["sessions"][str(session)] = session_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["sessions"], indent=2))


if __name__ == "__main__":
    main()
