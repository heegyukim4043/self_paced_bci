from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


SOURCE_N = {
    "source_5": 5,
    "source_10": 10,
    "source_20": 20,
    "source_40": 40,
    "source_all_53": 53,
}


def fit_random_intercept(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> dict:
    unique_groups = np.unique(groups)

    def solve(log_variances: np.ndarray):
        subject_var, residual_var = np.exp(log_variances)
        xt_vinv_x = np.zeros((x.shape[1], x.shape[1]))
        xt_vinv_y = np.zeros(x.shape[1])
        logdet = 0.0
        quadratic_parts = []
        blocks = []
        for group in unique_groups:
            index = np.flatnonzero(groups == group)
            xg, yg = x[index], y[index]
            n = len(index)
            covariance = residual_var * np.eye(n) + subject_var * np.ones((n, n))
            inverse = np.linalg.inv(covariance)
            sign, block_logdet = np.linalg.slogdet(covariance)
            logdet += block_logdet
            xt_vinv_x += xg.T @ inverse @ xg
            xt_vinv_y += xg.T @ inverse @ yg
            blocks.append((xg, yg, inverse))
        beta = np.linalg.solve(xt_vinv_x, xt_vinv_y)
        for xg, yg, inverse in blocks:
            residual = yg - xg @ beta
            quadratic_parts.append(float(residual @ inverse @ residual))
        nll = 0.5 * (
            len(y) * np.log(2 * np.pi) + logdet + sum(quadratic_parts)
        )
        return nll, beta, np.linalg.inv(xt_vinv_x)

    initial = np.log([max(np.var(y) * 0.5, 1e-5), max(np.var(y) * 0.5, 1e-5)])
    result = minimize(lambda value: solve(value)[0], initial, method="L-BFGS-B")
    _, beta, covariance = solve(result.x)
    standard_error = np.sqrt(np.diag(covariance))
    z_value = beta / standard_error
    p_value = 2 * norm.sf(np.abs(z_value))
    return {
        "converged": bool(result.success),
        "coefficients": [
            {
                "name": name,
                "estimate": float(estimate),
                "standard_error": float(se),
                "ci_95": [float(estimate - 1.96 * se), float(estimate + 1.96 * se)],
                "z": float(z),
                "p": float(p),
            }
            for name, estimate, se, z, p in zip(
                ("intercept_source_n20", "is_target", "log_source_n_centered"),
                beta,
                standard_error,
                z_value,
                p_value,
            )
        ],
        "subject_random_intercept_variance": float(np.exp(result.x[0])),
        "residual_variance": float(np.exp(result.x[1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/matched_lda_scaling_subset20.json"),
    )
    parser.add_argument("--target-fp", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/matched_lda_mixed_model.json"),
    )
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))["rows"]
    selected = [
        row
        for row in rows
        if float(row["target_fp_axis"]) == args.target_fp
        and (row["arm"] == "target_50" or row["arm"] in SOURCE_N)
    ]
    y = np.asarray([row["excess_tpr"] for row in selected])
    is_target = np.asarray([row["arm"] == "target_50" for row in selected], dtype=float)
    log_n = np.asarray(
        [
            0.0
            if row["arm"] == "target_50"
            else np.log(SOURCE_N[row["arm"]]) - np.log(20)
            for row in selected
        ]
    )
    x = np.c_[np.ones(len(selected)), is_target, log_n]
    groups = np.asarray([row["target_subject"] for row in selected])
    model = fit_random_intercept(y, x, groups)
    payload = {
        "target_fp_axis": args.target_fp,
        "rows": len(selected),
        "subjects": int(len(np.unique(groups))),
        "interpretation": {
            "is_target": "Target-trained minus source-trained at the centered 20-source-subject reference.",
            "log_source_n_centered": "Change in excess TPR per one-unit increase in log(source subjects), among source arms.",
        },
        "model": model,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
