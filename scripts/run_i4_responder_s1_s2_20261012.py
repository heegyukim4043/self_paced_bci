"""Post-hoc, fixed-rule S1 to S2 responder audit on frozen I4 rows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_SHA256 = "668CDC8EA6DC35E00EE3EE8C367A5292901013313A7C29913A90EE7A64C0BC8E"
SEED = 20261012


def describe(values):
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def run(source):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest().upper()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Input SHA-256 changed: {digest}")
    data = json.loads(raw)
    rows = data["rows"]
    if len(rows) != 864:
        raise ValueError(f"Expected 864 rows, got {len(rows)}")
    keyed = {}
    for row in rows:
        key = (row["subject"], row["session"], row["arm"], row["target_fp"])
        if key in keyed:
            raise ValueError(f"Duplicate row {key}")
        if abs(row["mi"]["excess_2s"] - (row["mi"]["sen_2s"] - row["mi"]["null_2s"])) > 1e-10:
            raise ValueError(f"MI excess mismatch {key}")
        if abs(row["fixation"]["excess_2s"] - (row["fixation"]["sen_2s"] - row["fixation"]["null_2s"])) > 1e-10:
            raise ValueError(f"Fixation excess mismatch {key}")
        keyed[key] = row
    subjects = sorted({r["subject"] for r in rows})
    if len(subjects) != 54 or any(
        (s, session, arm, float(fp)) not in keyed
        for s in subjects
        for session in (1, 2)
        for arm in ("T_all", "future_idle_oracle")
        for fp in (1, 2, 4, 8)
    ):
        raise ValueError("Incomplete 54-subject, 2-session, 2-arm, 4-target matrix")

    def get(s, session, arm, fp=2):
        return keyed[(s, session, arm, float(fp))]

    selected = [s for s in subjects if get(s, 1, "T_all")["mi"]["excess_2s"] >= 0.15]
    if len(selected) != 10:
        raise ValueError(f"Selection changed: {len(selected)} subjects")
    selected_set = set(selected)
    per_subject = []
    for s in subjects:
        record = {"subject": s, "selected": s in selected_set}
        for name, sess, arm, fp in (
            ("s1_tall_fp2", 1, "T_all", 2),
            ("s2_oracle_fp2", 2, "future_idle_oracle", 2),
            ("s2_tall_fp2", 2, "T_all", 2),
            ("s2_oracle_fp4", 2, "future_idle_oracle", 4),
        ):
            r = get(s, sess, arm, fp)
            record[name] = {
                "fp_per_min": r["future_idle"]["fp_per_min"],
                "idle_alarm_count": r["future_idle"]["count"],
                "idle_seconds": r["future_idle"]["seconds"],
                "sen_2s": r["mi"]["sen_2s"],
                "null_2s": r["mi"]["null_2s"],
                "excess_2s": r["mi"]["excess_2s"],
                "fixation_sen_2s": r["fixation"]["sen_2s"],
                "fixation_null_2s": r["fixation"]["null_2s"],
                "fixation_excess_2s": r["fixation"]["excess_2s"],
                "mi_minus_fixation_excess": r["mi"]["excess_2s"] - r["fixation"]["excess_2s"],
            }
        per_subject.append(record)

    groups = {}
    for label, members in (("selected", [p for p in per_subject if p["selected"]]),
                           ("comparison", [p for p in per_subject if not p["selected"]])):
        groups[label] = {}
        for arm in ("s1_tall_fp2", "s2_oracle_fp2", "s2_tall_fp2", "s2_oracle_fp4"):
            groups[label][arm] = {
                field: describe([p[arm][field] for p in members])
                for field in ("fp_per_min", "idle_alarm_count", "sen_2s", "null_2s", "excess_2s",
                              "fixation_excess_2s", "mi_minus_fixation_excess")
            }
        groups[label]["s2_oracle_fp2_sen_ge_0_7"] = sum(p["s2_oracle_fp2"]["sen_2s"] >= 0.7 for p in members)
        admissible = [p for p in members if 1.5 <= p["s2_tall_fp2"]["fp_per_min"] <= 2.5]
        groups[label]["s2_tall_fp2_admissible_n"] = len(admissible)
        groups[label]["s2_tall_fp2_admissible_sen"] = describe([p["s2_tall_fp2"]["sen_2s"] for p in admissible]) if admissible else None
        groups[label]["s2_tall_fp2_admissible_and_sen_ge_0_7"] = sum(p["s2_tall_fp2"]["sen_2s"] >= 0.7 for p in admissible)

    x = np.asarray([p["s2_oracle_fp2"]["excess_2s"] for p in per_subject if p["selected"]])
    y = np.asarray([p["s2_oracle_fp2"]["excess_2s"] for p in per_subject if not p["selected"]])
    observed = float(x.mean() - y.mean())
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(10000)
    for i in range(10000):
        bootstrap[i] = rng.choice(x, len(x), replace=True).mean() - rng.choice(y, len(y), replace=True).mean()
    pooled = np.concatenate((x, y))
    extreme = 0
    for _ in range(20000):
        perm = rng.permutation(pooled)
        diff = perm[:len(x)].mean() - perm[len(x):].mean()
        extreme += abs(diff) >= abs(observed) - 1e-15
    rho_oracle = spearmanr(
        [p["s1_tall_fp2"]["excess_2s"] for p in per_subject],
        [p["s2_oracle_fp2"]["excess_2s"] for p in per_subject],
    )
    rho_tall = spearmanr(
        [p["s1_tall_fp2"]["excess_2s"] for p in per_subject],
        [p["s2_tall_fp2"]["excess_2s"] for p in per_subject],
    )
    return {
        "status": "post_hoc_internal_reproducibility_diagnostic_not_preregistered",
        "input": str(source), "input_sha256": digest, "seed": SEED,
        "selection": {"rule": "S1 T_all FP2 MI excess >= 0.15", "subject_ids": selected,
                      "n_selected": len(selected), "n_comparison": len(y)},
        "primary": {"s2_oracle_fp2_mean_excess_difference": observed,
                    "bootstrap_percentile_95_ci": [float(v) for v in np.quantile(bootstrap, [0.025, 0.975])],
                    "permutation_two_sided_p": (extreme + 1) / 20001,
                    "bootstrap_draws": 10000, "permutation_draws": 20000},
        "spearman": {"s1_tall_vs_s2_oracle_fp2": {"rho": float(rho_oracle.statistic), "p": float(rho_oracle.pvalue)},
                     "s1_tall_vs_s2_tall_fp2": {"rho": float(rho_tall.statistic), "p": float(rho_tall.pvalue)}},
        "groups": groups,
        "per_subject": per_subject,
    }


def plot_subjects(result, destination):
    rows = result["per_subject"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    rng = np.random.default_rng(SEED)
    for selected, label, color, xpos in ((False, "Other (n=44)", "#808080", 0),
                                          (True, "S1-selected (n=10)", "#ce4b37", 1)):
        members = [r for r in rows if r["selected"] == selected]
        for ax, field, ylabel in ((axes[0], "excess_2s", "S2 oracle FP2 excess SEN@2s"),
                                  (axes[1], "sen_2s", "S2 oracle FP2 SEN@2s")):
            y = [r["s2_oracle_fp2"][field] for r in members]
            jitter = rng.uniform(-0.15, 0.15, len(y))
            ax.scatter(xpos + jitter, y, s=24, color=color, alpha=0.72, edgecolors="none")
            ax.plot([xpos - 0.20, xpos + 0.20], [np.median(y)] * 2, color="black", lw=2)
            ax.set_ylabel(ylabel)
    for ax in axes:
        ax.set_xticks([0, 1], ["Other\n(n=44)", "S1-selected\n(n=10)"])
        ax.grid(axis="y", alpha=0.2)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[1].axhline(0.7, color="#ca3333", linestyle="--", lw=1, label="provisional SEN 0.7")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Lee2019 S1-selected responders in S2 (post-hoc; oracle FP2)", fontsize=12)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/i4_low_fp_original_continuous_20261010.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/i4_s1_s2_responder_20261012.json"))
    parser.add_argument("--figure", type=Path, default=Path("data/processed/i4_s1_s2_responder_20261012.png"))
    args = parser.parse_args()
    result = run(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_subjects(result, args.figure)
    print(json.dumps({k: result[k] for k in ("selection", "primary", "spearman", "groups")}, indent=2))


if __name__ == "__main__":
    main()
