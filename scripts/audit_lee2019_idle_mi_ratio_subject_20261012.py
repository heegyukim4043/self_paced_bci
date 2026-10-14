"""Subject-clustered re-audit of the archived Lee CNN-GRU ratio simulation.

This does not create a new replay or validate its onset null. Five compositions
are technical repeats of the same subject/session and are collapsed first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RATIOS = (0.75, 1.5, 3.0)
MODES = ("transferred", "common_fp")
FIELDS = ("fp", "raw_tpr_1s", "null_tpr_1s", "excess_1s")
SEED = 20261012
BOOT = 10000


def collapse(payload: dict) -> list[dict]:
    rows = payload["rows"]
    assert len(rows) == 40
    assert tuple(payload["protocol"]["evaluation_idle_mi_ratios"]) == RATIOS
    out = []
    for row in rows:
        subject, session = int(row["subject"]), int(row["session"])
        assert session in (1, 2)
        conditions = row["conditions"]
        assert len(conditions) == 15
        assert {(float(c["idle_mi_ratio"]), int(c["repeat"])) for c in conditions} == {
            (r, k) for r in RATIOS for k in range(5)
        }
        for repeat in range(5):
            paired = [next(c for c in conditions if c["repeat"] == repeat and
                           c["idle_mi_ratio"] == ratio) for ratio in RATIOS]
            assert len({c["composition_seed"] for c in paired}) == 1
            assert all(abs(c["mi_seconds"] - 100.0) < 1e-6 for c in paired)
            assert all(abs(c["idle_seconds"] / c["mi_seconds"] - ratio) < 1e-6
                       for ratio, c in zip(RATIOS, paired))
        for ratio in RATIOS:
            cases = sorted((c for c in conditions if c["idle_mi_ratio"] == ratio),
                           key=lambda c: c["repeat"])
            item = {"subject": subject, "session": session, "idle_mi_ratio": ratio,
                    "n_compositions": 5, "idle_seconds": float(cases[0]["idle_seconds"]),
                    "mi_seconds": 100.0, "modes": {}}
            for mode in MODES:
                values = {field: np.asarray([c["metrics"][mode][field] for c in cases], float)
                          for field in FIELDS}
                assert all(np.all(np.isfinite(v)) for v in values.values())
                item["modes"][mode] = {
                    "median_over_compositions": {field: float(np.median(v))
                                                 for field, v in values.items()},
                    "fp_min_across_compositions": float(np.min(values["fp"])),
                    "fp_max_across_compositions": float(np.max(values["fp"])),
                    "false_alarms_median": float(np.median(
                        values["fp"] * item["idle_seconds"] / 60.0)),
                }
            out.append(item)
    for session in (1, 2):
        expected = {r["subject"] for r in out if r["session"] == session and
                    r["idle_mi_ratio"] == RATIOS[0]}
        assert len(expected) == 20
        for ratio in RATIOS:
            assert {r["subject"] for r in out if r["session"] == session and
                    r["idle_mi_ratio"] == ratio} == expected
    assert {r["subject"] for r in out if r["session"] == 1} == {
        r["subject"] for r in out if r["session"] == 2
    }
    return out


def median_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, len(values), size=(BOOT, len(values)))
    return [float(v) for v in np.quantile(np.median(values[indices], axis=1), (.025, .975))]


def summarize(rows: list[dict]) -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for session in (1, 2):
        cases = {ratio: sorted((r for r in rows if r["session"] == session and
                                r["idle_mi_ratio"] == ratio), key=lambda r: r["subject"])
                 for ratio in RATIOS}
        entry = {"ratios": {}, "contrasts": {}, "robustness": {}}
        for ratio, group in cases.items():
            stat = {"n_subjects": len(group), "idle_seconds": group[0]["idle_seconds"],
                    "mi_seconds": 100.0, "modes": {}}
            for mode in MODES:
                metric = [r["modes"][mode]["median_over_compositions"] for r in group]
                fp = np.asarray([v["fp"] for v in metric])
                sen = np.asarray([v["raw_tpr_1s"] for v in metric])
                excess = np.asarray([v["excess_1s"] for v in metric])
                alarms = np.asarray([r["modes"][mode]["false_alarms_median"] for r in group])
                stat["modes"][mode] = {
                    "fp_median": float(np.median(fp)),
                    "fp_q10_q90": [float(v) for v in np.quantile(fp, (.1, .9))],
                    "fp_median_95ci_subject_bootstrap": median_ci(fp, rng),
                    "raw_sen1_median": float(np.median(sen)),
                    "raw_sen1_95ci_subject_bootstrap": median_ci(sen, rng),
                    "legacy_excess1_median": float(np.median(excess)),
                    "false_alarms_median": float(np.median(alarms)),
                    "n_subjects_fp_1p5_to_2p5": int(np.sum((fp >= 1.5) & (fp <= 2.5))),
                    "median_within_subject_fp_range_5_compositions": float(np.median([
                        r["modes"][mode]["fp_max_across_compositions"] -
                        r["modes"][mode]["fp_min_across_compositions"] for r in group])),
                }
            entry["ratios"][str(ratio)] = stat
        for lo, hi in ((0.75, 1.5), (1.5, 3.0), (0.75, 3.0)):
            delta = np.asarray([
                b["modes"]["transferred"]["median_over_compositions"]["fp"] -
                a["modes"]["transferred"]["median_over_compositions"]["fp"]
                for a, b in zip(cases[lo], cases[hi])
            ])
            entry["contrasts"][f"{hi}_minus_{lo}_transferred_fp"] = {
                "median_paired_delta_fp_per_min": float(np.median(delta)),
                "95ci_subject_bootstrap": median_ci(delta, rng),
                "n_increased": int(np.sum(delta > 0)), "n_decreased": int(np.sum(delta < 0)),
                "n_equal": int(np.sum(delta == 0)),
            }
        for mode in MODES:
            delta_sen = np.asarray([
                cases[3.0][i]["modes"][mode]["median_over_compositions"]["raw_tpr_1s"] -
                cases[0.75][i]["modes"][mode]["median_over_compositions"]["raw_tpr_1s"]
                for i in range(20)
            ])
            entry["contrasts"][f"3.0_minus_0.75_{mode}_raw_sen1"] = {
                "median_paired_delta": float(np.median(delta_sen)),
                "95ci_subject_bootstrap": median_ci(delta_sen, rng),
                "n_increased": int(np.sum(delta_sen > 0)),
                "n_decreased": int(np.sum(delta_sen < 0)),
                "n_equal": int(np.sum(delta_sen == 0)),
            }
        ranges = np.asarray([
            max(cases[q][i]["modes"]["transferred"]["median_over_compositions"]["fp"]
                for q in RATIOS) -
            min(cases[q][i]["modes"]["transferred"]["median_over_compositions"]["fp"]
                for q in RATIOS) for i in range(20)
        ])
        all_band = sum(all(1.5 <= cases[q][i]["modes"]["transferred"]
                           ["median_over_compositions"]["fp"] <= 2.5 for q in RATIOS)
                       for i in range(20))
        entry["robustness"] = {
            "median_within_subject_fp_range_across_ratios": float(np.median(ranges)),
            "n_subjects_all_three_ratios_fp_1p5_to_2p5": int(all_band),
        }
        out[str(session)] = entry
    return out


def plot(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    colors = {1: "#2563eb", 2: "#d97706"}
    for session in (1, 2):
        subjects = sorted({r["subject"] for r in rows if r["session"] == session})
        for axis, field in zip(axes, ("fp", "raw_tpr_1s")):
            matrix = np.asarray([[next(r for r in rows if r["session"] == session and
                                      r["subject"] == subject and r["idle_mi_ratio"] == ratio)
                                  ["modes"]["transferred"]["median_over_compositions"][field]
                                  for ratio in RATIOS] for subject in subjects])
            axis.plot(RATIOS, np.median(matrix, axis=0), color=colors[session],
                      marker="o", lw=2, label=f"S{session} median")
            q10, q90 = np.quantile(matrix, (.1, .9), axis=0)
            axis.fill_between(RATIOS, q10, q90, color=colors[session], alpha=.10)
    axes[0].axhline(2, color="black", ls="--", lw=1, label="2 FP/min")
    axes[0].set_ylabel("Achieved FP/min")
    axes[1].set_ylabel("Raw SEN@1 s (legacy replay)")
    for axis in axes:
        axis.set_xticks(RATIOS, ["0.75:1", "1.5:1", "3:1"])
        axis.set_xlabel("Simulated evaluation idle/MI duration")
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Archived CNN-GRU ratio replay: subject-level medians (n=20; 5 compositions)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(
        "data/processed/lee2019_cnn_gru_eval_idle_mi_ratio_subset20.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/lee2019_idle_mi_ratio_subject_audit_20261012.json"))
    parser.add_argument("--figure", type=Path, default=Path(
        "data/processed/lee2019_idle_mi_ratio_subject_audit_20261012.png"))
    args = parser.parse_args()
    if args.output.exists() or args.figure.exists():
        raise FileExistsError("audit outputs already exist")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = collapse(payload)
    result = {
        "scope": "post-hoc subject-clustered re-audit of archived subset20 simulation",
        "not_test_phase_replication": True,
        "not_validated_onset_null": True,
        "n_subjects": 20,
        "n_sessions": 2,
        "n_compositions_per_ratio": 5,
        "ratios": RATIOS,
        "method": "median over five compositions per subject/session/ratio; subject bootstrap",
        "summary": summarize(rows),
        "subject_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    plot(rows, args.figure)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
