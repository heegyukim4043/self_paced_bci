"""Null controls for selected LOSO arm-A sessions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from uncued_mi.riemannian import tangent_features
from scripts.evaluate_lee2019_loso_deployment import (
    SUBJECTS, fit_model, load_covariances, prepare_target, source_training,
    source_reference, tune_anchor,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z, ma_trace, alarms_from_ma

N_SHIFT = 10
N_PERM = 5


def budget_sensitivity(ma, base_threshold, allowed, evalmask, idle, times, onsets):
    exposure = idle.sum() * .1
    points = []
    for offset in np.linspace(-5, 5, 31):
        alarms = alarms_from_ma(ma, base_threshold + offset, allowed) & evalmask
        fp = float(np.sum(alarms & idle) / (exposure / 60))
        sensitivity = float(np.mean([np.any(alarms & (times >= onset) & (times <= onset + 2.)) for onset in onsets]))
        points.append((fp, sensitivity))
    admissible = [sen for fp, sen in points if fp <= 4.]
    return (float(max(admissible)) if admissible else np.nan,
            float(max(fp for fp, _ in points if fp <= 4.)) if admissible else np.nan)


def main():
    result_path = Path("data/processed/lee2019_loso_deployment.json")
    result = json.loads(result_path.read_text())
    rng = np.random.default_rng(20260813)
    rows = []
    for session in (1, 2):
        selected = result["summary"][str(session)]["A_target_rest"]["selected_subjects"]
        domains = load_covariances(session)
        own_cache = {}
        for subject in SUBJECTS:
            from scripts.run_lee2019_pseudo_uncued_riemannian import reference
            ref = reference(domains[subject]["pre"])
            own_cache[subject] = {
                "command": tangent_features(domains[subject]["command"], ref),
                "idle": tangent_features(domains[subject]["idle"], ref),
            }
        for subject in selected:
            anchor_cov, stream_cov, events, times, allowed = prepare_target(subject, session)
            ref = reference(domains[subject]["pre"])
            x_command, x_idle = source_training(domains, subject, "A_target_rest", source_reference(domains, subject), own_cache)
            scaler, model = fit_model(x_command, x_idle)
            anchor_features = tangent_features(anchor_cov, ref)
            stream_features = tangent_features(stream_cov, ref)
            anchor = model.decision_function(scaler.transform(anchor_features))
            raw = model.decision_function(scaler.transform(stream_features))
            threshold = tune_anchor(anchor)[2.]
            z = ewma_z(raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed)
            evalmask = times >= events[10, 2] / 250.
            onsets = events[10:, 5] / 250. + .5
            stops = events[10:, 5] / 250. + 4.
            state = np.zeros(len(times), bool)
            for onset, stop in zip(onsets, stops):
                state |= (times >= onset) & (times <= stop)
            idle = evalmask & allowed & ~state
            observed, observed_fp = budget_sensitivity(ma_trace(z), threshold, allowed, evalmask, idle, times, onsets)
            reversed_value, _ = budget_sensitivity(ma_trace(z[::-1]), threshold, allowed, evalmask, idle, times, onsets)
            valid_shifts = np.arange(int(20/.1), len(z)-int(20/.1))
            shifted = []
            for shift in rng.choice(valid_shifts, N_SHIFT, replace=False):
                value, _ = budget_sensitivity(ma_trace(np.roll(z, int(shift))), threshold, allowed, evalmask, idle, times, onsets)
                shifted.append(value)
            x = np.r_[x_command, x_idle]
            labels = np.r_[np.ones(len(x_command)), np.zeros(len(x_idle))]
            permuted = []
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            for _ in range(N_PERM):
                y = rng.permutation(labels)
                ss = StandardScaler().fit(x)
                mm = LogisticRegression(C=1., class_weight="balanced", solver="liblinear", max_iter=2000, random_state=20260813).fit(ss.transform(x), y)
                pa = mm.decision_function(ss.transform(anchor_features)); pr = mm.decision_function(ss.transform(stream_features))
                pth = tune_anchor(pa)[2.]; pz = ewma_z(pr, float(pa.mean()), float(pa.var(ddof=1)), allowed)
                value, _ = budget_sensitivity(ma_trace(pz), pth, allowed, evalmask, idle, times, onsets)
                permuted.append(value)
            rows.append({"session":session,"subject":subject,"observed":observed,"actual_fp":observed_fp,
                "time_reversed":reversed_value,"circular_shift_mean":float(np.mean(shifted)),
                "training_label_permutation_mean":float(np.mean(permuted))})
            print(f"session{session} sub-{subject:02d}", flush=True)
    summary={"n":len(rows)}
    for key in ("observed","actual_fp","time_reversed","circular_shift_mean","training_label_permutation_mean"):
        summary[key+"_median"]=float(np.median([row[key] for row in rows]))
    output=Path("data/processed/lee2019_loso_arm_a_nulls.json")
    output.write_text(json.dumps({"protocol":{"fp_budget_max":4,"selection":"maximum sensitivity among thresholds with actual FP<=4/min","latency":2,"shifts":N_SHIFT,"label_permutations":N_PERM},"summary":summary,"rows":rows},indent=2))
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
