"""Pilot a Riemannian-idle gated alarm-rate controller for frozen CNN-GRU scores.

The CNN-GRU remains the command score.  A target-pre-rest Riemannian
geodesic gate only declares samples eligible for threshold calibration.
Candidate thresholds are replayed through the complete MA/dwell/refractory
chain, avoiding a window-exceedance-to-alarm-rate approximation.

This is a Lee2019 cue-relative pseudo-uncued simulation, not a validation on
truly self-initiated MI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.decision import moving_average_dwell
from uncued_mi.riemannian import tangent_features
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS,
    HALF_LIFE_SECONDS,
    MA_SECONDS,
    REFRACTORY_SECONDS,
    STEP,
    normalize_anchor,
    tune_ma_dwell,
)
from scripts.evaluate_lee2019_loso_deployment import (
    SUBJECTS,
    load_covariances,
    reference,
)
from scripts.run_lee2019_cnn_gru_oracle_alarm_controller import (
    interval_mask,
    metrics,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import (
    SUBSET20,
    load_cache,
)
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration
from scripts.run_lee2019_stateful_gru import build_no_guard_stream

TARGET_FP = 2.0
PRIMARY_EPSILON = 0.10


def geodesic_score(covariances, idle_reference):
    """Affine-invariant distance represented by the tangent-vector norm."""
    return np.linalg.norm(tangent_features(covariances, idle_reference), axis=1)


def source_central_idle_gate(
    domains,
    target,
    epsilon=PRIMARY_EPSILON,
):
    """Choose a central-rest distance cutoff on source subjects only.

    Geodesic distances are normalized by each source's pre-rest.  The cutoff
    maximizes source pre-rest exposure while no more than ``epsilon`` of
    source command windows are incorrectly declared central/idle.  Unlike a
    CNN-score lower-tail selector, this gate is independent of the command
    score being controlled.
    """
    pre_z, command_z = [], []
    for source in SUBJECTS:
        if source == target:
            continue
        own_ref = reference(domains[source]["pre"])
        pre = geodesic_score(domains[source]["pre"], own_ref)
        location = float(np.mean(pre))
        scale_value = max(float(np.std(pre, ddof=1)), 1e-8)
        pre_z.append((pre - location) / scale_value)
        cmd = geodesic_score(domains[source]["command"], own_ref)
        command_z.append((cmd - location) / scale_value)
    idle = np.concatenate(pre_z)
    command = np.concatenate(command_z)
    candidates = np.unique(np.quantile(idle, np.linspace(.01, .995, 240)))
    chosen = float(candidates[0])
    chosen_idle_coverage = 0.0
    chosen_command_coverage = 1.0
    for upper in candidates:
        idle_rate = float(np.mean(idle <= upper))
        command_rate = float(np.mean(command <= upper))
        if command_rate <= epsilon and idle_rate >= chosen_idle_coverage:
            chosen = float(upper)
            chosen_idle_coverage = idle_rate
            chosen_command_coverage = command_rate
    return {
        "upper": chosen,
        "source_idle_coverage": chosen_idle_coverage,
        "source_command_misidentified_as_idle": chosen_command_coverage,
        "epsilon": float(epsilon),
    }


def target_riemannian_stream(subject, session, domains, cnn_sequence):
    gate = source_central_idle_gate(domains, subject)
    path = Path(f"data/interim/lee2019_four_rest/session{session}/sub-{subject:02d}.npz")
    with np.load(path, allow_pickle=False) as data:
        own_ref = reference(data["pre_train_covariance"])
        pre_scores = geodesic_score(data["pre_train_covariance"], own_ref)
        location = float(np.mean(pre_scores))
        scale_value = max(float(np.std(pre_scores, ddof=1)), 1e-8)

        command_scores = (
            geodesic_score(data["train_command_covariance"], own_ref) - location
        ) / scale_value
        trials = np.asarray(data["train_command_trial"], int)
        # CNN command windows use [onset+1 s, stop) and therefore contain
        # exactly 20 samples.  The covariance cache includes the right edge
        # for two floating-point-aligned trials, so trim that endpoint here.
        command = [command_scores[trials == trial][:20] for trial in range(50, 100)]

        rests = []
        for name in ("pre_train", "post_train", "pre_test", "post_test"):
            values = (
                geodesic_score(data[f"{name}_covariance"], own_ref) - location
            ) / scale_value
            # CNN rest_windows splits by physical time (>30 s), not by the
            # midpoint of the post-transient 571-window array.  This is the
            # final 300 windows of each 60 s rest block.
            rests.append(values[-300:])

    values, _, _, template = build_no_guard_stream(
        rests, command, 20260728 + 100 * session + subject
    )
    values = np.asarray(values, float).reshape(-1)
    if not (
        len(values) == len(cnn_sequence.times)
        and np.allclose(template.times, cnn_sequence.times)
        and np.allclose(template.idle_intervals, cnn_sequence.idle_intervals)
        and np.allclose(template.command_intervals, cnn_sequence.command_intervals)
    ):
        raise RuntimeError(
            f"CNN/Riemannian replay misalignment for S{session} sub-{subject:02d}: "
            f"lengths={len(values)}/{len(cnn_sequence.times)}, "
            f"templates={len(template.command_intervals)}/{len(cnn_sequence.command_intervals)}, "
            f"ends={template.times[-1] if len(template.times) else None}/"
            f"{cnn_sequence.times[-1] if len(cnn_sequence.times) else None}"
        )
    eligible = values <= gate["upper"]
    gate.update({
        "target_pre_mean": location,
        "target_pre_sd": scale_value,
        "target_eligible_fraction": float(np.mean(eligible)),
    })
    return values, eligible, gate


def fixed_replay_alarms(scores, threshold):
    return moving_average_dwell(
        np.asarray(scores, float),
        float(threshold),
        STEP,
        MA_SECONDS,
        DWELL_SECONDS,
        REFRACTORY_SECONDS,
    ).alarms


def replay_rate(scores, eligible, threshold):
    exposure = float(np.sum(eligible) * STEP)
    if exposure <= 0:
        return np.nan
    alarms = fixed_replay_alarms(scores, threshold)
    return float(np.sum(alarms & eligible) / (exposure / 60.0))


def replay_controller(
    scores,
    eligible,
    initial_threshold,
    target_fp=TARGET_FP,
    horizon_seconds=120.0,
    minimum_eligible_seconds=30.0,
    update_seconds=5.0,
    rho=0.10,
    grid_radius=4.0,
    grid_size=121,
):
    """Causal threshold control using fixed-threshold replay of past data."""
    scores = np.asarray(scores, float)
    eligible = np.asarray(eligible, bool)
    n = len(scores)
    horizon = max(1, int(round(horizon_seconds / STEP)))
    update = max(1, int(round(update_seconds / STEP)))
    minimum = max(1, int(round(minimum_eligible_seconds / STEP)))
    ma_steps = max(1, int(np.ceil(MA_SECONDS / STEP)))
    dwell_steps = max(1, int(np.ceil(DWELL_SECONDS / STEP)))
    refractory_steps = max(0, int(np.ceil(REFRACTORY_SECONDS / STEP)))

    alarms = np.zeros(n, bool)
    thresholds = np.full(n, float(initial_threshold))
    ma = np.full(n, np.nan)
    tau = float(initial_threshold)
    history = []
    active_count = 0
    blocked_until = -1
    updates = []

    for index, value in enumerate(scores):
        if index > 0 and index % update == 0:
            start = max(0, index - horizon)
            past_eligible = eligible[start:index]
            exposure_samples = int(np.sum(past_eligible))
            if exposure_samples >= minimum:
                grid = np.linspace(
                    initial_threshold - grid_radius,
                    initial_threshold + grid_radius,
                    grid_size,
                )
                rates = np.asarray([
                    replay_rate(scores[start:index], past_eligible, candidate)
                    for candidate in grid
                ])
                error = np.abs(np.log((rates + 0.05) / (target_fp + 0.05)))
                best = int(np.nanargmin(error))
                tau_hat = float(grid[best])
                tau = float((1.0 - rho) * tau + rho * tau_hat)
                updates.append({
                    "time": float(index * STEP),
                    "eligible_seconds": float(exposure_samples * STEP),
                    "tau_hat": tau_hat,
                    "tau": tau,
                    "replay_rate": float(rates[best]),
                })
        thresholds[index] = tau
        history.append(float(value))
        if len(history) > ma_steps:
            history.pop(0)
        if len(history) == ma_steps:
            ma[index] = float(np.mean(history))
        active = len(history) == ma_steps and ma[index] >= tau
        active_count = active_count + 1 if active else 0
        if active_count == dwell_steps and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
    return alarms, thresholds, ma, updates


def controller_diagnostics(sequence, alarms, thresholds, eligible, updates):
    true_idle = interval_mask(sequence.times, sequence.idle_intervals)
    command = interval_mask(sequence.times, sequence.command_intervals)
    eligible_exposure = float(np.sum(eligible) * STEP)
    proxy_rate = (
        float(np.sum(alarms & eligible) / (eligible_exposure / 60.0))
        if eligible_exposure > 0 else None
    )
    return {
        "operating_point_log_error": float(abs(np.log((metrics(sequence, alarms, np.array([1]))["fp_per_min"] + .05) / (TARGET_FP + .05)))),
        "eligibility_wall_fraction": float(np.mean(eligible)),
        "eligibility_idle_coverage": float(np.sum(eligible & true_idle) / max(np.sum(true_idle), 1)),
        "eligibility_command_coverage": float(np.sum(eligible & command) / max(np.sum(command), 1)),
        "eligibility_offline_purity": float(np.sum(eligible & true_idle) / max(np.sum(eligible), 1)),
        "proxy_alarm_rate_per_min": proxy_rate,
        "threshold_initial": float(thresholds[0]),
        "threshold_final": float(thresholds[-1]),
        "threshold_median": float(np.median(thresholds)),
        "threshold_min": float(np.min(thresholds)),
        "threshold_max": float(np.max(thresholds)),
        "controller_updates": len(updates),
        "update_trace": updates,
    }


def analyze(path, subject, session, domains, shifts, args):
    sequence, blocks = load_cache(path)
    mean, variance, normalized = normalize_anchor(blocks)
    point = tune_ma_dwell([calibration(values) for values in normalized], 160)
    ewma = ewma_normalize_and_decide(
        sequence.scores,
        initial_mean=mean,
        initial_variance=variance,
        threshold=point.threshold,
        step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS,
        update_mode="all",
        ma_seconds=MA_SECONDS,
        dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS,
        sigma_floor_fraction=.25,
    )
    riemannian, riemannian_eligible, gate = target_riemannian_stream(
        subject, session, domains, sequence
    )
    true_idle = interval_mask(sequence.times, sequence.idle_intervals)
    time_only = np.ones(len(sequence.times), bool)
    rng = np.random.default_rng(20260909 + session * 1000 + subject)
    offsets = rng.integers(1, len(sequence.times), shifts)

    arms = {"A_ewma_baseline": metrics(sequence, ewma.alarms, offsets)}
    eligibility = {
        "B_oracle_idle": true_idle,
        "C_riemannian_idle": riemannian_eligible,
        "D_time_only": time_only,
    }
    for name, eligible in eligibility.items():
        alarms, thresholds, _, updates = replay_controller(
            ewma.statistic,
            eligible,
            point.threshold,
            horizon_seconds=args.horizon,
            minimum_eligible_seconds=args.minimum_eligible,
            update_seconds=args.update,
            rho=args.rho,
        )
        result = metrics(sequence, alarms, offsets)
        result.update(controller_diagnostics(sequence, alarms, thresholds, eligible, updates))
        arms[name] = result
    return {
        "subject": subject,
        "session": session,
        "baseline_stratum": args.stratum,
        "cnn_anchor_threshold": float(point.threshold),
        "riemannian_gate": gate,
        "riemannian_stream_summary": {
            "mean": float(np.mean(riemannian)),
            "sd": float(np.std(riemannian, ddof=1)),
        },
        "arms": arms,
    }


def bootstrap_median(values, seed, repeats=20000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def summarize(rows):
    result = {}
    arm_names = list(rows[0]["arms"])
    for arm_index, arm in enumerate(arm_names):
        points = [row["arms"][arm] for row in rows]
        item = {}
        for key in ("fp_per_min", "observed_sen1", "null_sen1", "excess_sen1"):
            values = [point[key] for point in points]
            item[f"{key}_median"] = float(np.median(values))
            item[f"{key}_ci95"] = bootstrap_median(
                values, 202609090 + arm_index * 10 + len(key)
            )
        errors = [abs(np.log((p["fp_per_min"] + .05) / (TARGET_FP + .05))) for p in points]
        item["operating_point_log_error_median"] = float(np.median(errors))
        item["operating_point_log_error_ci95"] = bootstrap_median(
            errors, 202609190 + arm_index
        )
        item["admissible_n"] = int(sum(1.5 <= p["fp_per_min"] <= 2.5 for p in points))
        if arm != "A_ewma_baseline":
            baseline = [row["arms"]["A_ewma_baseline"] for row in rows]
            delta_error = [
                abs(np.log((p["fp_per_min"] + .05) / (TARGET_FP + .05)))
                - abs(np.log((b["fp_per_min"] + .05) / (TARGET_FP + .05)))
                for p, b in zip(points, baseline)
            ]
            delta_excess = [p["excess_sen1"] - b["excess_sen1"] for p, b in zip(points, baseline)]
            item["paired_delta_log_error_median"] = float(np.median(delta_error))
            item["paired_delta_log_error_ci95"] = bootstrap_median(delta_error, 202609290 + arm_index)
            item["paired_delta_excess_median"] = float(np.median(delta_excess))
            item["paired_delta_excess_ci95"] = bootstrap_median(delta_excess, 202609390 + arm_index)
        result[arm] = item
    return result


def select_stratified_pilot(conditioning_path, seed=20260909):
    payload = json.loads(Path(conditioning_path).read_text(encoding="utf-8"))
    pools = {"under": [], "near": [], "over": []}
    for row in payload["rows"]:
        fp = float(row["deployed_fp_per_min"])
        key = "under" if fp < 1.0 else "near" if 1.5 <= fp <= 2.5 else "over" if fp > 3.0 else None
        if key:
            pools[key].append((int(row["session"]), int(row["subject"]), fp))
    rng = np.random.default_rng(seed)
    selected = []
    for key in ("under", "near", "over"):
        indices = rng.choice(len(pools[key]), size=2, replace=False)
        selected.extend([(key, *pools[key][int(index)]) for index in indices])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path, default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    parser.add_argument("--conditioning", type=Path, default=Path("data/processed/lee2019_cnn_gru_threshold_conditioning.json"))
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--sessions", type=int, nargs="*")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--horizon", type=float, default=120.0)
    parser.add_argument("--minimum-eligible", type=float, default=30.0)
    parser.add_argument("--update", type=float, default=5.0)
    parser.add_argument("--rho", type=float, default=.10)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_riemannian_idle_controller_pilot.json"))
    args = parser.parse_args()

    if args.pilot or not args.subjects:
        selected = select_stratified_pilot(args.conditioning)
    else:
        sessions = args.sessions or [1, 2]
        selected = [("manual", session, subject, np.nan) for session in sessions for subject in args.subjects]

    rows = []
    for session in sorted({item[1] for item in selected}):
        domains = load_covariances(session)
        for stratum, selected_session, subject, baseline_fp in selected:
            if selected_session != session:
                continue
            args.stratum = stratum
            row = analyze(
                args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz",
                subject,
                session,
                domains,
                args.shifts,
                args,
            )
            row["selection_baseline_fp_per_min"] = float(baseline_fp)
            rows.append(row)
            print(f"{stratum} S{session} sub-{subject:02d} baselineFP={baseline_fp:.3f}", flush=True)

    payload = {
        "protocol": {
            "representation": "frozen 3-seed CNN-GRU + all-score EWMA H30",
            "controller": "causal past-window fixed-threshold replay through MA0.5+dwell0.3+refractory1.0",
            "primary_deployable_arm": "Riemannian-only central-idle eligibility; CNN score never enters eligibility",
            "riemannian_training": "one-class affine-invariant distance to each domain's pre-rest reference; no target command labels",
            "riemannian_gate_selection": "source-only widest central geodesic-distance region with command-as-idle rate <= 0.10",
            "target_gate_normalization": "target pre-rest, label-free",
            "target_fp_per_min": TARGET_FP,
            "horizon_seconds": args.horizon,
            "minimum_eligible_seconds": args.minimum_eligible,
            "update_seconds": args.update,
            "rho": args.rho,
            "selection": "two subject-sessions sampled per pre-existing transferred-FP stratum using seed 20260909",
            "claim_limit": "Lee2019 cue-relative pseudo-uncued replay",
        },
        "selected": [
            {"stratum": key, "session": session, "subject": subject, "baseline_fp": fp}
            for key, session, subject, fp in selected
        ],
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
