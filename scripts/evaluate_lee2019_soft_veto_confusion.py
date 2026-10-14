"""Full-subject event-opportunity confusion matrices for soft-veto ensembles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.operating_point import tune_source_operating_points
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, calibration, reference, rolling_covariance,
)

LAMBDAS = (0., .25, .5, 1.)
TARGET_FP = 2.
GUARD = 3.


def diag_fit(x):
    return x.mean(0), np.maximum(x.var(0, ddof=1), .05)


def diag_logpdf(x, params):
    mean, var = params
    return -.5*np.sum(np.log(2*np.pi*var)+(x-mean)**2/var, axis=1)


def standardize(x, anchor):
    return (x-anchor.mean())/max(anchor.std(ddof=1), 1e-8)


def threshold(anchor_score):
    z = standardize(anchor_score, anchor_score)
    points = tune_source_operating_points(
        [calibration(z)], "ma_dwell", [TARGET_FP], STEP,
        ma_seconds=.5, dwell_seconds=.2, refractory_seconds=1., grid_size=160,
    )
    return float(points[TARGET_FP].threshold)


def opportunity_confusion(times, alarms, events, guard):
    """MI trials are positive; disjoint 4-s valid-idle blocks are negative."""
    tp = fn = fp = tn = 0
    for event in events:
        begin, end = event[5]/SFREQ + 1.5, event[5]/SFREQ + 4.0
        detected = bool(np.any(alarms & (times >= begin) & (times < end) & ~guard))
        tp += int(detected); fn += int(not detected)

    positive_support = np.zeros(len(times), bool)
    for event in events:
        positive_support |= (times >= event[5]/SFREQ + 1.5) & (times < event[5]/SFREQ + 4.0)
    valid_idle = ~positive_support & ~guard
    block_steps = int(round(4./STEP))
    indices = np.flatnonzero(valid_idle)
    if len(indices):
        cuts = np.flatnonzero(np.diff(indices) > 1) + 1
        for run in np.split(indices, cuts):
            for start in range(0, len(run)-block_steps+1, block_steps):
                block = run[start:start+block_steps]
                detected = bool(np.any(alarms[block]))
                fp += int(detected); tn += int(not detected)
    return np.asarray([[tp, fn], [fp, tn]], dtype=np.int64)


def analyze(subject, session, roots):
    stream_root, raw_root, cov_root = roots
    with np.load(cov_root/f"sub-{subject:02d}.npz", allow_pickle=False) as d:
        ct, it = d["train_command_trial"], d["train_idle_iti_trial"]
        command_cov = d["train_command_covariance"][ct < 50]
        iti_cov = d["train_idle_iti_covariance"][it < 50]
        fixation_cov = d["train_fixation_covariance"][d["train_fixation_trial"] < 50]
        pre_cov = d["pre_train_covariance"]

    ref = reference(iti_cov)
    command = tangent_features(command_cov, ref)
    iti = tangent_features(iti_cov, ref)
    scaler = StandardScaler().fit(np.r_[command, iti])
    lr = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000,
                            random_state=20260811).fit(
        scaler.transform(np.r_[command, iti]), np.r_[np.ones(len(command)), np.zeros(len(iti))])

    idle_all = np.r_[iti_cov, fixation_cov, pre_cov]
    refg = reference(idle_all)
    scaler_g = StandardScaler().fit(tangent_features(idle_all, refg))
    command_g = scaler_g.transform(tangent_features(command_cov, refg))
    idle_groups = [scaler_g.transform(tangent_features(x, refg))
                   for x in (iti_cov, fixation_cov, pre_cov)]
    command_params = diag_fit(command_g)
    idle_params = [diag_fit(x) for x in idle_groups]

    def llr(cov):
        x = scaler_g.transform(tangent_features(cov, refg))
        idle_ll = np.column_stack([diag_logpdf(x, p) for p in idle_params])
        return diag_logpdf(x, command_params)-logsumexp(idle_ll, axis=1)+np.log(len(idle_params))

    with np.load(raw_root/f"sub-{subject:02d}.npz", allow_pickle=False) as d:
        anchor_cov, _ = rolling_covariance(d["pre_train"])
    with np.load(stream_root/f"sub-{subject:02d}.npz", allow_pickle=False) as d:
        signal, events = d["signal"], d["events"]
    stream_cov, ends = rolling_covariance(signal)
    times = ends/SFREQ; starts = (ends-int(WINDOW*SFREQ))/SFREQ

    base_anchor = lr.decision_function(scaler.transform(tangent_features(anchor_cov, ref)))
    base_stream = lr.decision_function(scaler.transform(tangent_features(stream_cov, ref)))
    llr_anchor, llr_stream = llr(anchor_cov), llr(stream_cov)
    base_anchor_z, base_stream_z = standardize(base_anchor, base_anchor), standardize(base_stream, base_anchor)
    llr_anchor_z, llr_stream_z = standardize(llr_anchor, llr_anchor), standardize(llr_stream, llr_anchor)

    seams = np.unique(np.r_[events[:, 4], events[1:, 2]])/SFREQ
    guard = np.zeros(len(times), bool)
    for seam in seams:
        guard |= (starts < seam+GUARD) & (times > seam)

    matrices = {}
    for lam in LAMBDAS:
        anchor_score = base_anchor_z-lam*np.maximum(0., -llr_anchor_z)
        stream_score = base_stream_z-lam*np.maximum(0., -llr_stream_z)
        output = ewma_normalize_and_decide(
            stream_score, initial_mean=float(anchor_score.mean()),
            initial_variance=float(anchor_score.var(ddof=1)), threshold=threshold(anchor_score),
            step_seconds=STEP, half_life_seconds=30., update_mode="all", update_mask=~guard,
            ma_seconds=.5, dwell_seconds=.2, refractory_seconds=1., sigma_floor_fraction=.25,
        )
        matrices[str(lam)] = opportunity_confusion(times, output.alarms, events, guard)
    return matrices


def plot_matrix(ax, matrix, title):
    row_sum = matrix.sum(axis=1, keepdims=True)
    rate = np.divide(matrix, row_sum, where=row_sum > 0)
    ax.imshow(rate, vmin=0, vmax=1, cmap="Blues")
    names = np.asarray([["TP", "FN"], ["FP", "TN"]])
    for i in range(2):
        for j in range(2):
            color = "white" if rate[i,j] > .55 else "black"
            ax.text(j, i, f"{names[i,j]}\n{matrix[i,j]:,}\n{rate[i,j]*100:.1f}%",
                    ha="center", va="center", fontsize=12, color=color, fontweight="bold")
    ax.set_xticks([0,1], ["Positive", "Negative"])
    ax.set_yticks([0,1], ["Positive", "Negative"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--subjects", type=int, nargs="*")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/soft_veto_ensemble"))
    a = p.parse_args(); subjects = a.subjects or list(range(1,55))
    roots = (
        Path(f"data/interim/lee2019_pseudo_uncued_raw/session{a.session}"),
        Path(f"data/interim/lee2019_four_rest_raw/session{a.session}"),
        Path(f"data/interim/lee2019_four_rest/session{a.session}"),
    )
    totals = {str(lam): np.zeros((2,2), dtype=np.int64) for lam in LAMBDAS}
    rows = []
    for n, subject in enumerate(subjects, 1):
        result = analyze(subject, a.session, roots)
        serial = {key: value.tolist() for key, value in result.items()}
        rows.append({"subject": subject, "matrices": serial})
        for key, value in result.items(): totals[key] += value
        print(f"[{n:02d}/{len(subjects)}] subject {subject:02d}", flush=True)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"session": a.session, "subjects": subjects,
               "definition": "MI trial vs disjoint 4-s valid-idle opportunity",
               "totals": {k:v.tolist() for k,v in totals.items()}, "rows": rows}
    json_path = a.output_dir/f"session{a.session}_confusion.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
    for ax, lam in zip(axes, LAMBDAS):
        title = "TS-LR baseline" if lam == 0 else f"Soft veto λ={lam:g}"
        plot_matrix(ax, totals[str(lam)], title)
    fig.suptitle(f"Lee2019 session {a.session}: all {len(subjects)} subjects", fontweight="bold")
    fig_path = a.output_dir/f"session{a.session}_confusion.png"
    fig.savefig(fig_path, dpi=180)
    print(json.dumps({"json":str(json_path), "figure":str(fig_path),
                      "totals":payload["totals"]}, indent=2))


if __name__ == "__main__":
    main()
