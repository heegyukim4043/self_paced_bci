"""Plot frozen CNN-GRU scores under random evaluation idle/MI ratios."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from uncued_mi.decision import instantaneous
from uncued_mi.operating_point import ScoredSequence
from scripts.run_lee2019_cnn_gru import train_model, predict
from scripts.run_lee2019_cnn_gru_idle_ratio import prepare, build_ratio_stream
from scripts.run_lee2019_cnn_gru_eval_idle_spacing import IDLE_MI_RATIOS, blocks
from scripts.run_lee2019_stateful_gru import SEEDS, build_no_guard_stream, tune_threshold, calibration_sequence

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=3); p.add_argument("--session", type=int, default=1)
    p.add_argument("--stop", type=float, default=120); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--operating", choices=("common_fp", "transferred"), default="common_fp")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--output", type=Path)
    a = p.parse_args(); device = torch.device(a.device)
    train_idle, train_cmd, canonical, ct, cals = prepare(ROOT/"data/raw/lee2019_mi", ROOT/"data/interim/lee2019_mil_raw", a.subject, a.session)
    eval_idle, eval_cmd = blocks(canonical, ct, ct.idle_intervals), blocks(canonical, ct, ct.command_intervals)
    models, cal_scores = [], []
    for seed in SEEDS:
        x, y, w, _ = build_no_guard_stream(train_idle, train_cmd, 20260700 + 100*a.session + a.subject)
        model = train_model(x, y, w, seed, a.epochs, device); models.append(model)
        cal_scores.append([predict(model, v, device) for v in cals])
    calibration = [np.mean([z[i] for z in cal_scores], axis=0) for i in range(len(cals))]
    transferred = tune_threshold([calibration_sequence(x) for x in calibration]); rows = []
    composition_seed = 202609080 + 100*a.session + a.subject
    for ratio in IDLE_MI_RATIOS:
        x, _, _, template, audit = build_ratio_stream(eval_idle, eval_cmd, ratio, composition_seed, minimum_steps=5)
        score = np.mean([predict(model, x, device) for model in models], axis=0)
        seq = ScoredSequence(template.times, score, template.idle_intervals, template.command_intervals)
        point = tune_threshold([seq]) if a.operating == "common_fp" else transferred
        alarms = instantaneous(score, point.threshold, .1, 1.).alarms
        rows.append((ratio, seq, point.threshold, alarms, audit))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for ax, (ratio, seq, threshold, alarms, audit) in zip(axes, rows):
        view = seq.times <= a.stop
        for j, (lo, hi) in enumerate(seq.command_intervals):
            if lo <= a.stop: ax.axvspan(lo, min(hi, a.stop), color="#e98b9a", alpha=.24, label="true MI" if j == 0 else None)
        margin = seq.scores - threshold
        ax.plot(seq.times[view], margin[view], color="#2878b5", lw=1.15, label="CNN-GRU score - threshold")
        ax.axhline(0, color="#c82433", ls="--", lw=1.1, label="decision boundary")
        alarm_mask = alarms & view
        ax.scatter(seq.times[alarm_mask], margin[alarm_mask], marker="v", s=32, color="black", zorder=5, label="alarm")
        idle = sum(hi-lo for lo, hi in seq.idle_intervals); total = seq.times[-1] + .1
        ax.set_title(f"idle/MI={ratio:g}:1 (idle occupancy={ratio/(1+ratio):.1%}) | random idle={idle:.1f}s, full stream={total:.1f}s, threshold={threshold:.2f}", loc="left")
        ax.set_ylabel("margin"); ax.grid(axis="y", alpha=.18); ax.legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlim(0, a.stop); axes[-1].set_xlabel("Evaluation-stream time (s)")
    fig.suptitle(f"Frozen CNN-GRU: random idle/MI-ratio robustness — Lee2019 sub-{a.subject:02d} S{a.session}\n{a.operating}; same model and paired MI-trial order", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    out = a.output or ROOT/f"data/processed/lee2019_cnn_gru_eval_idle_mi_ratio_timeline_sub{a.subject:02d}_s{a.session}_{a.operating}_0-{a.stop:g}s.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=180, bbox_inches="tight"); print(out)

if __name__ == "__main__": main()
