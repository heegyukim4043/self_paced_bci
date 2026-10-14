"""Combine the two locked-session soft-veto confusion results."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_lee2019_soft_veto_confusion import LAMBDAS, plot_matrix


def metrics(m):
    tp, fn, fp, tn = m[0,0], m[0,1], m[1,0], m[1,1]
    return {
        "sensitivity": float(tp/(tp+fn)), "specificity": float(tn/(tn+fp)),
        "precision": float(tp/(tp+fp)),
        "balanced_accuracy": float(.5*(tp/(tp+fn)+tn/(tn+fp))),
    }


def main():
    root = Path("data/processed/soft_veto_ensemble")
    payloads = [json.loads((root/f"session{s}_confusion.json").read_text(encoding="utf-8"))
                for s in (1,2)]
    totals = {str(lam): sum((np.asarray(p["totals"][str(lam)], dtype=np.int64)
                             for p in payloads), np.zeros((2,2), dtype=np.int64))
              for lam in LAMBDAS}
    output = {"sessions":[1,2], "subject_sessions":108,
              "definition":"MI trial vs disjoint 4-s valid-idle opportunity",
              "totals":{k:v.tolist() for k,v in totals.items()},
              "metrics":{k:metrics(v) for k,v in totals.items()}}
    (root/"combined_confusion.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    fig, axes = plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
    for ax, lam in zip(axes,LAMBDAS):
        plot_matrix(ax,totals[str(lam)],"TS-LR baseline" if lam==0 else f"Soft veto λ={lam:g}")
    fig.suptitle("Lee2019 sessions 1+2: 108 subject-sessions",fontweight="bold")
    fig.savefig(root/"combined_confusion.png",dpi=180)
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
