"""Batch the frozen I6 FP2 four-panel plot for all 54 Lee subjects.

Each PNG contains two adjacent subjects (S1 and S2 each) so the layout is
identical to the example figure. No model training or threshold tuning occurs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import plot_i6_original_continuous_four_panel_20261014 as plotter

OUTDIR = ROOT / "data/processed/i6_cnn_gru_original_continuous_fp2_timelines_all_subjects_20261014"
FULL_OUTDIR = ROOT / "data/processed/i6_cnn_gru_original_continuous_fp2_full_future_20261014"
ONE_ROW_OUTDIR = ROOT / "data/processed/i6_cnn_gru_original_continuous_fp2_full_future_one_row_20261014"
FOUR_ROWS_OUTDIR = ROOT / "data/processed/i6_cnn_gru_original_continuous_fp2_full_future_four_rows_20261014"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-future", action="store_true",
                        help="plot all future trials 70--99 with task-idle strata")
    parser.add_argument("--one-row", action="store_true",
                        help="plot four full-future panels horizontally in one row")
    parser.add_argument("--four-rows", action="store_true",
                        help="plot four full-future panels vertically in one column")
    args = parser.parse_args()
    if (args.one_row or args.four_rows) and not args.full_future:
        parser.error("--one-row and --four-rows require --full-future")
    if args.one_row and args.four_rows:
        parser.error("choose only one panel layout")
    outdir = (FOUR_ROWS_OUTDIR if args.four_rows else ONE_ROW_OUTDIR if args.one_row
              else FULL_OUTDIR if args.full_future else OUTDIR)
    result = json.loads(plotter.RESULT.read_text(encoding="utf-8"))
    rows = {
        (int(r["subject"]), int(r["session"])): r
        for r in result["arms"]["stateful"]["rows"]
        if r["arm"] == "T_all" and r["target_fp"] == 2.0
    }
    expected = {(subject, session) for subject in range(1, 55) for session in (1, 2)}
    if set(rows) != expected:
        raise AssertionError("I6 result does not have all 54 subjects x 2 sessions")
    for subject, session in sorted(expected):
        score = plotter.CACHE / f"session{session}" / f"sub-{subject:02d}.npz"
        tier0 = plotter.TIER0 / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz"
        if not score.exists() or not tier0.exists():
            raise FileNotFoundError(f"missing score or tier-0 cache for {subject}, {session}")
    targets = [
        outdir / f"subjects_{subject:02d}_{subject + 1:02d}.png"
        for subject in range(1, 55, 2)
    ]
    targets += [outdir / "index.csv", outdir / "README.md"]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise FileExistsError(f"batch refuses to overwrite existing outputs: {existing[:3]}")
    outdir.mkdir(parents=True, exist_ok=True)

    index = []
    for subject in range(1, 55, 2):
        filename = f"subjects_{subject:02d}_{subject + 1:02d}.png"
        cases = (
            (subject, 1), (subject, 2),
            (subject + 1, 1), (subject + 1, 2),
        )
        plotter.main(cases=cases, out=outdir / filename,
                     view_seconds=None if args.full_future else 200.0,
                     show_strata=args.full_future, one_row=args.one_row,
                     four_rows=args.four_rows)
        for s, session in cases:
            row = rows[(s, session)]
            index.append({
                "subject": s,
                "session": session,
                "figure": filename,
                "panel": ("row-1" if (s, session) == cases[0] else
                          "row-2" if (s, session) == cases[1] else
                          "row-3" if (s, session) == cases[2] else "row-4") if args.four_rows else
                         ("left-1" if (s, session) == cases[0] else
                          "left-2" if (s, session) == cases[1] else
                          "left-3" if (s, session) == cases[2] else "left-4") if args.one_row else
                         "top-left" if (s, session) == cases[0] else
                         "top-right" if (s, session) == cases[1] else
                         "bottom-left" if (s, session) == cases[2] else "bottom-right",
                "target_fp_per_min": 2,
                "t_all_threshold": row["calibration"]["threshold"],
                "future_idle_fp_per_min": row["future_idle"]["fp_per_min"],
                "future_sen_2s": row["mi"]["sen_2s"],
            })
    if len(index) != 108:
        raise AssertionError("incomplete figure index")
    with (outdir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    readme = """# Frozen I6 CNN-GRU FP2 time-domain figures, Lee2019

- 27 four-panel PNGs cover every subject 01–54 and both sessions (108 panels). {layout_description}
- Each panel shows {view_description}. Panel-title FP/min and SEN@2s use the entire future trials 70–99.
- Score: existing source-pooled CNN-GRU, stateful inference on the original continuous Lee train recording. This is **not a newly improved or larger model**.
- Threshold: existing T-all FP2 calibration from cued task idle in trials 10–69. Zero is the score minus threshold decision boundary. Blue is EWMA-z margin; red is MA(0.5 s) margin.
- Pink denotes cue-defined MI interval (cue+0.5 to cue+3.5 s). {strata_description} Black triangles are alarms in the cue-defined MI interval; red crosses are task-idle alarms; open circles are alarms outside both masks.
- All four panels in each PNG share one y-axis scale, calculated from that PNG's displayed samples. Y limits can differ between PNGs; numerical comparison across subjects should use `index.csv` or the original result JSON, not apparent visual amplitude.
- The recording is cue-paced. These figures do not establish uncued covert-MI onset detection or MI-specific responses.
- Source result: `../i6_cnn_gru_original_continuous_20261012.json`.
- Plot scripts: `../../../scripts/plot_i6_original_continuous_four_panel_20261014.py` and `../../../scripts/plot_i6_all_subject_timelines_20261014.py`.
""".format(
        layout_description=("Panels are stacked top-to-bottom: subject A S1, subject A S2, subject B S1, subject B S2; each panel has a long shared time axis."
                            if args.four_rows else
                            "Panels are left-to-right: subject A S1, subject A S2, subject B S1, subject B S2; each panel has a longer time axis than in the earlier 2x2 figures."
                            if args.one_row else "Panels are arranged in a 2x2 grid."),
        view_description=("the full original continuous future trial 70–99 interval"
                          if args.full_future else "the first 200 s after future-evaluation onset"),
        strata_description=("Pale blue, amber, and green denote eligible fixation, ITI-early, and ITI-late decision points. ITI-early may contain MI carryover."
                            if args.full_future else ""),
    )
    (outdir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Complete: {len(targets) - 2} PNGs, {len(index)} subject-session panels in {outdir}")


if __name__ == "__main__":
    main()
