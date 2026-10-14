from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.data import lee2019_path


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, default=Path("data/processed/anticipatory_emg.json"))
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            path = lee2019_path(args.data_root, subject, session)
            mat = loadmat(
                path, variable_names=["EEG_MI_train"], squeeze_me=True,
                struct_as_record=False, verify_compressed_data_integrity=False,
            )["EEG_MI_train"]
            emg = np.asarray(mat.EMG, dtype=float)
            fs = float(mat.fs)
            cues = np.asarray(mat.t, dtype=int).reshape(-1)
            sos = butter(4, [20.0, 200.0], btype="bandpass", fs=fs, output="sos")
            filtered = sosfilt(sos, emg, axis=0)
            early, late = [], []
            bins = [[] for _ in range(15)]
            for cue in cues:
                early.append(rms(filtered[cue - int(3 * fs): cue - int(2 * fs)]))
                late.append(rms(filtered[cue - int(fs): cue]))
                for index in range(15):
                    start = cue - int(3 * fs) + int(index * 0.2 * fs)
                    stop = start + int(0.2 * fs)
                    bins[index].append(rms(filtered[start:stop]))
            rows.append({
                "subject": subject,
                "session": session,
                "early_fixation_rms": float(np.median(early)),
                "peri_cue_rms": float(np.median(late)),
                "log_peri_over_early": float(np.log(np.median(late) / np.median(early))),
                "fixation_bin_rms": [float(np.median(x)) for x in bins],
            })
            print(f"done s{subject:02d} session{session}")
    summary = {}
    for session in args.sessions:
        values = np.asarray([x["log_peri_over_early"] for x in rows if x["session"] == session])
        summary[str(session)] = {
            "subjects": len(values),
            "median_log_ratio": float(np.median(values)),
            "median_ratio": float(np.exp(np.median(values))),
            "wilcoxon_p": float(wilcoxon(values).pvalue),
        }
    result = {"method": "causal 20-200 Hz EMG, median trial RMS; cue-1..0 versus cue-3..-2", "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
