"""Lee2019 G2 Method-B sensitivity: causal high-beta oscillator state.

This freezes every G2-A ROI, endpoint, and replication choice and replaces
only the amplitude estimator.  A causal AR(2) resonator centered at 25 Hz
with Q=2.5 (10-Hz 3-dB bandwidth) represents the 20--30-Hz oscillator state;
its squared output receives the same 150-ms causal EMA used by Method A.  It
is an estimator-family sensitivity analysis, not a parameter search intended
to rescue G2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import iirpeak, lfilter

import analyze_lee2019_rest_to_mi_g2 as g2


CENTER_HZ = 25.0
HALF_BANDWIDTH_HZ = 5.0


def oscillator_power(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Causal AR(2) oscillator/resonator power."""
    quality_factor = CENTER_HZ / (2.0 * HALF_BANDWIDTH_HZ)
    numerator, denominator = iirpeak(CENTER_HZ, quality_factor, fs=sfreq)
    oscillator_state = lfilter(numerator, denominator, signal, axis=0)
    instantaneous_power = oscillator_state * oscillator_state
    power_alpha = 1.0 - np.exp(-1.0 / (sfreq * g2.SMOOTHER_TAU_SECONDS))
    return lfilter(
        [power_alpha], [1.0, -(1.0 - power_alpha)],
        instantaneous_power, axis=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("data/processed/lee2019_rest_to_mi_g2_state_space_ar2_checkpoint.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_rest_to_mi_g2_state_space.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("data/processed/lee2019_rest_to_mi_g2_state_space.png"),
    )
    args = parser.parse_args()

    specification = {
        "method": "G2 Method B estimator-family sensitivity",
        "mapping_version": g2.MAPPING_VERSION,
        "primary_band_hz": list(g2.PRIMARY_BAND),
        "estimator_version": "ar2_resonator_v2",
        "amplitude_estimator": "causal AR(2) state-space resonator",
        "oscillator_center_hz": CENTER_HZ,
        "oscillator_half_bandwidth_hz": HALF_BANDWIDTH_HZ,
        "oscillator_q": CENTER_HZ / (2.0 * HALF_BANDWIDTH_HZ),
        "power_smoother_tau_s": g2.SMOOTHER_TAU_SECONDS,
        "reference": "CAR across all 62 channels before ROI selection",
        "filter_history": "continuous with 60-s pre-rest warm-up",
        "trial_baseline_s": [-1.0, 0.0],
        "curve_s": [g2.CURVE_START, g2.CURVE_STOP, g2.CURVE_STEP],
        "fit_range_s": list(g2.FIT_RANGE),
        "changepoint_search_s": list(g2.CHANGE_SEARCH),
        "medial_broad": list(g2.MEDIAL_BROAD),
        "medial_strict_sensitivity": list(g2.MEDIAL_STRICT),
        "quality_delta_bic": g2.QUALITY_DELTA_BIC,
        "primary_contrast": "tau_contralateral - tau_medial_broad",
        "discovery_replication": "unchanged specification in S1 and S2",
        "claim_limit": "cue-locked scalp state; Method-B sensitivity only",
    }

    rows: list[dict] = []
    if args.checkpoint.exists():
        cached = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if cached.get("specification") != specification:
            raise RuntimeError("Checkpoint specification differs; use a new path")
        rows = cached.get("rows", [])
    complete = {(int(row["subject"]), int(row["session"])) for row in rows}
    for session in args.sessions:
        for index, subject in enumerate(args.subjects, start=1):
            if (subject, session) in complete:
                print(f"G2-B S{session} sub-{subject:02d} cached", flush=True)
                continue
            row = g2.analyze_one(
                args.data_root, subject, session, power_estimator=oscillator_power
            )
            rows.append(row)
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            args.checkpoint.write_text(json.dumps({
                "specification": specification, "rows": rows,
            }), encoding="utf-8")
            print(f"G2-B S{session} sub-{subject:02d} [{index}/{len(args.subjects)}]", flush=True)

    expected = {(subject, session) for subject in args.subjects for session in args.sessions}
    rows = [row for row in rows if (row["subject"], row["session"]) in expected]
    payload = {
        "protocol": specification,
        "summary": g2.make_summary(rows, args.bootstrap),
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    g2.plot_results(rows, args.figure, "causal AR(2) oscillator + power EMA")
    print(args.output)
    print(args.figure)


if __name__ == "__main__":
    main()
