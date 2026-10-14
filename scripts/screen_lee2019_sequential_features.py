"""Exploratory sequential feature screen in Riemannian non-response trials.

Feature direction is learned on trials 0--49.  Thresholds are estimated only
from pre-train rest.  Evaluation is restricted to locked trials 50--99 that
did not cross the deployed Riemannian FP=2 threshold.  This screen ranks
scalar MI-state features; it does not classify left versus right imagery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import lee2019_path, load_lee2019_run

FS = 250.0
WINDOW = 1.0
STEP = .1
REL_ENDS = np.arange(-1., 4.0001, STEP)
BANDS = {
    "theta_4_8": (4., 8.), "mu_low_8_10": (8., 10.),
    "mu_high_10_13": (10., 13.), "beta_low_13_18": (13., 18.),
    "beta_mid_18_24": (18., 24.), "beta_high_24_30": (24., 30.),
    "gamma_55_90": (55., 90.),
}


def car_select(values, channels, selected):
    lookup = {name.casefold(): i for i, name in enumerate(channels)}
    indices = [lookup[name.casefold()] for name in selected]
    centered = values - values.mean(axis=1, keepdims=True)
    return centered[:, indices]


def downsample(values, native_fs):
    ratio = int(round(native_fs / FS))
    if not np.isclose(native_fs / ratio, FS):
        raise ValueError("non-integral downsample ratio")
    return resample_poly(values, 1, ratio, axis=0)


def windows(values):
    width, step = int(WINDOW * FS), int(STEP * FS)
    ends = np.arange(width, len(values) + 1, step)
    return np.asarray([values[end-width:end] for end in ends])


def band_logpower(power, frequencies, low, high, channels=None):
    keep = (frequencies >= low) & (frequencies < high)
    selected = power[:, :, keep] if channels is None else power[:, channels][:, :, keep]
    return np.log(np.maximum(selected.mean(axis=(1, 2)), 1e-20))


def permutation_entropy(values):
    # Direction-agnostic order-pattern entropy, m=3, central mean signal.
    patterns = np.argsort(np.stack([values[:, :-2], values[:, 1:-1], values[:, 2:]], -1), -1)
    code = patterns[..., 0] * 9 + patterns[..., 1] * 3 + patterns[..., 2]
    output = np.zeros(len(values))
    for row in range(len(values)):
        _, counts = np.unique(code[row], return_counts=True)
        probabilities = counts / counts.sum()
        output[row] = -np.sum(probabilities * np.log(probabilities)) / np.log(6.)
    return output


def extract_features(values, channels, individual_mu):
    frames = windows(values)
    frames = frames - frames.mean(axis=1, keepdims=True)
    tapered = frames * np.hanning(frames.shape[1])[None, :, None]
    spectrum = np.fft.rfft(tapered, axis=1).transpose(0, 2, 1)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(frames.shape[1], 1 / FS)
    lookup = {name.casefold(): i for i, name in enumerate(channels)}
    central = [lookup[n.casefold()] for n in ("C3", "Cz", "C4")]
    c3, cz, c4 = central
    output = {name: band_logpower(power, frequencies, *band, central)
              for name, band in BANDS.items()}
    output["individual_mu"] = band_logpower(
        power, frequencies, max(6., individual_mu - 1.5), min(16., individual_mu + 1.5), central)

    # Surface-Laplacian approximations around C3 and C4.
    left_neighbours = [lookup[n.casefold()] for n in ("FC3", "C5", "C1", "CP3")]
    right_neighbours = [lookup[n.casefold()] for n in ("FC4", "C2", "C6", "CP4")]
    lap_left = frames[:, :, c3] - frames[:, :, left_neighbours].mean(axis=2)
    lap_right = frames[:, :, c4] - frames[:, :, right_neighbours].mean(axis=2)
    lap_spec = np.abs(np.fft.rfft(
        np.stack([lap_left, lap_right], axis=2) * np.hanning(frames.shape[1])[None, :, None], axis=1
    ).transpose(0, 2, 1)) ** 2
    output["laplacian_mu"] = band_logpower(lap_spec, frequencies, 8., 13.)

    c3_mu = band_logpower(power, frequencies, 8., 13., [c3])
    c4_mu = band_logpower(power, frequencies, 8., 13., [c4])
    output["mu_lateralization_abs"] = np.abs(c3_mu - c4_mu)

    covariance = np.asarray([np.cov(frame, rowvar=False) for frame in frames])
    diagonal = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-12))
    correlation = covariance / (diagonal[:, :, None] * diagonal[:, None, :])
    output["corr_c3_c4_abs"] = np.abs(correlation[:, c3, c4])
    output["corr_central_mean_abs"] = (
        np.abs(correlation[:, c3, cz]) + np.abs(correlation[:, cz, c4])) / 2
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-12)
    proportions = eigenvalues / eigenvalues.sum(axis=1, keepdims=True)
    output["cov_effective_rank"] = np.exp(-np.sum(proportions * np.log(proportions), axis=1))
    output["cov_top_fraction"] = proportions[:, -1]
    output["cov_logdet"] = np.log(eigenvalues).sum(axis=1)

    keep = (frequencies >= 8) & (frequencies <= 30)
    cross = spectrum[:, c3, keep] * np.conj(spectrum[:, c4, keep])
    denom = np.sqrt(power[:, c3, keep].sum(1) * power[:, c4, keep].sum(1))
    output["imaginary_coupling_c3_c4"] = np.abs(np.imag(cross).sum(1)) / np.maximum(denom, 1e-12)

    central_signal = frames[:, :, central].mean(axis=2)
    variance0 = np.var(central_signal, axis=1)
    variance1 = np.var(np.diff(central_signal, axis=1), axis=1)
    variance2 = np.var(np.diff(central_signal, n=2, axis=1), axis=1)
    mobility = np.sqrt(variance1 / np.maximum(variance0, 1e-12))
    output["hjorth_mobility"] = mobility
    output["hjorth_complexity"] = (
        np.sqrt(variance2 / np.maximum(variance1, 1e-12)) / np.maximum(mobility, 1e-12))
    central_power = power[:, central].mean(axis=1)
    slope_keep = (frequencies >= 2) & (frequencies <= 40)
    x = np.log(frequencies[slope_keep])
    x_centered = x - x.mean()
    output["aperiodic_slope_2_40"] = (
        np.log(np.maximum(central_power[:, slope_keep], 1e-20)) @ x_centered
        / np.sum(x_centered ** 2))
    probabilities = central_power[:, (frequencies >= 4) & (frequencies <= 40)]
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-20)
    output["spectral_entropy_4_40"] = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-20)), axis=1) / np.log(probabilities.shape[1])
    output["permutation_entropy"] = permutation_entropy(central_signal)
    return output


def peak_mu(rest_values, channels):
    frames = windows(rest_values)
    spectrum = np.abs(np.fft.rfft(
        (frames - frames.mean(axis=1, keepdims=True)) * np.hanning(frames.shape[1])[None, :, None], axis=1
    )) ** 2
    frequencies = np.fft.rfftfreq(frames.shape[1], 1 / FS)
    lookup = {name.casefold(): i for i, name in enumerate(channels)}
    central = [lookup[n.casefold()] for n in ("C3", "Cz", "C4")]
    psd = spectrum[:, :, central].mean(axis=(0, 2))
    keep = (frequencies >= 7) & (frequencies <= 14)
    return float(frequencies[keep][np.argmax(psd[keep])])


def sustained_response(mask, steps=3):
    return any(np.all(mask[i:i+steps]) for i in range(max(0, len(mask)-steps+1)))


def analyze_subject(subject, data_root, nonresponse):
    run = load_lee2019_run(lee2019_path(data_root, subject, 1), "train")
    config = ExperimentConfig()
    selected = config.sensorimotor_channels
    rest_native = car_select(run.pre_rest, run.channels, selected)
    pre_rest = downsample(rest_native, run.sfreq)
    mu_peak = peak_mu(pre_rest, selected)
    rest_features = extract_features(pre_rest, selected, mu_peak)

    # Trial support cue-2..+4 yields window ends cue-1..+4.
    before, after = int(2 * run.sfreq), int(4 * run.sfreq)
    trial_features = []
    for cue in run.cue_samples:
        epoch = car_select(run.eeg[int(cue)-before:int(cue)+after], run.channels, selected)
        trial_features.append(extract_features(downsample(epoch, run.sfreq), selected, mu_peak))

    fit_mi = (REL_ENDS >= 1.5) & (REL_ENDS <= 3.5)
    baseline = (REL_ENDS >= -.5) & (REL_ENDS <= 0.)
    locked_ids = sorted(nonresponse)
    records = {}
    for name in rest_features:
        rest = np.asarray(rest_features[name])
        fit_values = np.asarray([np.mean(trial_features[t][name][fit_mi]) for t in range(50)])
        rest_center = float(np.median(rest))
        direction = 1. if np.median(fit_values) >= rest_center else -1.
        directed_rest = direction * rest
        threshold = float(np.quantile(directed_rest, .95))
        rest_occupancy = float(np.mean(directed_rest >= threshold))
        trial_responses, mi_values, baseline_values = [], [], []
        trajectories = []
        for trial in locked_ids:
            values = np.asarray(trial_features[trial][name])
            directed = direction * values
            trial_responses.append(sustained_response(directed[fit_mi] >= threshold))
            mi_values.append(float(np.mean(directed[fit_mi])))
            baseline_values.append(float(np.mean(directed[baseline])))
            scale = max(float(np.median(np.abs(rest - rest_center))) * 1.4826, 1e-8)
            trajectories.append(direction * (values - rest_center) / scale)
        # Rest event-response probability for a duration matching the 2.1-s
        # strict evaluation support, using non-overlapping starts.
        span = int(np.sum(fit_mi))
        rest_events = [sustained_response(directed_rest[i:i+span] >= threshold)
                       for i in range(0, max(1, len(rest)-span+1), span)]
        records[name] = {
            "direction": int(direction), "individual_mu_hz": mu_peak,
            "nonresponse_trials": len(locked_ids),
            "rescue_fraction": float(np.mean(trial_responses)) if trial_responses else None,
            "rest_event_false_response_fraction": float(np.mean(rest_events)),
            "net_rescue": (float(np.mean(trial_responses)) - float(np.mean(rest_events)))
            if trial_responses else None,
            "mi_minus_precue": float(np.median(np.asarray(mi_values)-np.asarray(baseline_values)))
            if trial_responses else None,
            "trajectory": np.mean(trajectories, axis=0).tolist() if trajectories else None,
        }
    return {"subject": subject, "nonresponse_trials": len(locked_ids), "features": records}


def bootstrap_median(values, rng, repeats=5000):
    values = np.asarray(values, float)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def summarize(rows):
    rng = np.random.default_rng(20260811)
    names = list(rows[0]["features"])
    output = {}
    for name in names:
        valid = [r["features"][name] for r in rows if r["features"][name]["rescue_fraction"] is not None]
        rescue = np.asarray([v["rescue_fraction"] for v in valid])
        false = np.asarray([v["rest_event_false_response_fraction"] for v in valid])
        net = rescue - false
        change = np.asarray([v["mi_minus_precue"] for v in valid])
        try:
            p = float(wilcoxon(change, alternative="greater").pvalue)
        except ValueError:
            p = 1.
        trajectory = np.mean([v["trajectory"] for v in valid], axis=0)
        output[name] = {
            "subjects": len(valid), "rescue_fraction_median": float(np.median(rescue)),
            "rest_false_response_median": float(np.median(false)),
            "net_rescue_median": float(np.median(net)),
            "net_rescue_ci95": bootstrap_median(net, rng),
            "mi_minus_precue_median": float(np.median(change)),
            "mi_greater_precue_wilcoxon_p_uncorrected": p,
            "trajectory_mean_z": trajectory.tolist(),
        }
    ordered = sorted(output, key=lambda x: output[x]["net_rescue_median"], reverse=True)
    for rank, name in enumerate(ordered, 1):
        output[name]["rank"] = rank
    return output, ordered


def plot(summary, ordered, output):
    top = ordered[:12]
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), sharex=True)
    for axis, name in zip(axes.flat, top):
        axis.plot(REL_ENDS, summary[name]["trajectory_mean_z"], color="#16833b", lw=1.7)
        # x is the END time of a 1-s window. Ends in 0.5--1.5 s only
        # partially overlap the nominal MI interval; full strict support
        # (window start >= cue+0.5) begins at end time cue+1.5 s.
        axis.axvspan(.5, 1.5, color="#e69f00", alpha=.10)
        axis.axvspan(1.5, 3.5, color="#a40018", alpha=.12)
        axis.axvline(0, color="black", ls="--", lw=.8)
        axis.axhline(0, color="0.6", lw=.6)
        axis.set_title(f"{summary[name]['rank']}. {name}\nnet={summary[name]['net_rescue_median']:.3f}", fontsize=9)
    for axis in axes[-1]: axis.set_xlabel("Window end from cue (s)")
    for axis in axes[:, 0]: axis.set_ylabel("Directed feature (rest MAD z)")
    fig.suptitle(
        "Lee2019 Riemannian-nonresponse trials: sequential feature screen\n"
        "orange=partial MI overlap; red=complete 1-s support inside MI"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--onset-json", type=Path,
                        default=Path("data/processed/lee2019_onset_occupancy.json"))
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_sequential_feature_screen.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("data/processed/lee2019_sequential_feature_screen.png"))
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--rank-by", choices=("net", "precue"), default="net")
    args = parser.parse_args()
    if args.plot_only:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        ordered = payload["ranking"]
        if args.rank_by == "precue":
            ordered = sorted(
                payload["summary"],
                key=lambda name: payload["summary"][name]["mi_minus_precue_median"],
                reverse=True,
            )
            # Plot titles use rank stored for the original net ranking; replace
            # it only in-memory for the requested visualization.
            summary = json.loads(json.dumps(payload["summary"]))
            for rank, name in enumerate(ordered, 1):
                summary[name]["rank"] = rank
            plot(summary, ordered, args.figure)
        else:
            plot(payload["summary"], ordered, args.figure)
        return
    onset = json.loads(args.onset_json.read_text(encoding="utf-8"))
    mapping = {int(r["subject"]): {int(e["source_trial"]) for e in r["event_records"]
                                  if not e["threshold_response"]}
               for r in onset["subjects"]}
    subjects = args.subjects or sorted(mapping)
    rows = []
    for index, subject in enumerate(subjects, 1):
        print(f"feature screen {index}/{len(subjects)} subject={subject}", flush=True)
        rows.append(analyze_subject(subject, args.data_root, mapping[subject]))
    summary, ordered = summarize(rows)
    payload = {
        "design": {
            "fit_trials": "0-49 determine direction only",
            "evaluation_trials": "50-99 restricted to Riemannian FP2 non-response",
            "threshold": "one-sided preTrainRest 95th percentile; 0.3-s sustained",
            "claim_limit": "exploratory non-causal screen; selected candidates require locked causal validation",
        },
        "ranking": ordered, "summary": summary, "subjects": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(summary, ordered, args.figure)
    print(json.dumps({name: summary[name] for name in ordered[:10]}, indent=2))


if __name__ == "__main__":
    main()
