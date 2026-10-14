"""CNN-GRU LOSO validation on the exact seam-guarded Arm-A replay stream.

This is the corrected replacement for the earlier pilot runner.  The earlier
runner evaluated a separately randomized synthetic stream and is not a valid
Arm-A comparison.  Here the 3-band causal EEG stream is rebuilt from the raw
Lee2019 runs using the locked pseudo-uncued stream schedule; event rows must
match the saved Riemannian Arm-A stream sample-for-sample before evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import lee2019_path, load_lee2019_run
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence,
    tune_source_operating_points,
)
from uncued_mi.pseudo_uncued import (
    build_pseudo_uncued_raw_stream,
    fixed_budget_exponential_chunks,
)
from scripts.run_fbcnet_validation import filter_banks, normalize, windows
from scripts.run_lee2019_cnn_gru import (
    CausalCNNGRU,
    SFREQ,
    STEP,
    make_windows,
    predict,
    rest_windows,
)
from scripts.run_lee2019_stateful_gru import build_no_guard_stream
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma,
    ewma_z,
    interval_metrics,
    intervals_from_alarm,
    ma_trace,
)


SUBJECTS = tuple(range(1, 55))
SCREEN = 10
WINDOW = 250
STEP_SAMPLES = 25
MA_SECONDS = 0.5
DWELL_SECONDS = 0.3
REFRACTORY_SECONDS = 1.0
TARGETS = (2.0, 4.0)
DELTAS = (0.5, 1.0, 2.0, 3.0, 4.0)
BANDS = ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0))
STREAM_SEED = 20260804
MODEL_PROTOCOL_NAME = "causal CNN-GRU, source-pooled LOSO (target subject excluded)"


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = max(float(values.std(ddof=1)), 1e-8)
    return (values - float(values.mean())) / scale


def _window_ends(n_samples: int, *, first_end: int = WINDOW) -> np.ndarray:
    return np.arange(first_end, n_samples + 1, STEP_SAMPLES, dtype=np.int64)


@torch.no_grad()
def predict_signal(model, signal: np.ndarray, device: torch.device,
                   *, ends: np.ndarray | None = None, batch_size: int = 100) -> np.ndarray:
    """Run the causal CNN-GRU in chronological chunks without materializing all windows."""
    if ends is None:
        ends = _window_ends(signal.shape[-1])
    ends = np.asarray(ends, dtype=np.int64)
    ends = ends[(ends >= WINDOW) & (ends <= signal.shape[-1])]
    model.eval()
    hidden = None
    output: list[np.ndarray] = []
    for offset in range(0, len(ends), batch_size):
        selected = ends[offset : offset + batch_size]
        batch = np.stack(
            [signal[:, :, end - WINDOW : end] for end in selected], axis=0
        ).astype(np.float32, copy=False)
        logits, hidden = model(torch.from_numpy(batch[None]).to(device), hidden)
        output.append(logits[0].float().cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def _normalize_banks(signal: np.ndarray, pre: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calibration = pre[..., int(2 * SFREQ) :]
    mean = calibration.mean(axis=-1, keepdims=True)
    std = calibration.std(axis=-1, keepdims=True).clip(1e-6)
    return (signal - mean) / std, (pre - mean) / std, mean, std


def _source_training(data_root: Path, cache_root: Path, subject: int,
                     session: int, max_source_windows: int) -> dict:
    run_path = lee2019_path(data_root, subject, session)
    train = load_lee2019_run(run_path, "train")
    config = ExperimentConfig(window_seconds=1.0, feature_step_seconds=STEP)
    selected = config.sensorimotor_channels
    signal_raw = filter_banks(train.eeg, train.channels, train.sfreq, selected)
    pre_raw = filter_banks(train.pre_rest, train.channels, train.sfreq, selected)
    signal, pre, _, _ = _normalize_banks(signal_raw, pre_raw)

    cache_path = cache_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        command_trials = []
        for start, stop in cache["command_intervals"]:
            ends = np.arange(
                int(round((float(start) + 1.0) * SFREQ)),
                int(round(float(stop) * SFREQ)), STEP_SAMPLES,
            )
            command_trials.append(make_windows(signal, ends))
        iti_trials = []
        for trial in range(50):
            mask = cache["iti_instance_trials"] == trial
            if np.any(mask):
                iti_trials.append(make_windows(signal, cache["iti_end_samples"][mask]))

    rest_fit, _ = rest_windows(pre)
    train_x, train_y, train_w, _ = build_no_guard_stream(
        iti_trials + [rest_fit], command_trials[:50],
        20260700 + 100 * session + subject,
    )
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    train_w = np.asarray(train_w, dtype=np.float32)
    if max_source_windows > 0 and len(train_x) > max_source_windows:
        # Preserve a contiguous sequence; randomly shuffled windows would
        # corrupt the temporal context the GRU is meant to learn.
        length = int(max_source_windows)
        stride = max(1, length // 10)
        starts = np.arange(0, len(train_x) - length + 1, stride)
        cumulative = np.r_[0, np.cumsum(train_y)]
        positives = cumulative[starts + length] - cumulative[starts]
        negatives = length - positives
        balance = np.minimum(positives, negatives)
        best = starts[balance == balance.max()]
        rng = np.random.default_rng(20260911 + 100 * session + subject)
        start = int(rng.choice(best))
        stop = start + length
        train_x, train_y, train_w = train_x[start:stop], train_y[start:stop], train_w[start:stop]

    # Only the second half is used for source-side transferred-threshold
    # calibration, keeping it disjoint from the source training idle block.
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_w": train_w,
        "anchor_signal": pre.astype(np.float32),
        "anchor_midpoint_seconds": float(pre.shape[-1] / SFREQ / 2.0),
    }


def _cached_source_training(
    data_root: Path, cache_root: Path, prepared_root: Path | None,
    subject: int, session: int, max_source_windows: int,
) -> dict:
    """Reuse model-independent filtered source sequences across architectures."""
    path = None if prepared_root is None else (
        prepared_root / f"session{session}"
        / f"sub-{subject:02d}_n{max_source_windows}.npz"
    )
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as data:
            return {
                "train_x": np.asarray(data["train_x"], np.float32),
                "train_y": np.asarray(data["train_y"], np.float32),
                "train_w": np.asarray(data["train_w"], np.float32),
                "anchor_signal": np.asarray(data["anchor_signal"], np.float32),
                "anchor_midpoint_seconds": float(data["anchor_midpoint_seconds"]),
            }
    result = _source_training(
        data_root, cache_root, subject, session, max_source_windows,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            train_x=result["train_x"], train_y=result["train_y"],
            train_w=result["train_w"], anchor_signal=result["anchor_signal"],
            anchor_midpoint_seconds=np.asarray(result["anchor_midpoint_seconds"]),
        )
    return result


def _rebuild_exact_target_stream(data_root: Path, stream_root: Path,
                                 subject: int, session: int,
                                 stream_seed: int = STREAM_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recreate 3-band input and assert exact correspondence with locked events."""
    train = load_lee2019_run(lee2019_path(data_root, subject, session), "train")
    test = load_lee2019_run(lee2019_path(data_root, subject, session), "test")
    config = ExperimentConfig(window_seconds=1.0, feature_step_seconds=STEP)
    selected = config.sensorimotor_channels
    train_raw = filter_banks(train.eeg, train.channels, train.sfreq, selected)
    rest_raw = [
        filter_banks(values, channels, sfreq, selected)
        for values, channels, sfreq in (
            (train.pre_rest, train.channels, train.sfreq),
            (train.post_rest, train.channels, train.sfreq),
            (test.pre_rest, test.channels, test.sfreq),
            (test.post_rest, test.channels, test.sfreq),
        )
    ]
    train_signal, pre, mean, std = _normalize_banks(train_raw, rest_raw[0])
    normalized_rests = [(value - mean) / std for value in rest_raw]
    transient = int(round(config.transient_seconds * SFREQ))
    blocks = [value[..., transient:-transient] for value in normalized_rests[1:]]

    locked_path = stream_root / f"session{session}" / f"sub-{subject:02d}.npz"
    with np.load(locked_path, allow_pickle=False) as locked:
        locked_events = np.asarray(locked["events"])
        chronological_spans = (np.asarray(locked["rest_source_spans_250hz"])
                               if "rest_source_spans_250hz" in locked else None)
        locked_trials = np.asarray(locked["trial_ids"])
        locked_sources = np.asarray(locked["rest_source_ids"])

    base_seed = stream_seed + 1000 * session + subject
    chunks_by_band: list[list[np.ndarray]] = [[] for _ in range(3)]
    if chronological_spans is not None:
        trial_indices = locked_trials
        prepared_sources = locked_sources
        for band in range(3):
            chunks_by_band[band] = [
                normalized_rests[1 + int(source)][band, :, int(start):int(stop)].T
                for source, (start, stop) in zip(prepared_sources, chronological_spans)
            ]
    else:
        rng = np.random.default_rng(base_seed)
        selected_order = rng.permutation(np.arange(50, 100))
        selected_parts: list[np.ndarray] = []
        sources_by_fold: list[np.ndarray] = []
        for fold in range(5):
            selected_parts.append(selected_order[10 * fold : 10 * (fold + 1)])
            fold_sources = None
            for band in range(3):
                fold_chunks, fold_source_ids = fixed_budget_exponential_chunks(
                    [block[band].T for block in blocks],
                    chunks_per_block=(3, 3, 4), sfreq=SFREQ,
                    minimum_seconds=5.0,
                    seed=stream_seed + 2000 * session + 100 * fold + subject,
                )
                chunks_by_band[band].extend(fold_chunks)
                if fold_sources is None:
                    fold_sources = fold_source_ids
                elif not np.array_equal(fold_sources, fold_source_ids):
                    raise AssertionError("Bandwise rest chunk source order diverged")
            assert fold_sources is not None
            sources_by_fold.append(fold_sources)
        trial_indices = np.concatenate(selected_parts)
        prepared_sources = np.concatenate(sources_by_fold)
    cue_samples = np.rint(train.cue_samples / train.sfreq * SFREQ).astype(np.int64)
    band_streams = []
    event_reference = None
    for band in range(3):
        stream = build_pseudo_uncued_raw_stream(
            continuous_trial_signal=train_signal[band].T,
            cue_samples=cue_samples,
            labels=train.labels,
            rest_blocks=[block[band].T for block in blocks],
            sfreq=SFREQ,
            trial_indices=trial_indices,
            minimum_rest_seconds=5.0,
            maximum_rest_seconds=60.0,
            seed=base_seed,
            prepared_rest_chunks=chunks_by_band[band],
            prepared_rest_source_ids=prepared_sources,
        )
        if event_reference is None:
            event_reference = stream.events
        elif not np.array_equal(event_reference, stream.events):
            raise AssertionError("Bandwise pseudo-stream event timing diverged")
        band_streams.append(stream.signal.T)
    assert event_reference is not None

    if not np.array_equal(event_reference, locked_events):
        mismatch = np.argwhere(event_reference != locked_events)
        first = mismatch[0].tolist() if len(mismatch) else None
        raise AssertionError(
            f"Rebuilt event schedule differs from Arm-A cache at {first}; refusing metrics"
        )
    signal = np.asarray(band_streams, dtype=np.float32)
    return signal, event_reference, pre.astype(np.float32)


def train_source_pool(blocks: list[dict], seed: int, epochs: int,
                      device: torch.device, sequence_batch_size: int = 16,
                      model_class: type[nn.Module] = CausalCNNGRU,
                      hidden: int = 32) -> nn.Module:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model_class(20, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(blocks))
        for start in range(0, len(order), sequence_batch_size):
            selected = [blocks[int(i)] for i in order[start : start + sequence_batch_size]]
            max_len = min(len(item["train_x"]) for item in selected)
            x = np.stack([item["train_x"][:max_len] for item in selected])
            y = np.stack([item["train_y"][:max_len] for item in selected])
            w = np.stack([item["train_w"][:max_len] for item in selected])
            tx = torch.from_numpy(x).to(device)
            ty = torch.from_numpy(y).to(device)
            tw = torch.from_numpy(w).to(device)
            pos = max(float(np.sum(y == 0) / max(np.sum(y == 1), 1)), 1.0)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tx)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, ty, weight=tw,
                pos_weight=torch.tensor(pos, device=device),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def _calibration_threshold(z_series: list[np.ndarray], target: float) -> tuple[float, float]:
    sequences = []
    for values in z_series:
        z = np.asarray(values, dtype=float)
        times = np.arange(len(z), dtype=float) * STEP
        sequences.append(ScoredSequence(
            times, z,
            np.asarray([[0.0, len(z) * STEP]], dtype=float),
            np.empty((0, 2), dtype=float),
        ))
    point = tune_source_operating_points(
        sequences, "ma_dwell", [target], STEP,
        ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, grid_size=160,
    )[target]
    return float(point.threshold), float(point.source_fp_per_minute)


def _eval_threshold(times: np.ndarray, scores: np.ndarray, anchor: np.ndarray,
                    threshold: float, allowed: np.ndarray, evalmask: np.ndarray,
                    events: np.ndarray, shifts: int, seed: int) -> dict:
    initial_mean = float(np.mean(anchor))
    initial_var = float(np.var(anchor, ddof=1))
    z = ewma_z(scores, initial_mean, initial_var, allowed)
    ma = ma_trace(z, n=int(round(MA_SECONDS / STEP)))
    alarms = alarms_from_ma(
        ma, threshold, allowed,
        dwell=int(round(DWELL_SECONDS / STEP)),
        refractory=int(round(REFRACTORY_SECONDS / STEP)),
    ) & evalmask

    cues = events[:, 5] / SFREQ
    onsets = cues[SCREEN:] + 0.5
    gt = np.c_[onsets, cues[SCREEN:] + 4.0]
    state = np.zeros(len(times), dtype=bool)
    for start, stop in gt:
        state |= (times >= start) & (times <= stop)
    idle = evalmask & allowed & ~state
    exposure = float(idle.sum() * STEP)
    fp_count = int(np.sum(alarms & idle))

    def sens(candidate: np.ndarray, delta: float) -> float:
        return float(np.mean([
            np.any(candidate & (times >= onset) & (times <= onset + delta))
            for onset in onsets
        ]))

    onset_results = {}
    for delta in DELTAS:
        onset_results[str(delta)] = sens(alarms, delta)
    latency = []
    for onset in onsets:
        hit = times[alarms & (times >= onset) & (times <= onset + 4.0)]
        if len(hit):
            latency.append(float(hit[0] - onset))

    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, max(2, len(alarms)), size=shifts)
    null = {str(delta): [] for delta in DELTAS}
    for offset in offsets:
        rolled = np.roll(alarms, int(offset))
        for delta in DELTAS:
            null[str(delta)].append(sens(rolled, delta))
    null_summary = {key: float(np.median(value)) for key, value in null.items()}

    pred = intervals_from_alarm(
        times, alarms, ma, threshold, allowed & evalmask,
    )
    tolerant = interval_metrics(pred, gt, 0.1, 0.1)
    strict = interval_metrics(pred, gt, 0.7, 0.7)
    return {
        "actual_fp_per_min": float(fp_count / (exposure / 60.0)) if exposure else None,
        "false_alarm_count": fp_count,
        "valid_idle_exposure_seconds": exposure,
        "onset_sensitivity": onset_results,
        "circular_shift_null_median": null_summary,
        "circular_shift_excess": {
            key: float(onset_results[key] - null_summary[key]) for key in onset_results
        },
        "median_latency_seconds": float(np.median(latency)) if latency else None,
        "interval_0.1": tolerant,
        "interval_0.7": strict,
        "predicted_intervals": int(len(pred)),
        "alarm_count": int(alarms.sum()),
        "anchor_ewma_mean": initial_mean,
        "anchor_ewma_sd": float(np.sqrt(initial_var)),
        "threshold": float(threshold),
    }


def run_target(subject: int, session: int, train_data: dict,
               source_data: list[dict], target_stream: np.ndarray,
               events: np.ndarray, shifts: int, seeds: tuple[int, ...],
               epochs: int, device: torch.device,
               score_cache_path: Path | None = None,
               score_cache_only: bool = False) -> dict:
    ends = _window_ends(target_stream.shape[-1])
    times = ends / SFREQ
    starts = (ends - WINDOW) / SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    allowed = np.ones(len(times), dtype=bool)
    for seam in seams:
        allowed &= ~((starts < seam + 3.0) & (times > seam))
    evalmask = times >= events[SCREEN, 2] / SFREQ
    anchor_ends = _window_ends(
        train_data["anchor_signal"].shape[-1],
        first_end=3 * SFREQ,
    )
    source_anchor_preds: list[list[np.ndarray]] = [[] for _ in source_data]
    target_anchor_preds: list[np.ndarray] = []
    target_scores: list[np.ndarray] = []
    for seed_index, seed in enumerate(seeds, start=1):
        print(
            f"  S{session} target={subject:02d}: train seed {seed_index}/{len(seeds)} "
            f"({len(source_data)} source sequences x {epochs} epochs)",
            flush=True,
        )
        model = train_source_pool(source_data, seed, epochs, device)
        target_scores.append(predict_signal(model, target_stream, device, ends=ends))
        target_anchor_preds.append(predict_signal(
            model, train_data["anchor_signal"], device, ends=anchor_ends,
        ))
        for index, source in enumerate(source_data) if not score_cache_only else ():
            source_ends = _window_ends(
                source["anchor_signal"].shape[-1], first_end=3 * SFREQ,
            )
            values = predict_signal(
                model, source["anchor_signal"], device, ends=source_ends,
            )
            keep_from = int(np.searchsorted(
                source_ends / SFREQ,
                source["anchor_midpoint_seconds"],
                side="right",
            ))
            source_anchor_preds[index].append(values[keep_from:])
            if index in (9, 19, 29, 39, 49, len(source_data) - 1):
                print(
                    f"    source-anchor scoring {index + 1}/{len(source_data)} "
                    f"for seed {seed_index}/{len(seeds)}", flush=True,
                )

    target_raw = np.mean(np.asarray(target_scores), axis=0)
    target_anchor = np.mean(np.asarray(target_anchor_preds), axis=0)
    if score_cache_path is not None:
        score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            score_cache_path,
            subject=np.asarray(subject, dtype=np.int16),
            session=np.asarray(session, dtype=np.int8),
            times=times.astype(np.float64),
            starts=starts.astype(np.float64),
            allowed=allowed.astype(np.uint8),
            events=events.astype(np.float64),
            raw_scores=target_raw.astype(np.float32),
            anchor_scores=target_anchor.astype(np.float32),
            seeds=np.asarray(seeds, dtype=np.int64),
        )
    if score_cache_only:
        return {
            "subject": subject,
            "session": session,
            "score_cache": str(score_cache_path),
            "eval_windows": int(len(times)),
        }
    source_z = [
        _zscore(np.mean(np.asarray(seed_scores), axis=0))
        for seed_scores in source_anchor_preds
    ]
    target_z = _zscore(target_anchor)
    rows: dict[str, dict] = {}
    for target_fp in TARGETS:
        transferred_thr, transferred_anchor_fp = _calibration_threshold(source_z, target_fp)
        target_thr, target_anchor_fp = _calibration_threshold([target_z], target_fp)
        rows[str(target_fp)] = {
            "source_transferred": {
                "anchor_fp_per_min": transferred_anchor_fp,
                "metrics": _eval_threshold(
                    times, target_raw, target_anchor, transferred_thr,
                    allowed, evalmask, events, shifts,
                    seed=88000 + 100 * session + subject + int(target_fp * 10),
                ),
            },
            "common_fp_target_pre_rest": {
                "anchor_fp_per_min": target_anchor_fp,
                "metrics": _eval_threshold(
                    times, target_raw, target_anchor, target_thr,
                    allowed, evalmask, events, shifts,
                    seed=99000 + 100 * session + subject + int(target_fp * 10),
                ),
            },
        }
    return {
        "subject": subject,
        "session": session,
        "source_subjects": len(source_data),
        "source_train_windows_per_subject": [
            int(len(item["train_y"])) for item in source_data
        ],
        "eval_stream_seconds": float(target_stream.shape[-1] / SFREQ),
        "eval_windows": int(len(times)),
        "guarded_windows": int((~allowed).sum()),
        "screen_trials_excluded": SCREEN,
        "metrics": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--train-cache-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    parser.add_argument("--stream-root", type=Path, default=Path("data/interim/lee2019_pseudo_uncued_raw"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260729, 20260730, 20260731])
    parser.add_argument("--max-source-windows", type=int, default=100)
    parser.add_argument(
        "--prepared-source-root", type=Path,
        default=Path("data/interim/lee2019_cnn_gru_loso_prepared_source"),
        help="Model-independent filtered source-sequence cache; set to an empty path only by editing the runner",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_cnn_gru_loso_arm_a_stream_v2_full54.json"))
    parser.add_argument("--score-cache-root", type=Path, default=None,
                        help="Optional directory for per-target raw CNN-GRU/anchor traces")
    parser.add_argument("--score-cache-only", action="store_true",
                        help="Rebuild source-pooled scores and save traces without threshold metrics")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    if args.score_cache_only and args.score_cache_root is None:
        raise ValueError("--score-cache-only requires --score-cache-root")

    rows: list[dict] = []
    partial = args.output.with_suffix(".partial.json")
    if partial.exists():
        rows = json.loads(partial.read_text(encoding="utf-8"))
    done = {(int(row["session"]), int(row["subject"])) for row in rows}

    for session in args.sessions:
        source_pool: dict[int, dict] = {}
        for source in SUBJECTS:
            source_pool[source] = _cached_source_training(
                args.data_root, args.train_cache_root, args.prepared_source_root,
                source, session, args.max_source_windows,
            )
            print(
                f"prepared contiguous source sequence S{session} sub-{source:02d} "
                f"n={len(source_pool[source]['train_y'])}", flush=True,
            )
        for target in args.subjects:
            score_cache_path = (
                args.score_cache_root / f"session{session}" / f"sub-{target:02d}.npz"
                if args.score_cache_root is not None else None
            )
            if args.score_cache_only and score_cache_path is not None and score_cache_path.exists():
                print(f"score-cache-hit exact-stream S{session} sub-{target:02d}", flush=True)
                continue
            if (session, target) in done:
                print(f"resume-skip exact-stream S{session} sub-{target:02d}", flush=True)
                continue
            target_stream, events, target_pre = _rebuild_exact_target_stream(
                args.data_root, args.stream_root, target, session,
            )
            train_data = source_pool[target]
            # The target anchor must be generated from the exact pre-rest used
            # in stream construction, rather than the source-pool copy.
            train_data = {**train_data, "anchor_signal": target_pre}
            source_data = [source_pool[s] for s in SUBJECTS if s != target]
            result = run_target(
                target, session, train_data, source_data,
                target_stream, events, args.shifts, tuple(args.seeds),
                args.epochs, device, score_cache_path, args.score_cache_only,
            )
            if args.score_cache_only:
                print(
                    f"saved exact-stream scores S{session} sub-{target:02d} "
                    f"windows={result['eval_windows']} device={device}",
                    flush=True,
                )
                del target_stream, target_pre
                continue
            rows.append(result)
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(rows), encoding="utf-8")
            done.add((session, target))
            print(
                f"completed exact Arm-A stream S{session} sub-{target:02d} "
                f"duration={result['eval_stream_seconds']:.1f}s device={device}",
                flush=True,
            )
            del target_stream, target_pre

    if args.score_cache_only:
        print(f"score caches ready under {args.score_cache_root}", flush=True)
        return

    payload = {
        "protocol": {
            "model": MODEL_PROTOCOL_NAME,
            "preprocessing": "causal 8-13/13-20/20-30 Hz filter bank; 20 sensorimotor channels; per-subject pre-rest normalization",
            "stream": "raw-rebuilt 50-trial pseudo-uncued stream; event array asserted exactly equal to locked Riemannian Arm-A stream",
            "target_rest": "pre-train rest only; target anchor initializes EWMA and target-pre-rest threshold arm",
            "decision": "EWMA H30 + MA 0.5 s + dwell 0.3 s + refractory 1 s",
            "screening": "first 10 trials excluded; evaluation starts at locked event 10 rest boundary",
            "source_sampling": f"best-balanced contiguous sequence of at most {args.max_source_windows} windows per source subject",
            "sequence_batch_size": 16,
            "training_note": "independent source sequences are batch-parallelized; hidden state is separate per source sequence and reset between batches",
            "thresholds": "source-transferred from source pre-rest anchors vs target pre-rest common-FP calibration; no evaluation labels used to set thresholds",
            "primary_metrics": "cue-relative pseudo-onset sensitivity at 0.5/1/2/3/4 s; actual FP/min; circular-shift null/excess; interval coverage/IoU/strict F1",
            "seeds": args.seeds,
            "epochs": args.epochs,
            "circular_shifts": args.shifts,
            "claim_limit": "Lee2019 cue-relative pseudo-onset; not actual self-initiated onset",
            "invalid_prior_outputs": "data/processed/lee2019_cnn_gru_loso_arm_a_* before stream_v2 used a separately randomized evaluation sequence and must not be compared or cited",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if partial.exists():
        partial.unlink()
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
