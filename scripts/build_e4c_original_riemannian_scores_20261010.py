"""Build LOSO Riemannian scores on Lee2019 original continuous EEG.

Source training is trial 0-49 of the other 53 subjects in each session.
The target contributes only its pre-rest reference/normalization. Replay
scores are used strictly for a raw-score parity check, never for E4c metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.build_lee2019_revalidation_norm_cov import covariances_at
from scripts.evaluate_lee2019_revalidation_guard_rearm_smoke import (
    SUBJECTS, fit_loso, source_domains,
)
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, WINDOW, reference, rolling_covariance,
)
from uncued_mi.riemannian import tangent_features

PARITY_ATOL = 5e-5


def parity_check(signal: np.ndarray, scaler, model, ref: np.ndarray,
                 replay_path: Path, old_scores_path: Path) -> dict:
    with np.load(replay_path, allow_pickle=False) as data:
        events = np.asarray(data["events"])
        spans = np.asarray(data["trial_source_spans_250hz"])
        replay_signal = np.asarray(data["signal"])
        ids = np.asarray(data["trial_ids"])
    with np.load(old_scores_path, allow_pickle=False) as data:
        old_times = np.asarray(data["times"], dtype=float)
        old_raw = np.asarray(data["raw_scores"], dtype=float)
    if not np.array_equal(ids, np.arange(50, 100)) or len(spans) != 50:
        raise AssertionError("replay trial ID provenance changed")
    chosen_replay_ends = []
    chosen_original_ends = []
    for j in (0, 10, 20, 30, 40):
        replay_start = int(events[j, 4])
        # The stored MI endpoint is cue+3.5 s; the copied trial ends at cue+4 s.
        replay_stop = int(events[j, 5] + 4 * SFREQ)
        source_start, source_stop = map(int, spans[j])
        if replay_stop - replay_start != source_stop - source_start:
            raise AssertionError("replay/source span duration mismatch")
        candidates = np.rint(old_times * SFREQ).astype(np.int64)
        candidates = candidates[(candidates >= replay_start + int(WINDOW * SFREQ) + 25) &
                                (candidates <= replay_stop - 25)]
        if len(candidates) < 3:
            raise AssertionError("not enough interior replay windows")
        for re in candidates[[0, len(candidates) // 2, -1]]:
            oe = int(re - replay_start + source_start)
            if oe < int(WINDOW * SFREQ) or oe > len(signal):
                raise AssertionError("mapped window outside original signal")
            if not np.allclose(replay_signal[int(re)-250:int(re)],
                               signal[oe-250:oe], atol=1e-6, rtol=0):
                raise AssertionError("replay/original EEG samples differ")
            chosen_replay_ends.append(int(re))
            chosen_original_ends.append(oe)
    cov = covariances_at(signal, np.asarray(chosen_original_ends))
    new = model.decision_function(scaler.transform(tangent_features(cov, ref)))
    lookup = {int(round(t * SFREQ)): float(value) for t, value in zip(old_times, old_raw)}
    old = np.asarray([lookup[e] for e in chosen_replay_ends])
    max_error = float(np.max(np.abs(new - old)))
    if max_error > PARITY_ATOL:
        raise AssertionError(f"raw-score replay parity failure: {max_error:.6g}")
    return {"n_windows": len(old), "max_abs_raw_score_error": max_error,
            "atol": PARITY_ATOL}


def build_one(subject: int, session: int, domains: dict[int, dict],
              input_root: Path, replay_root: Path, old_scores_root: Path,
              output_root: Path) -> dict:
    source = input_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz"
    replay = replay_root / f"session{session}" / f"sub-{subject:02d}.npz"
    old_scores = old_scores_root / f"session{session}" / f"sub-{subject:02d}.npz"
    output = output_root / f"session{session}" / f"sub-{subject:02d}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as existing:
            if (int(existing["subject"]), int(existing["session"])) != (subject, session):
                raise AssertionError("existing score cache identity mismatch")
            sources = np.asarray(existing["source_subjects"])
            if len(sources) != 53 or subject in sources:
                raise AssertionError("existing score cache training exposure mismatch")
            provenance = json.loads(str(existing["provenance_json"]))
            if provenance["target_future_used_for_fit"]:
                raise AssertionError("existing score cache fitted on future")
            return {"subject": subject, "session": session, "output": str(output),
                    "n_continuous_scores": len(existing["raw_scores"]),
                    "parity": json.loads(str(existing["parity_json"])),
                    "n_source_subjects": 53, "reused": True}
    with np.load(source, allow_pickle=False) as data:
        signal = np.asarray(data["signal"], dtype=np.float32)
        channels = np.asarray(data["channels"])
        sfreq = float(data["sfreq"])
        preprocessing = str(data["preprocessing"])
        command = np.asarray(data["command_intervals"], dtype=float)
    if sfreq != SFREQ or len(command) != 100 or signal.shape[1] != 20:
        raise AssertionError("original continuous input geometry changed")
    if "pre-rest z-score" not in preprocessing:
        raise AssertionError("original input lacks target pre-rest normalization")
    if len(domains) != 54 or set(domains) != set(SUBJECTS):
        raise AssertionError("not 54 source domains")
    if any(len(domains[s]["command"]) == 0 or len(domains[s]["idle"]) == 0
           for s in SUBJECTS if s != subject):
        raise AssertionError("empty source class")
    ref = reference(domains[subject]["pre"])
    scaler, model = fit_loso(domains, subject, ref)
    pre_times = np.arange(750, 15000 + 1, 25) / SFREQ
    pre_raw = model.decision_function(scaler.transform(
        tangent_features(domains[subject]["pre"], ref)))
    if len(pre_raw) != len(pre_times):
        raise AssertionError("target pre-rest/reference mismatch")
    replay_parity = parity_check(signal, scaler, model, ref, replay, old_scores)
    covariance, ends = rolling_covariance(signal)
    raw = model.decision_function(scaler.transform(tangent_features(covariance, ref)))
    times = ends / SFREQ
    if not np.all(np.diff(times) > 0) or not np.isclose(times[0], 1.):
        raise AssertionError("invalid continuous score grid")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output,
        subject=np.int16(subject), session=np.int8(session),
        raw_scores=raw.astype(np.float32), times=times.astype(np.float32),
        pre_raw_scores=pre_raw.astype(np.float32), pre_times=pre_times.astype(np.float32),
        channels=channels, cue_times=(command[:, 0] - .5).astype(np.float64),
        source_subjects=np.asarray([s for s in SUBJECTS if s != subject], dtype=np.int16),
        model_coef=model.coef_.astype(np.float32),
        model_intercept=model.intercept_.astype(np.float32),
        source_file=np.asarray(str(source)),
        parity_json=np.asarray(json.dumps(replay_parity, sort_keys=True)),
        provenance_json=np.asarray(json.dumps({
            "source_training": "same-session 53 other subjects, each trial 0-49 only",
            "target_model_data": "pre_train rest reference and normalization only",
            "target_future_used_for_fit": False,
            "preprocessing": preprocessing,
            "window_s": WINDOW, "step_s": .1,
        }, sort_keys=True)))
    return {"subject": subject, "session": session, "output": str(output),
            "n_continuous_scores": len(raw), "parity": replay_parity,
            "n_source_subjects": 53}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--input-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    ap.add_argument("--cov-root", type=Path, default=Path("data/interim/lee2019_revalidation_norm_cov"))
    ap.add_argument("--replay-root", type=Path, default=Path("data/interim/lee2019_revalidation_chronological"))
    ap.add_argument("--old-scores-root", type=Path, default=Path("data/interim/lee2019_revalidation_riemannian_scores"))
    ap.add_argument("--output-root", type=Path, default=Path("data/interim/lee2019_e4c_riemannian_original"))
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/e4c_riemannian_original_score_manifest_20261010.json"))
    ap.add_argument("--limit-cases", type=int, default=0)
    args = ap.parse_args()
    rows = []
    for session in args.sessions:
        domains = source_domains(session, args.cov_root)
        for subject in args.subjects:
            rows.append(build_one(subject, session, domains, args.input_root,
                                  args.replay_root, args.old_scores_root,
                                  args.output_root))
            print(f"E4c score S{session} sub-{subject:02d}: {len(rows)}", flush=True)
            if args.limit_cases and len(rows) >= args.limit_cases:
                print("limit-cases reached; partial diagnostic output only")
                return
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    if len(rows) != 108:
        raise AssertionError("full manifest requires 108 scores")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"status": "E4c score cache built, not E4c FP result",
                                         "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
