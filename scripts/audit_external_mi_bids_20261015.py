"""Inventory Dreyer/WBCIC BIDS files without inventing task or idle labels.

This is a G0 *preflight*, not a training-data generator. In particular, gaps
between events are never counted as idle. Original-source event parity and
protocol timing still require a separate, documented review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


EEG_SUFFIXES = (".edf", ".bdf", ".vhdr", ".set", ".fif", ".gdf", ".cnt")
SUBJECT_RE = re.compile(r"(?:^|[/_])sub-([^/_]+)")
SESSION_RE = re.compile(r"(?:^|[/_])ses-([^/_]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _subject(relative: str) -> str | None:
    match = SUBJECT_RE.search(relative.replace("\\", "/"))
    return match.group(1) if match else None


def _event_summary(path: Path) -> dict:
    rows = read_tsv(path)
    issues: list[str] = []
    types: Counter[str] = Counter()
    onset_values: list[float] = []
    duration_values: list[float] = []
    missing_duration = 0
    for index, row in enumerate(rows, 2):
        types[row.get("trial_type") or row.get("event_type") or "<untyped>"] += 1
        try:
            onset = float(row["onset"])
            if not math.isfinite(onset) or onset < 0:
                raise ValueError("non-finite or negative")
            onset_values.append(onset)
        except (KeyError, TypeError, ValueError):
            issues.append(f"line {index}: invalid onset")
        raw_duration = row.get("duration", "")
        if raw_duration in ("", "n/a", "N/A"):
            missing_duration += 1
        else:
            try:
                duration = float(raw_duration)
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError("non-finite or negative")
                duration_values.append(duration)
            except (TypeError, ValueError):
                issues.append(f"line {index}: invalid duration")
    if len(onset_values) == len(rows) and onset_values != sorted(onset_values):
        issues.append("events are not chronological in file order")
    gaps = [b - a for a, b in zip(onset_values, onset_values[1:])]
    return {
        "n_events": len(rows),
        "event_type_counts": dict(sorted(types.items())),
        "first_onset_s": min(onset_values) if onset_values else None,
        "last_onset_s": max(onset_values) if onset_values else None,
        "min_interevent_onset_gap_s": min(gaps) if gaps else None,
        "duration_values_s": sorted(set(duration_values)),
        "n_missing_duration": missing_duration,
        "issues": issues,
        "sha256": sha256(path),
    }


def _sidecar(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"path": str(path), "error": str(exc)}
    return {
        "path": str(path),
        "sha256": sha256(path),
        "SamplingFrequency": data.get("SamplingFrequency"),
        "EEGChannelCount": data.get("EEGChannelCount"),
        "RecordingDuration": data.get("RecordingDuration"),
        "EEGReference": data.get("EEGReference"),
        "PowerLineFrequency": data.get("PowerLineFrequency"),
    }


def audit(root: Path, dataset: str, max_events_files: int | None = None,
          hash_eeg: bool = False) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    event_files = sorted(root.rglob("*_events.tsv"))
    if not event_files:
        raise ValueError(f"no *_events.tsv files found under {root}; data may be absent")
    selected = event_files[:max_events_files] if max_events_files is not None else event_files
    participants_path = root / "participants.tsv"
    participants = read_tsv(participants_path) if participants_path.exists() else []
    participant_ids = [row.get("participant_id", "") for row in participants]
    runs = []
    all_subjects: set[str] = set()
    all_types: Counter[str] = Counter()
    missing_eeg = 0
    for events in selected:
        relative = events.relative_to(root).as_posix()
        stem = events.name.removesuffix("_events.tsv")
        eeg_files = [events.with_name(stem + "_eeg" + suffix)
                     for suffix in EEG_SUFFIXES]
        eeg_files = [path for path in eeg_files if path.exists()]
        sidecar_path = events.with_name(stem + "_eeg.json")
        channels_path = events.with_name(stem + "_channels.tsv")
        summary = _event_summary(events)
        all_types.update(summary["event_type_counts"])
        subject = _subject(relative)
        session_match = SESSION_RE.search(relative)
        session = session_match.group(1) if session_match else None
        if subject is not None:
            all_subjects.add(subject)
        if not eeg_files:
            missing_eeg += 1
        eeg_inventory = [{
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path) if hash_eeg else None,
            "possibly_annex_pointer": path.is_symlink() or path.stat().st_size < 1024,
        } for path in eeg_files]
        channels = read_tsv(channels_path) if channels_path.exists() else []
        runs.append({
            "events_path": relative,
            "subject": subject,
            "session": session,
            "events": summary,
            "eeg_files": eeg_inventory,
            "eeg_sidecar": _sidecar(sidecar_path if sidecar_path.exists() else None),
            "channels_path": channels_path.relative_to(root).as_posix() if channels else None,
            "channels_sha256": sha256(channels_path) if channels else None,
            "n_channels": len(channels) if channels else None,
            "channel_types": dict(Counter(row.get("type", "<missing>") for row in channels)),
            "channel_units": dict(Counter(row.get("units", "<missing>") for row in channels)),
        })
    by_subject: dict[str, int] = defaultdict(int)
    by_subject_session: dict[str, int] = defaultdict(int)
    for run in runs:
        if run["subject"] is not None:
            by_subject[run["subject"]] += 1
            by_subject_session[f"{run['subject']}|{run['session'] or '<none>'}"] += 1
    participant_subjects = {value.removeprefix("sub-") for value in participant_ids if value}
    return {
        "dataset": dataset,
        "dataset_root": str(root),
        "scope": "full" if max_events_files is None else "partial",
        "n_events_files_found": len(event_files),
        "n_events_files_audited": len(selected),
        "n_subjects_in_events": len(all_subjects),
        "n_participant_rows": len(participants),
        "duplicate_participant_ids": sorted({item for item in participant_ids if participant_ids.count(item) > 1}),
        "subjects_missing_from_participants": sorted(all_subjects - participant_subjects),
        "participants_without_events": sorted(participant_subjects - all_subjects),
        "event_types_all_runs": dict(sorted(all_types.items())),
        "run_count_by_subject": dict(sorted(by_subject.items())),
        "run_count_by_subject_session": dict(sorted(by_subject_session.items())),
        "n_runs_without_eeg_file": missing_eeg,
        "runs": runs,
        "training_ready": False,
        "unresolved_gates": [
            "original raw event-code/time parity with this BIDS conversion",
            "explicit cue/MI/feedback and positive-label support, including 2-s trailing window",
            "explicit idle labels and unique continuous idle support; unlabelled gaps are not idle",
            "raw acquisition order, channel units/montage and EEG-file completeness",
            "subject/session overlap across datasets and official checkpoint pretraining sources",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dreyer", "wbcic"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events-files", type=int, default=None,
                        help="Development smoke only; output is marked partial")
    parser.add_argument("--hash-eeg", action="store_true",
                        help="Read every EEG byte to SHA-256; slow on full datasets")
    args = parser.parse_args()
    if args.max_events_files is not None and args.max_events_files < 1:
        parser.error("--max-events-files must be positive")
    result = audit(args.dataset_root, args.dataset, args.max_events_files, args.hash_eeg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.dataset}: {result['n_events_files_audited']}/{result['n_events_files_found']} "
          f"event files, {result['n_subjects_in_events']} subjects, "
          f"{result['n_runs_without_eeg_file']} runs lacking local EEG; "
          f"training_ready={result['training_ready']}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
