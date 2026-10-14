"""Second-pass audit: artifact census, code dependencies, decoder controls.

No historical outputs are overwritten and no candidate models are trained.
Static import/call listings identify impact candidates, not proven effect sizes.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import evaluate_lee2019_riemannian_p0_ceiling as p0

OUT = ROOT / "data/processed/research_audit_20261001/followup"


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def census():
    paths = set(ROOT.glob("*.md"))
    for folder in ("scripts", "src", "tests", "openvibe_project"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix.lower() in (".py", ".md") and not any(
                part in {"__pycache__", ".venv", "data", "logs"}
                for part in path.relative_to(ROOT).parts
            ):
                paths.add(path)
    files, dependencies, syntax_errors = [], [], []
    for path in sorted(paths):
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        relative = path.relative_to(ROOT).as_posix()
        files.append({"path": relative, "bytes": len(raw), "lines": len(text.splitlines()),
                      "sha256": hashlib.sha256(raw).hexdigest()})
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(text, filename=relative)
        except SyntaxError as error:
            syntax_errors.append({"path": relative, "error": str(error)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "evaluate_lee2019_loso_deployment" in module or (
                    module == "scripts" and any(a.name == "evaluate_lee2019_loso_deployment" for a in node.names)
                ):
                    dependencies.append({"path": relative, "line": node.lineno,
                        "kind": "import", "detail": ast.get_source_segment(text, node)})
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else "")
                if name == "prepare_target":
                    dependencies.append({"path": relative, "line": node.lineno,
                        "kind": "prepare_target_call", "detail": ast.get_source_segment(text, node)})
    write_csv("expanded_file_inventory.csv", files)
    write_json("shared_preparation_dependencies.json", dependencies)
    write_json("syntax_errors.json", syntax_errors)

    documents = []
    for path in sorted(ROOT.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        documents.append({"document": path.name, "lines": len(text.splitlines()),
            "headings": " | ".join(re.findall(r"^#{1,2} (.+)$", text, re.M)),
            "audit_notice_present": "RESEARCH_PROCESS_SIMULATION_AUDIT_20261001.md" in text,
            "scope": "inventory, not independent verification of every numerical claim"})
    write_csv("root_document_register.csv", documents)

    artifacts = []
    # Snapshot before writing this run's artifact index; exclude this output directory.
    json_paths = sorted(p for p in (ROOT / "data/processed").rglob("*.json") if OUT not in p.parents)
    for path in json_paths:
        constants = []
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"),
                parse_constant=lambda token: constants.append(token) or None)
            status, error = "parsed", ""
            keys = "|".join(value.keys()) if isinstance(value, dict) else type(value).__name__
        except (ValueError, UnicodeError) as exc:
            status, error, keys = "parse_error", str(exc), ""
        artifacts.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "status": status, "nonstandard_numeric_tokens": len(constants),
            "top_level_keys": keys, "error": error})
    write_csv("processed_json_integrity.csv", artifacts)
    return {"files": len(files), "root_documents": len(documents),
        "python_files": sum(p.suffix == ".py" for p in paths),
        "syntax_errors": syntax_errors, "processed_json_files": len(artifacts),
        "json_parse_errors": [r for r in artifacts if r["status"] != "parsed"],
        "json_with_nonstandard_numbers": sum(r["nonstandard_numeric_tokens"] > 0 for r in artifacts),
        "import_paths": sorted({r["path"] for r in dependencies if r["kind"] == "import"}),
        "prepare_target_call_paths": sorted({r["path"] for r in dependencies if r["kind"] == "prepare_target_call"})}


def records_for(subject, session):
    path = ROOT / f"data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{subject:02d}.npz"
    with np.load(path, allow_pickle=False) as data:
        events, n = data["events"], len(data["state"])
    times = np.arange(250, n + 1, 25) / 250.
    starts = times - 1.
    allowed = np.ones(len(times), bool)
    for seam in np.unique(np.r_[events[:, 4], events[1:, 2]]) / 250.:
        allowed &= ~((starts < seam + 3.) & (times > seam))
    records = []
    for heldout in np.array_split(np.arange(10, 50), 5):
        keep = p0.cell_mask(times, events, heldout, times[-1] + .1)
        ft, fa = times[keep], allowed[keep]
        onsets = events[heldout, 5] / 250. + .5
        state = np.zeros(len(ft), bool)
        for onset in onsets:
            state |= (ft >= onset) & (ft <= onset + 3.5)
        records.append({"times": ft, "allowed": fa, "idle": fa & ~state,
                        "ma": np.ones(len(ft)), "onsets": onsets})
    return records


def controls():
    rows = []
    for session in (1, 2):
        for subject in range(1, 55):
            records = records_for(subject, session)
            baseline = p0.alarms_for(records, 0.)
            # Causal decoder runs across the guard, but output is excluded there.
            # This is a mechanistic ablation, not a proposed deployable policy.
            no_reset = [p0.alarms_from_ma(r["ma"], 0., np.ones(len(r["ma"]), bool),
                dwell=p0.DWELL_WINDOWS, refractory=p0.REFRACTORY_WINDOWS) & r["allowed"] for r in records]
            # A score rotation cannot change a constant; preserve guard and re-decode.
            shifted_records = [{**r, "ma": np.roll(r["ma"], 173)} for r in records]
            score_null = p0.sensitivity(records, p0.alarms_for(shifted_records, 0.), 1.)
            choices = {}
            for target in (2., 4.):
                threshold, alarms, actual = p0.choose_threshold(records, target)
                choices[str(target)] = {"threshold": threshold, "actual_fp": actual,
                    "sen_1s": p0.sensitivity(records, alarms, 1.)}
            rows.append({"subject": subject, "session": session,
                "constant_guard_reset_sen": p0.sensitivity(records, baseline, 1.),
                "constant_guard_reset_fp": p0.fp_rate(records, baseline),
                "constant_no_guard_reset_sen": p0.sensitivity(records, no_reset, 1.),
                "constant_no_guard_reset_fp": p0.fp_rate(records, no_reset),
                "constant_score_shift_redecode_null_sen": score_null,
                "constant_score_shift_excess": p0.sensitivity(records, baseline, 1.) - score_null,
                "nearest_target_choices": choices})
        print(f"controls S{session}: 54/54", flush=True)
    write_json("guard_ablation_controls.json", rows)
    return {"n": len(rows), **{k: [min(r[k] for r in rows), max(r[k] for r in rows)]
        for k in ("constant_guard_reset_sen", "constant_no_guard_reset_sen",
                  "constant_score_shift_excess", "constant_no_guard_reset_fp")},
        "fp2_constant_choice_sen_counts": dict(Counter(r["nearest_target_choices"]["2.0"]["sen_1s"] for r in rows)),
        "fp4_constant_choice_sen_counts": dict(Counter(r["nearest_target_choices"]["4.0"]["sen_1s"] for r in rows))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"census": census(), "decoder_controls": controls(),
        "limits": ["No model performance rerun or new training", "Static calls do not imply equal impact",
            "Constant-score shift invariance does not validate the null for nonconstant scores",
            "JSON readability does not establish scientific validity"]}
    write_json("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
