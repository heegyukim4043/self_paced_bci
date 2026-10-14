"""Read-only scientific audit of historical artifacts; writes only new audit files.

Does not replace historical model outputs or fit new performance candidates.
Source IDs are reconstructed from the exact frozen rest partition RNG and
verified against cached event lengths/source IDs, not inferred from class labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import evaluate_lee2019_riemannian_p0_ceiling as p0
from uncued_mi.pseudo_uncued import fixed_budget_exponential_chunks
from uncued_mi.riemannian import tangent_features

OUT = ROOT / "data/processed/research_audit_20261001"


def inventory():
    paths = sorted(set(ROOT.glob("*.md")) | set((ROOT / "scripts").glob("*.py"))
                   | set((ROOT / "src").rglob("*.py")) | set((ROOT / "tests").glob("*.py")))
    rows = []
    for path in paths:
        content = path.read_bytes()
        rows.append({"path": str(path.relative_to(ROOT)), "bytes": len(content),
                     "sha256": hashlib.sha256(content).hexdigest(),
                     "lines": len(content.splitlines())})
    with (OUT / "file_inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = (ROOT / "progress.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    headings = [(i, line) for i, line in enumerate(lines) if re.match(r"^#{1,4} ", line)]
    sections = []
    for index, (start, heading) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        body = "\n".join(lines[start + 1:end])
        paths_in_section = sorted(set(re.findall(r"(?:scripts|src|tests|data/(?:processed|interim))/[\w./{}*,-]+", body)))
        refs = [{"path": path, "exists": (ROOT / path).exists(),
                 "literal": not any(c in path for c in "{}*,")} for path in paths_in_section]
        sections.append({"line": start + 1, "heading": heading, "body": body, "references": refs})
    (OUT / "progress_section_inventory.json").write_text(
        json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")
    documents = []
    for path in sorted(ROOT.glob("*.md")):
        if path.name == "progress.md":
            continue
        text = path.read_text(encoding="utf-8-sig")
        claims = [{"line": i+1, "text": line} for i, line in enumerate(text.splitlines())
                  if re.search(r"conclu|decision|supersed|passed|failed|판정|결론|기각|확정|성공|실패|한계", line, re.I)]
        documents.append({"path": path.name, "claim_lines": claims})
    (OUT / "document_claim_inventory.json").write_text(
        json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"files": len(rows), "root_markdown": len(list(ROOT.glob("*.md"))),
            "progress_lines": len(lines), "progress_sections": len(sections),
            "missing_literal_references": sum(not ref["exists"] and ref["literal"]
                for section in sections for ref in section["references"])}


def source_map(subject, session, events, sources, metadata, n_samples):
    # Each repartition covers each original rest block fully.
    block_lengths = [int(sum(row[3]-row[2] for row, source in zip(events[:10], sources[:10])
                             if int(source) == block)) for block in range(3)]
    blocks, cursor = [], 0
    for length in block_lengths:
        blocks.append(np.arange(cursor, cursor+length, dtype=np.int64))
        cursor += length
    source_ids = np.full(n_samples, -1, dtype=np.int64)
    base_seed = int(metadata["seed"]) - 1000*session - subject
    for part in range(5):
        chunks, ids = fixed_budget_exponential_chunks(
            blocks, chunks_per_block=(3,3,4), sfreq=250., minimum_seconds=5.,
            seed=base_seed+2000*session+100*part+subject)
        for offset, (chunk, source) in enumerate(zip(chunks, ids)):
            i = 10*part + offset
            start, stop = events[i, 2:4]
            assert stop-start == len(chunk) and int(sources[i]) == int(source)
            source_ids[start:stop] = chunk
    return source_ids, cursor


def support_ids(source_ids, ends):
    support = np.zeros(len(source_ids)+1, dtype=np.int32)
    np.add.at(support, ends-250, 1)
    np.add.at(support, ends, -1)
    active = np.cumsum(support[:-1]) > 0
    return np.unique(source_ids[active & (source_ids >= 0)])


def provenance(subject, session, verify_signal=False):
    path = ROOT / f"data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{subject:02d}.npz"
    with np.load(path, allow_pickle=False) as data:
        events = data["events"]
        sources = data["rest_source_ids"]
        metadata = json.loads(str(data["metadata_json"]))
        n = len(data["state"])
        signal = data["signal"] if verify_signal else None
    ids, unique_count = source_map(subject, session, events, sources, metadata, n)
    ends = np.arange(250, n+1, 25)
    times, starts = ends/250., (ends-250)/250.
    seams = np.unique(np.r_[events[:,4], events[1:,2]])/250.
    allowed = np.ones(len(ends), bool)
    for seam in seams:
        allowed &= ~((starts < seam+3.) & (times > seam))
    folds = list(np.array_split(np.arange(10,50),5))
    overlap, constant_records, earliest = [], [], []
    for i, heldout in enumerate(folds):
        training = np.concatenate([fold for j, fold in enumerate(folds) if i != j])
        trainmask = p0.interval_mask(times, starts, events, training, 2,3) & allowed
        heldmask = p0.interval_mask(times, starts, events, heldout, 2,3) & allowed
        selected_train = p0.evenly(np.flatnonzero(trainmask))
        seen = support_ids(ids, ends[selected_train])
        held = support_ids(ids, ends[heldmask])
        overlap.append(float(len(np.intersect1d(seen, held))/len(held)))
        cmask = p0.cell_mask(times, events, heldout, times[-1]+.1)
        ft, fa = times[cmask], allowed[cmask]
        onset = events[heldout,5]/250.+.5
        gt = np.c_[onset, events[heldout,5]/250.+4.]
        state = np.zeros(len(ft), bool)
        for start, stop in gt:
            state |= (ft>=start) & (ft<=stop)
            eligible = ft[fa & (ft >= start-0.5) & (ft<=stop)]
            earliest.append(float(eligible[0]-start))
        constant_records.append({"times":ft,"allowed":fa,"idle":fa & ~state,
                                 "ma":np.ones(len(ft)),"onsets":onset})
    alarms = p0.alarms_for(constant_records, 0.)
    null = p0.null_distribution(constant_records, alarms, 1., 500, 20261001+100*session+subject)
    same = None
    if signal is not None:
        rest_positions = np.flatnonzero(ids>=0)
        order = rest_positions[np.argsort(ids[rest_positions], kind="stable")]
        assert len(order) == unique_count*5
        repeated = signal[order].reshape(unique_count,5,-1)
        same = bool(np.all(repeated == repeated[:,:1,:]))
    return {"subject":subject,"session":session,"original_rest_seconds":unique_count/250.,
            "replayed_rest_seconds":float(np.sum(ids>=0)/250.),
            "rest_source_sample_overlap_fraction_by_oof_fold":overlap,
            "repeated_raw_rest_samples_exactly_equal":same,
            "guard_earliest_eligible_seconds_after_pseudo_onset": [min(earliest),max(earliest)],
            "constant_score_control":{"sen_1s":p0.sensitivity(constant_records,alarms,1.),
                "null_1s_median":float(np.median(null)),
                "excess_1s":p0.sensitivity(constant_records,alarms,1.)-float(np.median(null)),
                "fp_per_min":p0.fp_rate(constant_records,alarms),
                "scope":"structural counterexample; constant ma=1 and fixed threshold=0; not FP matched"}}


def scale_probe(subject, session):
    with np.load(ROOT / f"data/interim/lee2019_four_rest/session{session}/sub-{subject:02d}.npz") as data:
        pre = data["pre_train_covariance"]
        ref = p0.base.reference(pre)
        cmd = data["train_command_covariance"][data["train_command_trial"] < 50]
        idle = data["train_idle_iti_covariance"][data["train_idle_iti_trial"] < 50]
    ac, sc, events, times, starts, allowed = p0.base.prepare_target(subject,session)
    first = p0.fit_model(tangent_features(cmd, ref), tangent_features(idle, ref))
    sf, af = tangent_features(sc,ref), tangent_features(ac,ref)
    records = p0.make_fold_records(sf,af,events,times,starts,allowed,"eval_oof_oracle",None)
    vals = np.concatenate([r["ma"][r["allowed"] & np.isfinite(r["ma"])] for r in records])
    thresholds = np.unique(np.r_[np.quantile(vals,np.linspace(0,1,321)),
                                 vals.max()+max(1e-8,np.std(vals)*1e-6)])[::-1]
    rates = np.array([p0.fp_rate(records,p0.alarms_for(records,t)) for t in thresholds])
    peak = int(np.argmax(rates))
    choices=[]
    for target in (2.,4.):
        idx=min(range(len(thresholds)),key=lambda j:(abs(rates[j]-target),-thresholds[j]))
        choices.append({"target_fp":target,"threshold":float(thresholds[idx]),
            "actual_fp":float(rates[idx]),"index_high_to_low":idx,
            "first_maximum_fp_index":peak,"past_first_fp_peak":bool(idx>peak)})
    scaler,model=first
    trainx=np.r_[tangent_features(cmd,ref),tangent_features(idle,ref)]
    mask=p0.interval_mask(times,starts,events,np.arange(10,50),6,7)&allowed
    probe=scaler.transform(sf[mask])
    return {"subject":subject,"session":session,
            "original_anchor_median_diagonal":float(np.median(np.diagonal(pre,axis1=1,axis2=2))),
            "normalized_anchor_median_diagonal":float(np.median(np.diagonal(ac,axis1=1,axis2=2))),
            "train_feature_abs_z_median":float(np.median(np.abs(scaler.transform(trainx)))),
            "evaluation_mi_feature_abs_z_median":float(np.median(np.abs(probe))),
            "oof_threshold_branch_probes":choices}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inv=inventory()
    print("inventory",json.dumps(inv),flush=True)
    rows=[]
    for session in (1,2):
        for subject in range(1,55):
            rows.append(provenance(subject,session,verify_signal=subject in (2,3,54)))
        print(f"provenance S{session} 54/54",flush=True)
    (OUT/"simulation_provenance_controls.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    probes=[]
    for session in (1,2):
        for subject in (2,3,54):
            probes.append(scale_probe(subject,session))
            print("scale/branch probe",subject,session,flush=True)
    (OUT/"p0_scale_branch_probes.json").write_text(json.dumps(probes,indent=2),encoding="utf-8")
    summary={"inventory":inv,"n_provenance_sessions":len(rows),
       "overlap_fraction_range":[float(np.min([v for r in rows for v in r['rest_source_sample_overlap_fraction_by_oof_fold']])),
                                 float(np.max([v for r in rows for v in r['rest_source_sample_overlap_fraction_by_oof_fold']]))],
       "constant_sen_range":[min(r['constant_score_control']['sen_1s'] for r in rows),max(r['constant_score_control']['sen_1s'] for r in rows)],
       "constant_excess_range":[min(r['constant_score_control']['excess_1s'] for r in rows),max(r['constant_score_control']['excess_1s'] for r in rows)],
       "constant_fp_range":[min(r['constant_score_control']['fp_per_min'] for r in rows),max(r['constant_score_control']['fp_per_min'] for r in rows)],
       "probes":probes}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
