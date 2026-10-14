from __future__ import annotations

import csv
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path("data/processed/direct_collection_literature_review_20260924")
SOURCE_DIR = ROOT / "sources"


SELECTED = {
    "10.3389/fnhum.2025.1540155": {
        "domain": "Onset reference",
        "role": "Direct, executed movement",
        "finding": "Finger motion capture; speed-threshold first crossing defined movement onset. Cue-trained decoder was evaluated on self-paced movement.",
        "design_implication": "Supports kinematic onset for overt control trials, not covert-MI intention onset.",
        "appraisal": "Small (10 enrolled; 1 excluded); offline; participant-specific threshold tuning on evaluation stream limits deployment inference.",
    },
    "10.3389/fneng.2012.00013": {
        "domain": "Onset reference",
        "role": "Direct, self-paced reaching",
        "finding": "Readiness-potential detector anticipated actual movement onset by about 0.5 s in 12 participants.",
        "design_implication": "Supports continuous behavioral onset reference and pre-onset EEG analysis.",
        "appraisal": "Small exploratory study; mostly healthy participants; executed movement rather than covert MI.",
    },
    "10.1109/TNSRE.2018.2855053": {
        "domain": "Onset reference",
        "role": "Direct, self-initiated movement",
        "finding": "Continuous classification and asynchronous detection were tested during self-initiated ankle movement.",
        "design_implication": "Supports event-level continuous evaluation around behavioral onset.",
        "appraisal": "Ten healthy participants; abstract does not establish a covert-MI onset reference.",
    },
    "10.1088/1741-2552/ac0b52": {
        "domain": "Onset reference",
        "role": "Direct, natural reach-and-grasp",
        "finding": "Self-initiated reach-and-grasp was time-locked to saccade onset; oculomotor-only controls exposed false detections.",
        "design_implication": "A saccade can provide timing but is also a physiological confound; it should not be the primary MI-onset marker.",
        "appraisal": "Naturalistic and online, but gaze onset is a proxy and not an independent intention timestamp.",
    },
    "10.1109/TNSRE.2004.827220": {
        "domain": "Continuous evaluation",
        "role": "Foundational asynchronous evaluation",
        "finding": "Defined asynchronous use as continuous analysis, maximizing hits during intention while minimizing rest false detections; used dwell and refractory rules.",
        "design_implication": "Supports separate onset sensitivity and idle false-alarm accounting.",
        "appraisal": "Simulated asynchronous use from cue-based MI; not a live self-initiated validation.",
    },
    "10.3389/fnins.2012.00007": {
        "domain": "Continuous evaluation",
        "role": "Continuous self-paced benchmark",
        "finding": "BCI Competition IV Dataset I framed MI versus non-control as continuous, nonstationary time-series prediction.",
        "design_implication": "Supports uninterrupted streams, but sample-wise prediction error is not interchangeable with FP/min and event sensitivity.",
        "appraisal": "Benchmark method paper; competition labels and metric do not fully represent prospective deployment.",
    },
    "10.3389/fnbot.2022.971547": {
        "domain": "Continuous evaluation",
        "role": "Asynchronous clinical pilot",
        "finding": "A four-channel two-phase detector targeted false-positive rejection and required recalibration for day-to-day variation.",
        "design_implication": "Supports explicit non-control conditions and prospective recalibration audits.",
        "appraisal": "Pilot sample (8 healthy, 9 stroke); reports FP rate rather than FP/min, limiting numerical comparison.",
    },
    "10.1186/s12984-020-0651-4": {
        "domain": "Continuous evaluation",
        "role": "Online real-life self-paced comparison",
        "finding": "Onset detection was evaluated during video viewing, reading and image browsing rather than quiet rest alone.",
        "design_implication": "Supports realistic non-control exposures and activity-stratified false-alarm reporting.",
        "appraisal": "Task and reported composite metrics differ from the proposed MI detector; not a direct performance comparator.",
    },
    "10.1088/1741-2560/12/3/036007": {
        "domain": "Session variability",
        "role": "Cross-session asynchronous walking",
        "finding": "Self-initiated walking pre-movement detection was studied without session-to-session recalibration.",
        "design_implication": "Supports explicit cross-session transfer testing rather than assuming stationarity.",
        "appraisal": "Walking execution and low-frequency correlates differ from covert hand MI.",
    },
    "10.1109/TNSRE.2017.2778178": {
        "domain": "Session variability",
        "role": "Inter-session MI variability",
        "finding": "Demonstrated substantial inter-subject and day-to-day changes in MI EEG dynamics and BCI performance.",
        "design_implication": "Supports repeated sessions and reporting within- versus between-subject/session effects.",
        "appraisal": "Multiclass cue-based MI; exact implications for continuous false alarms require validation.",
    },
    "10.1007/s10548-015-0429-3": {
        "domain": "Session variability",
        "role": "Resting-state session predictor",
        "finding": "C3 resting-state spectral entropy predicted later SMR-BCI aptitude in a returning subset.",
        "design_implication": "Supports collecting standardized pre-session rest and preregistering usability prediction separately from detector evaluation.",
        "appraisal": "40 in session 1, only 26 returned; single predictor and cue-based performance limit generalization.",
    },
    "10.1088/1741-2552/ad7a24": {
        "domain": "Session variability",
        "role": "Longitudinal multiverse analysis",
        "finding": "Across 62 participants and up to 11 sessions, mu-rhythm SNR related to session-wise BCI performance more robustly than connectivity.",
        "design_implication": "Strongly supports multi-session acquisition and session-state covariates.",
        "appraisal": "Secondary analysis of cue-based sensorimotor BCI; does not set a required number or spacing of sessions.",
    },
    "10.1016/j.biopsycho.2013.05.005": {
        "domain": "Feedback control",
        "role": "Playback-sham neurofeedback",
        "finding": "Real feedback (n=9) was compared with another person's playback (n=8) over eight sessions and produced different cortical activation patterns.",
        "design_implication": "Shows yoked/playback sham is feasible but not physiologically inert.",
        "appraisal": "NIRS neurofeedback efficacy study, not asynchronous EEG detector validation; very small groups.",
    },
    "10.1088/1741-2552/ac456d": {
        "domain": "Feedback control",
        "role": "No/sham/active feedback comparison",
        "finding": "Thirty healthy participants received no, false or actual feedback across repeated MI practice sessions.",
        "design_implication": "Feedback condition changes the estimand and can alter learned brain dynamics; a sham arm should be a separate mechanistic study.",
        "appraisal": "Connectivity outcome and training context differ from detector audit outcomes.",
    },
    "10.1093/cercor/bhac525": {
        "domain": "Feedback control",
        "role": "Double-blind EEG sham control",
        "finding": "Real SMR-ERD feedback versus other-participant feedback produced different learning and structural-change patterns after 30 min.",
        "design_implication": "Reinforces that playback sham is an active experimental manipulation, not a neutral add-on.",
        "appraisal": "Short neurofeedback intervention; structural outcomes do not quantify asynchronous detection validity.",
    },
    "10.1186/s12984-025-01597-w": {
        "domain": "Feedback control",
        "role": "False-positive feedback consequence",
        "finding": "Asynchronous MI-BCI feedback with versus without false positives altered cortical/hemodynamic responses.",
        "design_implication": "Supports treating false alarms as scientifically consequential and reporting exposure-specific FP burden.",
        "appraisal": "Small sample (12 healthy, 4 stroke); mechanism study rather than an estimator-validation study.",
    },
    "10.1002/14651858.CD015065.pub2": {
        "domain": "Feedback control",
        "role": "Contextual systematic review",
        "finding": "BCI versus sham evidence for post-stroke motor recovery remained uncertain, with blinding and allocation limitations.",
        "design_implication": "Supports caution: adding sham requires adequate power, blinding and a distinct causal question.",
        "appraisal": "Secondary clinical evidence; not specific to self-paced onset detection.",
    },
}


def normalize_doi(value: str | None) -> str:
    return (value or "").strip().lower().removeprefix("https://doi.org/")


def collect_records() -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in SOURCE_DIR.glob("pubmed_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data.get("records", []):
            doi = normalize_doi(record.get("doi"))
            if doi and doi not in records:
                records[doi] = record
    for path in SOURCE_DIR.glob("europepmc_*_narrow.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data.get("records", []):
            doi = normalize_doi(record.get("doi"))
            if not doi or doi in records:
                continue
            records[doi] = {
                "pmid": record.get("pmid"),
                "title": record.get("title"),
                "abstract": None,
                "authors": [record.get("authorString")],
                "journal": record.get("journalTitle"),
                "year": record.get("pubYear"),
                "doi": record.get("doi"),
                "pmcid": None,
                "publication_types": [record.get("pubType")],
            }
    records["10.3389/fnhum.2025.1540155"] = {
        "pmid": "40007882",
        "pmcid": "PMC11850329",
        "title": "A novel paradigm for fast training data generation in asynchronous movement-based BCIs.",
        "authors": ["Crell MR", "Kostoglou K", "Sterk K", "Müller-Putz GR"],
        "journal": "Frontiers in Human Neuroscience",
        "year": "2025",
        "doi": "10.3389/fnhum.2025.1540155",
        "abstract": None,
        "publication_types": ["Journal Article"],
    }
    return records


def build_rows() -> list[dict]:
    records = collect_records()
    rows = []
    missing = []
    for doi, evidence in SELECTED.items():
        canonical_doi = normalize_doi(doi)
        record = records.get(canonical_doi)
        if record is None:
            missing.append(doi)
            continue
        authors = record.get("authors") or []
        row = {
            "domain": evidence["domain"],
            "year": record.get("year"),
            "first_author": (authors[0] if authors else ""),
            "title": record.get("title"),
            "doi": canonical_doi,
            "pmid": record.get("pmid"),
            "role": evidence["role"],
            "finding": evidence["finding"],
            "design_implication": evidence["design_implication"],
            "appraisal": evidence["appraisal"],
        }
        rows.append(row)
    if missing:
        raise RuntimeError(f"Selected DOI records missing: {missing}")
    return rows


def write_outputs(rows: list[dict]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "selected_studies.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (ROOT / "selected_studies.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    queries = []
    for path in sorted(SOURCE_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "query" not in data:
            continue
        reconciliation = data.get("reconciliation", {})
        queries.append(
            {
                "file": path.name,
                "database": data.get("database") or data.get("api"),
                "access_date": data.get("access_date"),
                "query": data.get("query"),
                "expected_total": data.get("expected_total", reconciliation.get("expected_total")),
                "retrieved_total": data.get("retrieved_total", reconciliation.get("retrieved_total")),
                "complete": data.get("complete", reconciliation.get("complete")),
            }
        )
    (ROOT / "search_log.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def plot_evidence_map(rows: list[dict]) -> None:
    domain_text = {
        "Onset reference": (
            "Behavioral onset can be anchored\nwith kinematics/EMG for overt controls.\n"
            "It does not identify covert-MI\nintention onset."
        ),
        "Continuous evaluation": (
            "Continuous streams require both\nonset hits and idle false alarms.\n"
            "Sample/window accuracy alone\nis not deployment evidence."
        ),
        "Session variability": (
            "Repeated sessions are necessary.\nPre-session state can be informative,\n"
            "but no paper fixes the required\nspacing or calibration duration."
        ),
        "Feedback control": (
            "Sham/playback is feasible but active.\nIt changes learning and brain responses;\n"
            "keep it as a separate causal study."
        ),
    }
    counts = Counter(row["domain"] for row in rows)
    colors = ["#D9EAF7", "#DFF0D8", "#FFF0C7", "#F4DDE8"]
    domains = list(domain_text)
    fig, ax = plt.subplots(figsize=(13.8, 5.2))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for idx, (domain, color) in enumerate(zip(domains, colors)):
        x = idx + 0.06
        box = FancyBboxPatch(
            (x, 0.12), 0.88, 0.72,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.3, edgecolor="#334155", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + 0.44, 0.75, domain, ha="center", va="center", fontsize=11.5, weight="bold")
        ax.text(x + 0.44, 0.65, f"Included evidence: n={counts[domain]}", ha="center", va="center", fontsize=10)
        wrapped = "\n".join(textwrap.fill(line, width=31) for line in domain_text[domain].splitlines())
        ax.text(x + 0.44, 0.40, wrapped, ha="center", va="center", fontsize=8.7, linespacing=1.35)
    ax.text(
        2, 0.95,
        "Evidence map for the prospective self-paced MI collection design",
        ha="center", va="center", fontsize=16, weight="bold",
    )
    ax.text(
        2, 0.03,
        "Interpretation: the literature supports design principles; exact durations, event counts, and risk weights remain preregistered design choices.",
        ha="center", va="center", fontsize=9.5, color="#334155",
    )
    fig.savefig(ROOT / "fig_direct_collection_evidence_map.png", dpi=220, bbox_inches="tight")
    fig.savefig(ROOT / "fig_direct_collection_evidence_map.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    write_outputs(rows)
    plot_evidence_map(rows)
    print(f"Wrote {len(rows)} selected studies and evidence map to {ROOT}")


if __name__ == "__main__":
    main()
