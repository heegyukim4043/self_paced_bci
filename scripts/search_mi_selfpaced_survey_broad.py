"""Broad Europe PMC + PubMed queries and regex screening for the MI self-paced survey."""
import json, re, time, urllib.parse, urllib.request
from datetime import date
from pathlib import Path
import importlib.util

OUT = Path("data/processed/mi_selfpaced_survey_20261001")
spec = importlib.util.spec_from_file_location("s", "scripts/search_mi_selfpaced_survey.py")

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_Q = {
    "broad_epmc": '("motor imagery" OR "motor-imagery" OR "sensorimotor rhythm" OR "imagined movement") AND ("self-paced" OR "self paced" OR asynchronous OR uncued OR "self-initiated" OR "idle state" OR "non-control state" OR "no-control" OR "brain switch") AND (BCI OR "brain-computer interface" OR "brain computer interface" OR EEG)',
}
PUBMED_Q = {
    "broad_pubmed": '("motor imagery"[tiab] OR "sensorimotor rhythm"[tiab] OR "imagined movement"[tiab] OR "movement imagination"[tiab]) AND ("self-paced"[tiab] OR asynchronous[tiab] OR uncued[tiab] OR "self-initiated"[tiab] OR "idle state"[tiab] OR "non-control"[tiab] OR "brain switch"[tiab] OR "intentional control"[tiab])',
}


def epmc(query, cap=4000):
    hits, cursor = [], "*"
    while True:
        p = {"query": query, "format": "json", "resultType": "core", "pageSize": 1000, "cursorMark": cursor}
        with urllib.request.urlopen(API + "?" + urllib.parse.urlencode(p), timeout=180) as r:
            d = json.load(r)
        b = d.get("resultList", {}).get("result", [])
        hits += b
        n = d.get("nextCursorMark")
        if not b or not n or n == cursor or len(hits) >= min(d["hitCount"], cap):
            return d["hitCount"], hits
        cursor = n
        time.sleep(0.3)


def pubmed_ids(query):
    p = {"db": "pubmed", "term": query, "retmax": 5000, "retmode": "json"}
    with urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(p), timeout=120) as r:
        d = json.load(r)["esearchresult"]
    return int(d["count"]), d["idlist"]


def slim(h):
    return {"id": h.get("id"), "source": h.get("source"), "doi": (h.get("doi") or "").lower(),
            "pmid": h.get("pmid"), "title": h.get("title"), "year": h.get("pubYear"),
            "journal": ((h.get("journalInfo") or {}).get("journal") or {}).get("title"),
            "authors": h.get("authorString"), "cited_by": h.get("citedByCount"),
            "abstract": h.get("abstractText"), "pubtype": (h.get("pubTypeList") or {}).get("pubType")}


log = json.loads((OUT / "search_log.json").read_text(encoding="utf-8"))
records = {(r["doi"] or f"{r['source']}:{r['id']}"): r for r in json.loads((OUT / "records_dedup.json").read_text(encoding="utf-8"))}

for name, q in EPMC_Q.items():
    total, hits = epmc(q)
    log.append({"name": name, "database": "Europe PMC", "query": q, "hit_count": total, "retrieved": len(hits),
                "complete": len(hits) >= total, "access_date": date.today().isoformat()})
    for h in hits:
        s = slim(h); k = s["doi"] or f"{s['source']}:{s['id']}"
        records.setdefault(k, {**s, "queries": []})["queries"].append(name)
    print(name, total, len(hits))

pm_missing = []
for name, q in PUBMED_Q.items():
    total, ids = pubmed_ids(q)
    have = {r.get("pmid") for r in records.values()}
    miss = [i for i in ids if i not in have]
    log.append({"name": name, "database": "PubMed", "query": q, "hit_count": total, "retrieved": len(ids),
                "not_in_epmc_set": len(miss), "complete": len(ids) >= total, "access_date": date.today().isoformat()})
    for k, r in records.items():
        if r.get("pmid") in set(ids):
            r["queries"].append(name)
    pm_missing += miss
    print(name, total, len(ids), "missing from EPMC set:", len(miss))

# fetch missing PubMed records via Europe PMC by EXT_ID
for pmid in pm_missing:
    t, h = epmc(f"EXT_ID:{pmid} AND SRC:MED", cap=1)
    if h:
        s = slim(h[0]); k = s["doi"] or f"MED:{pmid}"
        records.setdefault(k, {**s, "queries": []})["queries"].append("broad_pubmed")
    time.sleep(0.2)

MI = re.compile(r"motor[- ]imager|sensorimotor rhythm|imagined (hand |foot |limb )?movement|movement imagination|kinesthetic imager|\bMI\b|\bSMR\b|brain[- ]switch", re.I)
SP = re.compile(r"self[- ]paced|asynchron|uncued|self[- ]initiat|idle|non[- ]?control|no[- ]control|brain[- ]switch|intentional control|onset detection|continuous (detection|classification|control)|pseudo[- ]online|false (positive|alarm)", re.I)
for r in records.values():
    txt = (r.get("title") or "") + " " + (r.get("abstract") or "")
    r["regex_mi"] = bool(MI.search(txt)); r["regex_sp"] = bool(SP.search(txt))
    r["screen1"] = r["regex_mi"] and r["regex_sp"]

(OUT / "search_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "records_dedup.json").write_text(json.dumps(list(records.values()), ensure_ascii=False), encoding="utf-8")
print("unique", len(records), "screen1 pass", sum(r["screen1"] for r in records.values()))
