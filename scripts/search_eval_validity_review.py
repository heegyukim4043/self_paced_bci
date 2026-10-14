"""Search for the evaluation-validity literature review (EEG/BCI decoding: leakage, autocorrelation, nulls, CV)."""
import json, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

OUT = Path("data/processed/lit_review_eval_validity_20261001")
SRC = OUT / "sources"
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
NEURO = '(TITLE_ABS:EEG OR TITLE_ABS:electroencephalograph* OR TITLE_ABS:MEG OR TITLE_ABS:"brain-computer interface" OR TITLE_ABS:BCI OR TITLE_ABS:neuroimaging OR TITLE_ABS:"neural decoding" OR TITLE_ABS:fMRI)'
Q = {
  "leakage_split": NEURO + ' AND (TITLE_ABS:"data leakage" OR TITLE_ABS:"information leakage" OR TITLE_ABS:"double dipping" OR TITLE_ABS:"circular analysis" OR TITLE_ABS:"data partitioning" OR TITLE_ABS:"subject-wise" OR TITLE_ABS:"cross-subject split" OR TITLE_ABS:"leave-one-subject-out")',
  "autocorr_inflation": NEURO + ' AND (TITLE_ABS:autocorrelation OR TITLE_ABS:"temporal correlation" OR TITLE_ABS:"block design" OR TITLE_ABS:"temporal dependence") AND (TITLE_ABS:decoding OR TITLE_ABS:classification OR TITLE_ABS:"cross-validation") AND (TITLE_ABS:inflat* OR TITLE_ABS:overestimat* OR TITLE_ABS:bias* OR TITLE_ABS:confound* OR TITLE_ABS:spurious)',
  "timeseries_null": '(TITLE_ABS:"circular shift" OR TITLE_ABS:"circular time shift" OR TITLE_ABS:"block permutation" OR TITLE_ABS:"surrogate data" OR TITLE_ABS:"session permutation" OR TITLE_ABS:"nonsense correlation") AND (TITLE_ABS:neural OR TITLE_ABS:neuron* OR TITLE_ABS:EEG OR TITLE_ABS:decoding OR TITLE_ABS:spike*) AND (TITLE_ABS:null OR TITLE_ABS:permutation OR TITLE_ABS:significance OR TITLE_ABS:"false positive")',
  "chance_level": NEURO + ' AND (TITLE_ABS:"chance level" OR TITLE_ABS:"chance-level" OR TITLE_ABS:"permutation test") AND (TITLE_ABS:classification OR TITLE_ABS:decoding OR TITLE_ABS:classifier)',
  "cv_uncertainty": NEURO + ' AND TITLE_ABS:"cross-validation" AND (TITLE_ABS:"small sample" OR TITLE_ABS:"error bars" OR TITLE_ABS:variance OR TITLE_ABS:pitfall* OR TITLE_ABS:caveat* OR TITLE_ABS:guideline*)',
  "bci_eval_metrics": '(TITLE_ABS:"brain-computer interface" OR TITLE_ABS:BCI) AND (TITLE_ABS:"performance evaluation" OR TITLE_ABS:"performance measurement" OR TITLE_ABS:"reporting" OR TITLE_ABS:"pseudo-online" OR TITLE_ABS:"evaluation metric*") AND (TITLE_ABS:asynchronous OR TITLE_ABS:"self-paced" OR TITLE_ABS:"motor imagery" OR TITLE_ABS:"false positive")',
}

def epmc(q, cap=1500):
    hits, cur = [], "*"
    while True:
        p = {"query": q, "format": "json", "resultType": "core", "pageSize": 1000, "cursorMark": cur}
        d = json.load(urllib.request.urlopen(API + "?" + urllib.parse.urlencode(p), timeout=180))
        b = d.get("resultList", {}).get("result", []); hits += b
        n = d.get("nextCursorMark")
        if not b or not n or n == cur or len(hits) >= min(d["hitCount"], cap):
            return d["hitCount"], hits
        cur = n; time.sleep(.3)

def slim(h):
    return {"source": h.get("source"), "id": h.get("id"), "doi": (h.get("doi") or "").lower(), "pmid": h.get("pmid"),
            "title": re.sub(r"<[^>]+>", "", h.get("title") or ""), "year": h.get("pubYear"),
            "journal": ((h.get("journalInfo") or {}).get("journal") or {}).get("title"),
            "authors": h.get("authorString"), "cited_by": h.get("citedByCount") or 0,
            "abstract": re.sub(r"<[^>]+>", "", h.get("abstractText") or "")}

log, recs = [], {}
for name, q in Q.items():
    total, hits = epmc(q)
    (SRC / f"europepmc_{name}.json").write_text(json.dumps([slim(h) for h in hits], ensure_ascii=False), encoding="utf-8")
    log.append({"name": name, "database": "Europe PMC", "query": q, "hit_count": total, "retrieved": len(hits),
                "complete": len(hits) >= total, "date": date.today().isoformat()})
    for h in hits:
        s = slim(h); k = s["doi"] or f"{s['source']}:{s['id']}"
        recs.setdefault(k, {**s, "queries": []})["queries"].append(name)
    print(name, total, len(hits))

AX = {"arxiv_eeg_leakage": '(abs:EEG OR abs:"brain-computer") AND (abs:leakage OR abs:"data partitioning" OR abs:autocorrelation OR abs:"temporal correlation") AND (abs:evaluation OR abs:"cross-validation" OR abs:split)'}
ns = "{http://www.w3.org/2005/Atom}"
for name, q in AX.items():
    root = ET.fromstring(urllib.request.urlopen("http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": 300}), timeout=120).read())
    tot = int(root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults").text); n0 = 0
    for e in root.findall(ns + "entry"):
        n0 += 1
        k = e.find(ns + "id").text
        recs.setdefault(k, {"source": "arXiv", "id": k, "doi": "", "title": " ".join(e.find(ns + "title").text.split()),
            "year": e.find(ns + "published").text[:4], "journal": "arXiv", "cited_by": 0,
            "authors": ", ".join(a.find(ns + "name").text for a in e.findall(ns + "author")),
            "abstract": " ".join(e.find(ns + "summary").text.split()), "queries": []})["queries"].append(name)
    log.append({"name": name, "database": "arXiv", "query": q, "hit_count": tot, "retrieved": n0, "complete": n0 >= tot, "date": date.today().isoformat()})
    print(name, tot, n0)

(OUT / "search_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "records_dedup.json").write_text(json.dumps(list(recs.values()), ensure_ascii=False), encoding="utf-8")
print("unique", len(recs))
