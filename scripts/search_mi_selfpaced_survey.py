"""Europe PMC search for the MI self-paced/asynchronous BCI survey (2026-10-01).

Stores raw hits, a query log and a deduplicated record table.
"""
import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

OUT = Path("data/processed/mi_selfpaced_survey_20261001")
OUT.mkdir(parents=True, exist_ok=True)
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

MI = '(TITLE_ABS:"motor imagery" OR TITLE_ABS:"motor-imagery" OR TITLE_ABS:"sensorimotor rhythm" OR TITLE_ABS:"imagined movement" OR TITLE_ABS:"movement imagination")'
QUERIES = {
    "core_selfpaced": f'{MI} AND (TITLE_ABS:"self-paced" OR TITLE_ABS:"asynchronous" OR TITLE_ABS:"uncued" OR TITLE_ABS:"self-initiated")',
    "idle_noncontrol": f'{MI} AND (TITLE_ABS:"idle state" OR TITLE_ABS:"non-control" OR TITLE_ABS:"no-control" OR TITLE_ABS:"rest state detection" OR TITLE_ABS:"intentional control")',
    "brain_switch": '(TITLE_ABS:"brain switch" OR TITLE_ABS:"brain-switch") AND (TITLE_ABS:EEG OR TITLE_ABS:BCI)',
    "onset_detection": f'{MI} AND (TITLE_ABS:"onset detection" OR TITLE_ABS:"intention detection" OR TITLE_ABS:"continuous detection" OR TITLE_ABS:"false positive")',
    "pseudo_online": f'{MI} AND (TITLE_ABS:"pseudo-online" OR TITLE_ABS:"pseudo online" OR TITLE_ABS:"continuous classification" OR TITLE_ABS:"sliding window")',
    "beta_rebound_mrcp": '(TITLE_ABS:"beta rebound" OR TITLE_ABS:"post-movement beta" OR TITLE_ABS:"movement-related cortical potential") AND (TITLE_ABS:"self-paced" OR TITLE_ABS:asynchronous OR TITLE_ABS:switch OR TITLE_ABS:"onset detection")',
}


def fetch(query, page_size=1000, max_records=3000):
    hits, cursor, total = [], "*", None
    while True:
        params = {"query": query, "format": "json", "resultType": "core",
                  "pageSize": page_size, "cursorMark": cursor}
        url = API + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=120) as r:
            data = json.load(r)
        total = data.get("hitCount", 0)
        batch = data.get("resultList", {}).get("result", [])
        hits.extend(batch)
        nxt = data.get("nextCursorMark")
        if not batch or not nxt or nxt == cursor or len(hits) >= min(total, max_records):
            break
        cursor = nxt
        time.sleep(0.3)
    return total, hits


def slim(h):
    return {
        "id": h.get("id"), "source": h.get("source"), "doi": (h.get("doi") or "").lower(),
        "pmid": h.get("pmid"), "title": h.get("title"), "year": h.get("pubYear"),
        "journal": (h.get("journalInfo") or {}).get("journal", {}).get("title") or h.get("bookOrReportDetails", {}).get("publisher"),
        "authors": h.get("authorString"), "pubtype": (h.get("pubTypeList") or {}).get("pubType"),
        "cited_by": h.get("citedByCount"), "abstract": h.get("abstractText"),
    }


log, records = [], {}
for name, q in QUERIES.items():
    total, hits = fetch(q)
    (OUT / f"europepmc_{name}.json").write_text(json.dumps([slim(h) for h in hits], ensure_ascii=False), encoding="utf-8")
    log.append({"name": name, "database": "Europe PMC", "query": q, "hit_count": total,
                "retrieved": len(hits), "complete": len(hits) >= total, "access_date": date.today().isoformat()})
    for h in hits:
        s = slim(h)
        key = s["doi"] or f"{s['source']}:{s['id']}"
        if key not in records:
            s["queries"] = [name]
            records[key] = s
        elif name not in records[key]["queries"]:
            records[key]["queries"].append(name)
    print(name, total, len(hits))

(OUT / "search_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "records_dedup.json").write_text(json.dumps(list(records.values()), ensure_ascii=False), encoding="utf-8")
print("unique", len(records))
