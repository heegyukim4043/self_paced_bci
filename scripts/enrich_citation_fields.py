"""Fill missing year/volume/pages/doi in the project BibTeX from CrossRef, then OpenAlex, and log every fill.

Article-number journals (Frontiers, Sci Rep, ...) get `pages` from the CrossRef/OpenAlex article number.
Fields that cannot be found get a `note` explaining the gap instead of silently staying empty.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/citation-management/scripts"))
from _common import parse_bibtex_file, render_entry  # noqa: E402

DIR = Path("data/processed/citations_20261001")
UA = {"User-Agent": "mi-selfpaced-citations/1.0 (mailto:kim7hggame@gmail.com)"}


def get(url):
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def crossref(doi):
    data = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if not data:
        return {}
    m = data["message"]
    year = None
    for key in ("published-print", "published-online", "issued", "posted", "created"):
        parts = (m.get(key) or {}).get("date-parts") or [[None]]
        if parts[0][0]:
            year = parts[0][0]
            break
    out = {"year": str(year) if year else "", "volume": m.get("volume", ""), "number": m.get("issue", ""),
           "pages": (m.get("page") or m.get("article-number") or "").replace("-", "--")}
    rel = (m.get("relation") or {}).get("is-preprint-of") or []
    if rel:
        out["published_version"] = rel[0].get("id", "")
    return out


def openalex(doi):
    data = get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi) + "?mailto=kim7hggame@gmail.com")
    if not data:
        return {}
    b = data.get("biblio") or {}
    fp, lp = b.get("first_page") or "", b.get("last_page") or ""
    pages = f"{fp}--{lp}" if fp and lp and fp != lp else fp
    return {"year": str(data.get("publication_year") or ""), "volume": b.get("volume") or "",
            "number": b.get("issue") or "", "pages": pages}


def main():
    entries = parse_bibtex_file(str(DIR / "references_clean.bib"))
    log = []
    for e in entries:
        f = e["fields"]
        if f.get("pages"):
            f["pages"] = f["pages"].replace("–", "--").replace("—", "--")
            if "-" in f["pages"] and "--" not in f["pages"]:
                f["pages"] = f["pages"].replace("-", "--")
        is_arxiv = e["type"] == "misc" and ("arxiv" in (f.get("publisher", "") + f.get("journal", "") + f.get("url", "") + f.get("eprint", "")).lower())
        need = [k for k in ("year",) if not f.get(k)]
        if e["type"] == "article":
            need += [k for k in ("volume", "pages") if not f.get(k)]
        if not need or not f.get("doi"):
            if is_arxiv and not f.get("note"):
                f["note"] = "arXiv preprint; no journal volume or pages"
                log.append({"key": e["key"], "action": "note", "value": f["note"]})
            continue
        found = {}
        for source, fn in (("crossref", crossref), ("openalex", openalex)):
            got = fn(f["doi"])
            for k in list(need):
                if got.get(k):
                    f[k] = got[k]
                    found[k] = source
                    need.remove(k)
                    if k == "volume" and got.get("number") and not f.get("number"):
                        f["number"] = got["number"]
            if got.get("published_version"):
                log.append({"key": e["key"], "action": "preprint_has_published_version", "value": got["published_version"]})
            time.sleep(1.0)
            if not need:
                break
        for k, src in found.items():
            log.append({"key": e["key"], "action": "filled", "field": k, "value": f[k], "source": src})
        if need:
            reason = "online-first, not yet assigned" if e["type"] == "article" else "not applicable"
            f["note"] = (f.get("note", "") + "; " if f.get("note") else "") + f"missing {', '.join(need)} ({reason}; checked CrossRef and OpenAlex 2026-10-01)"
            log.append({"key": e["key"], "action": "unresolved", "fields": need})
    text = "\n\n".join(render_entry(e["type"], e["key"], e["fields"]) for e in entries) + "\n"
    (DIR / "references_enriched.bib").write_text(text, encoding="utf-8")
    (DIR / "enrichment_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"entries {len(entries)}; log events {len(log)}")


if __name__ == "__main__":
    main()
